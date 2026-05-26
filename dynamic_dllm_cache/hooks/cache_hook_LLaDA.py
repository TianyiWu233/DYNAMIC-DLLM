import torch
from typing import Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
import types
from dynamic_dllm_cache.cache import DynamicDLLMCache


# ===========================================================================
# Layer budget tracking helpers
# ===========================================================================

def _safe_layer_budget(layer_sim_token, layer_id: int, fallback: float) -> float:
    """Safely index layer_sim_token, falling back to uniform if out of bounds."""
    if layer_sim_token is None:
        return fallback
    num_layers = layer_sim_token.numel()
    if num_layers == 0:
        return fallback
    if layer_id < 0 or layer_id >= num_layers:
        return fallback
    return layer_sim_token[layer_id]


# ===========================================================================
# Module-level helpers (avoid per-call redefinition overhead)
# ===========================================================================

def _project(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_normed = self.attn_norm(x)
    q = self.q_proj(x_normed)
    k = self.k_proj(x_normed)
    v = self.v_proj(x_normed)
    return q, k, v


def _compute_mlp(self, input_x: torch.Tensor) -> torch.Tensor:
    if self._activation_checkpoint_fn is not None:
        x = self._activation_checkpoint_fn(self.ff_norm, input_x)
    else:
        x = self.ff_norm(input_x)
    x, x_up = self.ff_proj(x), self.up_proj(x)
    if self._activation_checkpoint_fn is not None:
        x = self._activation_checkpoint_fn(self.act, x)
    else:
        x = self.act(x)
    x = x * x_up
    return self.ff_out(x)


def _attention_wrapper(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_bias: Optional[torch.Tensor],
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]],
    use_cache: bool,
    q_index: torch.Tensor = None,
) -> torch.Tensor:
    if self._activation_checkpoint_fn is not None:
        att, _ = self._activation_checkpoint_fn(
            self.attention, q, k, v, attention_bias,
            layer_past=layer_past, use_cache=use_cache, q_index=q_index,
        )
    else:
        att, _ = self.attention(
            q, k, v, attention_bias,
            layer_past=layer_past, use_cache=use_cache, q_index=q_index,
        )
    return att


def _get_centered_window_indices(
    center_idx: int, total_len: int, window_size: int
) -> torch.Tensor:
    """Return indices of a window centered at center_idx, clamped to [0, total_len)."""
    if window_size >= total_len:
        return torch.arange(total_len, dtype=torch.long).unsqueeze(0)
    radius = window_size // 2
    start = center_idx - radius
    end = center_idx + radius + 1
    if start < 0:
        end -= start
        start = 0
    if end > total_len:
        start -= (end - total_len)
        end = total_len
    return torch.arange(start, end, dtype=torch.long).unsqueeze(0)


def _compute_sim_token(
    selected: torch.Tensor, selected_cache: torch.Tensor
) -> torch.Tensor:
    """Compute per-token dissimilarity (1 - cosine_similarity) using L2-normalized dot product."""
    selected_norm = F.normalize(selected, p=2, dim=-1)
    selected_cache_norm = F.normalize(selected_cache, p=2, dim=-1)
    sim = (selected_norm * selected_cache_norm).sum(dim=-1)
    return 1.0 - sim


def _compute_sim_token_downsampled(
    selected: torch.Tensor, selected_cache: torch.Tensor, stride: int = 4
) -> torch.Tensor:
    """Compute sim_token on strided (downsampled) tokens for efficiency."""
    selected_ds = selected[:, ::stride, :]
    selected_cache_ds = selected_cache[:, ::stride, :]
    selected_norm = F.normalize(selected_ds, p=2, dim=-1)
    selected_cache_norm = F.normalize(selected_cache_ds, p=2, dim=-1)
    sim = (selected_norm * selected_cache_norm).sum(dim=-1)
    return 1.0 - sim


# ===========================================================================
# window_budget dynamic token selection
# ===========================================================================

def _dynamic_token_selection_wb(
    self,
    feature_cache: "DynamicDLLMCache",
    selected: torch.Tensor,
    selected_cache: torch.Tensor,
    gen_len: int,
    block_start: int,
    bs: int,
    device: torch.device,
    current_step: int,
    refresh_gen: bool,
    fast_update: bool = False,
) -> Optional[torch.Tensor]:
    """
    Dynamic token selection for window_budget mode.

    Returns:
        Index tensor of tokens to update, or None if this step shouldn't do partial update.
    Side effect: updates temp_features[0][0]["sim_token"] for layer budget computation.
    """
    if current_step == 1:
        return None

    select_from = feature_cache.select_from
    window_size = feature_cache.window_size
    layer_budget = feature_cache.layer_budget

    # Compute dissimilarity between current and cached features
    if select_from == "v":
        # selected is already V projection, selected_cache is cached V
        sim_token = _compute_sim_token(selected, selected_cache)
    else:
        sim_token = _compute_sim_token(selected, selected_cache)

    # --- Layer budget tracking (layer 0 is responsible for aggregation) ---
    temp_features_00 = feature_cache.temp_features[0][0]

    if self.layer_id == 0:
        if current_step > 2 and temp_features_00.get("sim_token"):
            layer_sim_token = torch.stack(temp_features_00["sim_token"], dim=0)
            temp_features_00["layer_sim_token"] = torch.softmax(
                layer_sim_token * 200, dim=0
            )
        temp_features_00["sim_token"] = []

    sim_token_mean = sim_token.mean(dim=-1)
    temp_features_00["sim_token"].append(sim_token_mean)

    if fast_update or refresh_gen:
        return None

    # --- Compute layer-specific budget ---
    if current_step > 2:
        total_budget = layer_budget * 32
        layer_sim_token = temp_features_00.get("layer_sim_token")
        if layer_sim_token is not None:
            w = _safe_layer_budget(layer_sim_token, self.layer_id, 1.0 / 32)
            layer_budget = total_budget * w
            layer_budget = layer_budget.clamp(max=1.0)

    # --- Sliding window around most recently decoded token ---
    key_token = int(feature_cache.get_cur_prompt()[-1].item()) - block_start
    window_tokens = _get_centered_window_indices(
        key_token, gen_len, window_size
    ).to(device)
    index = window_tokens

    # --- Vote for most-changed tokens outside the window ---
    vote = sim_token
    if window_tokens.numel() > 0:
        window_expanded = window_tokens.expand(bs, -1)
        mask_window = torch.zeros((bs, gen_len), dtype=torch.bool, device=device)
        mask_window.scatter_(dim=1, index=window_expanded, value=True)
        vote.masked_fill_(mask_window, float("-inf"))

    num_replace = int(layer_budget)
    num_replace = max(0, min(num_replace, gen_len))
    vote_index = torch.topk(vote, k=num_replace, dim=1, sorted=False).indices
    index = torch.cat([index, vote_index], dim=1)

    return index


# ===========================================================================
# Attention and RoPE hooks
# ===========================================================================

def logout_cache_LLaDA(model: nn.Module, tf_block_module_key_name: str) -> None:
    """Restore original functions for transformer blocks, attention, and rotary embeddings."""
    target_module: Optional[nn.ModuleList] = None
    for name, module in model.named_modules():
        if name == tf_block_module_key_name:
            target_module = module
            break
    if target_module is None:
        return
    for tf_block in target_module:
        if hasattr(tf_block, "_old_forward"):
            tf_block.forward = tf_block._old_forward
            delattr(tf_block, "_old_forward")
        if hasattr(tf_block, "_old_attention"):
            tf_block.attention = tf_block._old_attention
            delattr(tf_block, "_old_attention")
        if hasattr(tf_block.rotary_emb, "_old_forward"):
            tf_block.rotary_emb.forward = tf_block.rotary_emb._old_forward
            delattr(tf_block.rotary_emb, "_old_forward")


def register_cache_LLaDA(model: nn.Module, tf_block_module_key_name: str) -> None:
    target_module: Optional[nn.ModuleList] = None
    for name, module in model.named_modules():
        if name == tf_block_module_key_name:
            target_module = module
    for tf_block in target_module:
        setattr(tf_block, "_old_forward", tf_block.forward)
        tf_block.forward = types.MethodType(cache_hook_feature, tf_block)
        setattr(tf_block, "_old_attention", tf_block.attention)
        tf_block.attention = types.MethodType(_attention, tf_block)
        setattr(tf_block.rotary_emb, "_old_forward", tf_block.rotary_emb.forward)
        tf_block.rotary_emb.forward = types.MethodType(
            RoPe_forward, tf_block.rotary_emb
        )


def _attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_bias: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    q_index: torch.Tensor = None,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    B, q_len, C = q.size()
    B, k_len, C = k.size()
    B, v_len, C = v.size()
    dtype = k.dtype
    if self.q_norm is not None and self.k_norm is not None:
        q = self.q_norm(q).to(dtype=dtype)
        k = self.k_norm(k).to(dtype=dtype)
    q = q.view(B, q_len, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
    k = k.view(
        B, k_len, self.config.effective_n_kv_heads, C // self.config.n_heads
    ).transpose(1, 2)
    v = v.view(
        B, v_len, self.config.effective_n_kv_heads, C // self.config.n_heads
    ).transpose(1, 2)
    if layer_past is not None:
        past_key, past_value = layer_past
        k = torch.cat((past_key, k), dim=-2)
        v = torch.cat((past_value, v), dim=-2)
    present = (k, v) if use_cache else None
    query_len, key_len = q.shape[-2], k.shape[-2]
    if self.config.rope:
        q, k = self.rotary_emb(q, k, q_index=q_index)
    if attention_bias is not None:
        attention_bias = self._cast_attn_bias(
            attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
        )
    att = self._scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0 if not self.training else self.config.attention_dropout,
        is_causal=False,
    )
    att = att.transpose(1, 2).contiguous().view(B, q_len, C)
    return self.attn_out(att), present


def RoPe_forward(
    self, q: torch.Tensor, k: torch.Tensor, q_index: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if self.config.rope_full_precision:
        q_, k_ = q.float(), k.float()
    else:
        q_, k_ = q, k
    with torch.autocast(q.device.type, enabled=False):
        query_len, key_len = q_.shape[-2], k_.shape[-2]
        pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
        pos_sin = pos_sin.type_as(q_)
        pos_cos = pos_cos.type_as(q_)
        if q_index is not None:
            bs, _ = q_index.shape
            q_list = []
            for i in range(bs):
                q_i = self.apply_rotary_pos_emb(
                    pos_sin[:, :, q_index[i], :],
                    pos_cos[:, :, q_index[i], :],
                    q_[i].unsqueeze(0),
                )
                q_list.append(q_i)
            q_ = torch.cat(q_list, dim=0)
        else:
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
        k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
    return q_.type_as(q), k_.type_as(k)


# ===========================================================================
# Main cache hook
# ===========================================================================

def cache_hook_feature(
    self,
    x: torch.Tensor,
    attention_bias: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    feature_cache = DynamicDLLMCache()
    feature_cache.update_step(self.layer_id)
    prompt_length = feature_cache.prompt_length
    x_prompt = x[:, :prompt_length, :]
    x_gen = x[:, prompt_length:, :]
    refresh_gen = feature_cache.refresh_gen(layer_id=self.layer_id) or self.layer_id == 0
    refresh_prompt = feature_cache.refresh_prompt(layer_id=self.layer_id) or self.layer_id == 0
    bs, seq_len, dim = x.shape
    gen_len = seq_len - prompt_length

    return _cache_hook_feature_window_budget(
        self, x, x_prompt, x_gen,
        attention_bias, layer_past, use_cache,
        feature_cache, prompt_length, bs, seq_len, dim, gen_len,
        refresh_gen, refresh_prompt,
    )


# ===========================================================================
# window_budget mode
# ===========================================================================

def _cache_hook_feature_window_budget(
    self,
    x: torch.Tensor,
    x_prompt: torch.Tensor,
    x_gen: torch.Tensor,
    attention_bias: Optional[torch.Tensor],
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]],
    use_cache: bool,
    feature_cache: "DynamicDLLMCache",
    prompt_length: int,
    bs: int,
    seq_len: int,
    dim: int,
    gen_len: int,
    refresh_gen: bool,
    refresh_prompt: bool,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    window_size = feature_cache.window_size
    layer_budget = feature_cache.layer_budget
    select_from = feature_cache.select_from
    device = x.device
    current_step = feature_cache.get_current_step(self.layer_id)
    do_partial = (window_size > 0 or layer_budget > 0)
    index_expanded = None  # will be set by partial-update branches
    # ---- Attention section ----
    if refresh_gen and refresh_prompt:
        q, k, v = _project(self, x)
        feature_cache.set_cache(self.layer_id, "kv_cache",
            features={"k": k[:, :prompt_length, :], "v": v[:, :prompt_length, :]},
            cache_type="prompt")
        feature_cache.set_cache(self.layer_id, "kv_cache",
            features={"k": k[:, prompt_length:, :], "v": v[:, prompt_length:, :]},
            cache_type="gen")
        att = _attention_wrapper(self, q, k, v, attention_bias, layer_past, use_cache)
        feature_cache.set_cache(self.layer_id, "attn",
            features=att[:, :prompt_length, :], cache_type="prompt")
        feature_cache.set_cache(self.layer_id, "attn",
            features=att[:, prompt_length:, :], cache_type="gen")
        # Compute downsampled sim_token for future layer-budget allocation
        _compute_sim_for_budget_full_refresh(
            self, feature_cache, select_from, x_gen, v[:, prompt_length:, :], gen_len, current_step
        )

    elif refresh_gen and not refresh_prompt:
        q, k_gen, v_gen = _project(self, x_gen)
        feature_cache.set_cache(self.layer_id, "kv_cache",
            features={"k": k_gen, "v": v_gen}, cache_type="gen")
        kv_cache_prompt = feature_cache.get_cache(self.layer_id, "kv_cache", cache_type="prompt")
        k = torch.cat([kv_cache_prompt["k"], k_gen], dim=1)
        v = torch.cat([kv_cache_prompt["v"], v_gen], dim=1)
        att_gen = _attention_wrapper(self, q, k, v, attention_bias, layer_past, use_cache)
        feature_cache.set_cache(self.layer_id, "attn",
            features=att_gen, cache_type="gen")
        att_prompt_cache = feature_cache.get_cache(self.layer_id, "attn", cache_type="prompt")
        att = torch.cat([att_prompt_cache, att_gen], dim=1)
        # Compute downsampled sim_token for future layer-budget allocation
        _compute_sim_for_budget_full_refresh(
            self, feature_cache, select_from, x_gen, v_gen, gen_len, current_step
        )

    elif not refresh_gen and refresh_prompt:
        q_prompt, k_prompt, v_prompt = _project(self, x_prompt)
        feature_cache.set_cache(self.layer_id, "kv_cache",
            features={"k": k_prompt, "v": v_prompt}, cache_type="prompt")
        kv_cache_gen = feature_cache.get_cache(self.layer_id, "kv_cache", cache_type="gen")
        att_gen_cache = feature_cache.get_cache(self.layer_id, "attn", cache_type="gen")

        if do_partial:
            index = _do_partial_selection(
                self, feature_cache, x_gen, kv_cache_gen,
                select_from, gen_len, prompt_length, bs, dim, device, current_step, refresh_gen
            )
            index = torch.sort(index, dim=1)[0]
            index_expanded = index.unsqueeze(-1).expand(-1, -1, dim)

            x_gen_normed = self.attn_norm(x_gen)
            x_gen_selected = torch.gather(x_gen_normed, dim=1, index=index_expanded)
            q_gen_index = self.q_proj(x_gen_selected)
            k_gen_index = self.k_proj(x_gen_selected)
            kv_cache_gen["k"].scatter_(dim=1, index=index_expanded, src=k_gen_index)
            kv_cache_gen["v"] = self.v_proj(x_gen_normed)

        k = torch.cat([k_prompt, kv_cache_gen["k"]], dim=1)
        v = torch.cat([v_prompt, kv_cache_gen["v"]], dim=1)

        if do_partial:
            q_prompt_gen_index = torch.cat([q_prompt, q_gen_index], dim=1)
            prompt_index = torch.arange(prompt_length).unsqueeze(0).expand(bs, -1).to(device)
            gen_index = index + prompt_length
            att_prompt_gen_index = _attention_wrapper(
                self, q_prompt_gen_index, k, v, attention_bias, layer_past, use_cache,
                q_index=torch.cat([prompt_index, gen_index], dim=1))
            att_prompt = att_prompt_gen_index[:, :prompt_length, :]
            att_gen_index = att_prompt_gen_index[:, prompt_length:, :]
            att_gen_cache.scatter_(dim=1, index=index_expanded, src=att_gen_index)
        else:
            att_prompt = _attention_wrapper(
                self, q_prompt, k, v, attention_bias, layer_past, use_cache,
                q_index=torch.arange(prompt_length).unsqueeze(0).expand(bs, -1))

        feature_cache.set_cache(self.layer_id, "attn",
            features=att_prompt, cache_type="prompt")
        att = torch.cat([att_prompt, att_gen_cache], dim=1)

    else:  # neither refresh
        att_gen_cache = feature_cache.get_cache(self.layer_id, "attn", cache_type="gen")
        kv_cache_prompt = feature_cache.get_cache(self.layer_id, "kv_cache", cache_type="prompt")

        if do_partial:
            kv_cache_gen = feature_cache.get_cache(self.layer_id, "kv_cache", cache_type="gen")
            index = _do_partial_selection(
                self, feature_cache, x_gen, kv_cache_gen,
                select_from, gen_len, prompt_length, bs, dim, device, current_step, refresh_gen
            )
            index = torch.sort(index, dim=1)[0]
            index_expanded = index.unsqueeze(-1).expand(-1, -1, dim)

            x_gen_normed = self.attn_norm(x_gen)
            x_gen_selected = torch.gather(x_gen_normed, dim=1, index=index_expanded)
            q_gen_index = self.q_proj(x_gen_selected)
            k_gen_index = self.k_proj(x_gen_selected)
            kv_cache_gen["k"].scatter_(dim=1, index=index_expanded, src=k_gen_index)
            kv_cache_gen["v"] = self.v_proj(x_gen_normed)

            k = torch.cat([kv_cache_prompt["k"], kv_cache_gen["k"]], dim=1)
            v = torch.cat([kv_cache_prompt["v"], kv_cache_gen["v"]], dim=1)
            att_gen_index = _attention_wrapper(
                self, q_gen_index, k, v, attention_bias, layer_past, use_cache,
                q_index=index + prompt_length)
            att_gen_cache.scatter_(dim=1, index=index_expanded, src=att_gen_index)

        att_prompt_cache = feature_cache.get_cache(self.layer_id, "attn", cache_type="prompt")
        att = torch.cat([att_prompt_cache, att_gen_cache], dim=1)

    x = x + self.dropout(att)

    # ---- MLP section ----
    og_x = x
    x_prompt = x[:, :prompt_length, :]
    x_gen = x[:, prompt_length:, :]

    if refresh_gen and refresh_prompt:
        x = _compute_mlp(self, x)
        feature_cache.set_cache(self.layer_id, "mlp", x[:, prompt_length:, :], cache_type="gen")
        feature_cache.set_cache(self.layer_id, "mlp", x[:, :prompt_length, :], cache_type="prompt")

    elif refresh_gen and not refresh_prompt:
        x_gen = _compute_mlp(self, x_gen)
        feature_cache.set_cache(self.layer_id, "mlp", x_gen, cache_type="gen")
        x_prompt_cache = feature_cache.get_cache(self.layer_id, "mlp", cache_type="prompt")
        x = torch.cat([x_prompt_cache, x_gen], dim=1)

    elif refresh_prompt and not refresh_gen:
        x_gen_cache = feature_cache.get_cache(self.layer_id, "mlp", cache_type="gen")
        if do_partial:
            x_gen_selected = torch.gather(x_gen, dim=1, index=index_expanded)
            x_prompt_gen_index = torch.cat([x_prompt, x_gen_selected], dim=1)
            x_prompt_gen_index = _compute_mlp(self, x_prompt_gen_index)
            x_prompt = x_prompt_gen_index[:, :prompt_length, :]
            x_gen_index = x_prompt_gen_index[:, prompt_length:, :]
            x_gen_cache.scatter_(dim=1, index=index_expanded, src=x_gen_index)
        else:
            x_prompt = _compute_mlp(self, x_prompt)
        feature_cache.set_cache(self.layer_id, "mlp", x_prompt, cache_type="prompt")
        x = torch.cat([x_prompt, x_gen_cache], dim=1)

    else:
        x_gen_cache = feature_cache.get_cache(self.layer_id, "mlp", cache_type="gen")
        if do_partial:
            x_gen_selected = torch.gather(x_gen, dim=1, index=index_expanded)
            x_gen_index = _compute_mlp(self, x_gen_selected)
            x_gen_cache.scatter_(dim=1, index=index_expanded, src=x_gen_index)
        x_prompt_cache = feature_cache.get_cache(self.layer_id, "mlp", cache_type="prompt")
        x = torch.cat([x_prompt_cache, x_gen_cache], dim=1)

    x = self.dropout(x)
    x = og_x + x
    return x, None


# ===========================================================================
# window_budget helper functions
# ===========================================================================

def _compute_sim_for_budget_full_refresh(
    self,
    feature_cache: "DynamicDLLMCache",
    select_from: str,
    x_gen: torch.Tensor,
    v_gen: torch.Tensor,
    gen_len: int,
    current_step: int,
) -> None:
    """
    On full-refresh steps, compute downsampled sim_token to populate
    layer-level similarity stats for future layer_budget allocation.
    Uses stride=4 to reduce overhead.
    """
    if current_step == 1:
        return
    temp_features_00 = feature_cache.temp_features[0][0]

    if select_from == "v":
        selected = v_gen
        cached = feature_cache.get_cache(self.layer_id, "kv_cache", cache_type="gen")
        if isinstance(cached, dict):
            cached = cached.get("v", v_gen)
        else:
            cached = v_gen
    else:
        selected = x_gen
        cached = feature_cache.temp_features[0].get(self.layer_id, {}).get("x")
        if cached is None or (isinstance(cached, int) and cached == 0):
            cached = x_gen

    if gen_len > 4:
        sim_token = _compute_sim_token_downsampled(selected, cached, stride=4)
    else:
        sim_token = _compute_sim_token(selected, cached)

    if self.layer_id == 0:
        if current_step > 2 and temp_features_00.get("sim_token"):
            layer_sim_token = torch.stack(temp_features_00["sim_token"], dim=0)
            temp_features_00["layer_sim_token"] = torch.softmax(
                layer_sim_token * 1e-8, dim=0
            )
        temp_features_00["sim_token"] = []

    sim_token_mean = sim_token.mean(dim=-1)
    temp_features_00["sim_token"].append(sim_token_mean)

    if select_from == "x":
        feature_cache.temp_features[0][self.layer_id]["x"] = x_gen


def _do_partial_selection(
    self,
    feature_cache: "DynamicDLLMCache",
    x_gen: torch.Tensor,
    kv_cache_gen: dict,
    select_from: str,
    gen_len: int,
    prompt_length: int,
    bs: int,
    dim: int,
    device: torch.device,
    current_step: int,
    refresh_gen: bool,
) -> torch.Tensor:
    """
    Perform dynamic token selection for partial cache update.
    Returns index tensor of tokens to recompute.
    Updates temp_features for layer budget tracking as a side effect.
    """
    window_size = feature_cache.window_size
    layer_budget = feature_cache.layer_budget

    # Compute dissimilarity
    x_gen_normed = self.attn_norm(x_gen)
    if select_from == "v":
        v_gen = self.v_proj(x_gen_normed)
        selected = v_gen
        selected_cache = kv_cache_gen["v"]
    else:
        selected = x_gen_normed
        selected_cache = feature_cache.temp_features[0].get(self.layer_id, {}).get("x")
        if selected_cache is None or (isinstance(selected_cache, int) and selected_cache == 0):
            selected_cache = x_gen_normed

    sim_token = _compute_sim_token(selected, selected_cache)

    # --- Layer budget tracking ---
    temp_features_00 = feature_cache.temp_features[0][0]
    sim_token_mean = sim_token.mean(dim=-1)
    temp_features_00["sim_token"].append(sim_token_mean)

    if select_from == "x":
        feature_cache.temp_features[0][self.layer_id]["x"] = x_gen_normed

    # --- Compute layer-specific budget ---
    effective_budget = layer_budget
    if current_step > 2:
        total_budget = layer_budget * 32
        layer_sim_token = temp_features_00.get("layer_sim_token")
        if layer_sim_token is not None:
            w = _safe_layer_budget(layer_sim_token, self.layer_id, 1.0 / 32)
            effective_budget = total_budget * w
            # effective_budget = effective_budget.clamp(max=1.0)

    # --- Sliding window ---
    key_token = int(feature_cache.get_cur_prompt()[-1].item()) - prompt_length
    window_tokens = _get_centered_window_indices(key_token, gen_len, window_size).to(device)
    # Expand to batch dimension for compatibility with vote_index
    index = window_tokens.expand(bs, -1) if window_tokens.numel() > 0 else window_tokens

    # --- Vote outside window ---
    vote = sim_token
    if window_tokens.numel() > 0:
        mask_window = torch.zeros((bs, gen_len), dtype=torch.bool, device=device)
        mask_window.scatter_(dim=1, index=index, value=True)
        vote.masked_fill_(mask_window, float("-inf"))

    num_replace = int(effective_budget)
    num_replace = max(0, min(num_replace, gen_len))
    if num_replace > 0:
        vote_index = torch.topk(vote, k=num_replace, dim=1, sorted=False).indices
        index = torch.cat([index, vote_index], dim=1)
    return index

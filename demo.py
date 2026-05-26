import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import time
from datetime import datetime
from dynamic_dllm_cache.cache import DynamicDLLMCache, DynamicDLLMCacheConfig
from dynamic_dllm_cache.hooks import register_cache_LLaDA
from dataclasses import asdict
from transformers import AutoModel, AutoTokenizer
import torch
from utils import generate, generate_pd

# Configuration parameters
# --- generation mode: "default" or "pd" ---
generate_mode = "pd"  # "default" (fixed schedule) or "pd" (Prediction Dynamics adaptive threshold)
# --- pd mode params ---
pd_mode = 0          # 1 = global scalar threshold, 2 = per-token threshold
pd_threshold = 0.9   # initial threshold
pd_alpha = 0.01      # peak-confidence weight
pd_beta = 0.15       # distribution-shift weight
# --- cache params ---
prompt_interval_steps = 100
gen_interval_steps = 7
window_size = 32
layer_budget = 32
select_from = "v"
# --- common ---
use_cache = True
device = "cuda:0" if torch.cuda.is_available() else "cpu"
gen_length = 256
steps = 256
max_tokens = 2048
block_length = 8
# Load model and tokenizer
model = (
    AutoModel.from_pretrained(
        "/data1/wutianyi/models/LLaDA-8B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    .eval()
)
tokenizer = AutoTokenizer.from_pretrained(
    "/data1/wutianyi/models/LLaDA-8B-Instruct", trust_remote_code=True
)

# Initialize cache
if use_cache:
    DynamicDLLMCache.new_instance(
        **asdict(
            DynamicDLLMCacheConfig(
                prompt_interval_steps=prompt_interval_steps,
                gen_interval_steps=gen_interval_steps,
                window_size=window_size,
                layer_budget=layer_budget,
                select_from=select_from,
                gen_length=gen_length,
            )
        )
    )
    register_cache_LLaDA(model, "model.transformer.blocks")

# Store conversation history
conversation_history = []

def format_time():
    """Return current time in formatted string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def truncate_conversation(history, max_tokens):
    """Truncate conversation history to ensure total tokens do not exceed max_tokens"""
    total_tokens = 0
    truncated_history = []
    for msg in reversed(history):
        tokens = len(tokenizer(msg["content"])["input_ids"])
        if total_tokens + tokens <= max_tokens:
            truncated_history.insert(0, msg)
            total_tokens += tokens
        else:
            break
    return truncated_history


print("*" * 66)
print(
    f"** Answer Length: {gen_length}  | Sampling Steps: {steps}  | Cache Enabled: {use_cache}"
)
if generate_mode == "pd":
    print(f"** Generate Mode: PD  | pd_mode={pd_mode}  | pd_threshold={pd_threshold}  | alpha={pd_alpha}  | beta={pd_beta}")
else:
    print(f"** Generate Mode: Default (fixed schedule)")
print(f"** Cache: Window Size={window_size}  Layer Budget={layer_budget}  Select From={select_from}")
print("*" * 66)

user_input = (
    "Below is an instruction that describes a task. "
    "The response consists of two parts: the answer and the process analysis. "
    "The answer is a number, please ensure that the number of the final answer "
    "follows string \"The answer is:\". "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "Josh decides to try flipping a house. He buys a house for $80,000 and then "
    "puts in $50,000 in repairs. This increased the value of the house by 150%. "
    "How much profit did he make?\n\n"
    "### Response:\n\n"
)

# Record user input time
input_time = format_time()
conversation_history.append({"role": "user", "content": user_input, "time": input_time})

# Truncate conversation history to ensure it does not exceed max token limit
conversation_history = truncate_conversation(conversation_history, max_tokens)

# Apply chat template
formatted_input = tokenizer.apply_chat_template(
    conversation_history, add_generation_prompt=True, tokenize=False
)

# Encode input
input_ids = tokenizer(formatted_input)["input_ids"]
attention_mask = tokenizer(formatted_input)["attention_mask"]
input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
attention_mask = torch.tensor(attention_mask).to(device).unsqueeze(0)

# Reset cache
feature_cache = DynamicDLLMCache()
feature_cache.reset_cache(input_ids.shape[1])
feature_cache.set_parameter((select_from, window_size, layer_budget))
feature_cache.set_cur_prompt(torch.arange(input_ids.shape[1], device=device))

# Generate response
start_time = time.time()
if generate_mode == "pd":
    out, nfe = generate_pd(
        model=model,
        prompt=input_ids,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=0.,
        remasking='low_confidence',
        pd_mode=pd_mode,
        pd_threshold=pd_threshold,
        alpha=pd_alpha,
        beta=pd_beta,
        use_cache=use_cache,
        attention_mask=attention_mask,
    )
    generation_ids = out[:, input_ids.shape[1]:]
else:
    generation_ids = generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        model=model,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
    )
end_time = time.time()

# Decode response
answer = tokenizer.batch_decode(generation_ids, skip_special_tokens=True)[0]
reply_time = format_time()

# Store assistant response
conversation_history.append({"role": "assistant", "content": answer, "time": reply_time})

# Print conversation
print(f"LLaDA ({reply_time}): {answer}")
print(f"Generation Time: {end_time - start_time:.2f} seconds")

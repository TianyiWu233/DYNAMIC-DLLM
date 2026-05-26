# Dynamic-dLLM: Dynamic Cache-Budget and Adaptive Parallel Decoding for Training-Free Acceleration of Diffusion LLMs

This repository is the official PyTorch implementation of [Dynamic-dLLM](https://openreview.net/pdf?id=SdnkB5pGbq) (ICLR 2026).

## Overview

![Motivation of Dynamic-dLLM](assets/motivation_1.png)

(a-b) Layer input similarity and attention output similarity across adjacent denoising steps. Brighter regions indicate higher similarity — most tokens are stable across steps. (c-d) The number of tokens requiring updates varies across steps and layers, motivating layer-specific cache budgets. (e) Existing parallel decoding methods may discard valid candidates due to fixed thresholds.

![Illustration of Dynamic-dLLM](assets/frame_work.jpg)

Dynamic-dLLM consists of two components: **Dynamic Cache Updating (DCU)** reallocates the cache update budget per-layer at each denoising step, and **Adaptive Parallel Decoding (APD)** dynamically adjusts decoding thresholds for all tokens.

## News

- **[2026.01.26]** Paper accepted at **ICLR 2026**.

## Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers accelerate lm-eval datasets
```

### 2. Prepare Model

Download [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) and note its local path.

### 3. Run Demo

Edit `demo.py` to set your model path, then run:

```bash
python demo.py
```

The demo supports two generation modes configured at the top of the file:
- `generate_mode = "default"` — fixed-schedule block-wise masked diffusion
- `generate_mode = "pd"` — Prediction Dynamics adaptive threshold

Cache parameters:
- `window_size` — sliding window size for forced cache refresh (default 32)
- `layer_budget` — per-layer token update budget (default 32)
- `select_from` — feature source for similarity computation: `"v"`

### 4. Evaluation

```bash
python evaluation_script.py -m lm_eval \
  --model LLaDA \
  --model_args "pretrained=/path/to/LLaDA-8B-Instruct,is_feature_cache=True,window_size=32,layer_budget=32,select_from=v,prompt_interval_steps=5000,gen_interval_steps=5000,generate_mode=default" \
  --include_path ./lm_eval_tasks \
  --tasks gsm8k_local \
  --num_fewshot 4 \
  --batch_size 1 \
  --output_path ./gsm8k_log/default_window_budget_v \
  --log_samples \
  --apply_chat_template \
  --fewshot_as_multiturn
```

See `scripts/run_LLaDA_gsm8k_Instruct.sh` for a complete example.

## Project Structure

```
dynamic_dllm/
├── demo.py                         # Interactive/scripted generation demo
├── evaluation_script.py            # lm-eval-harness entry point
├── dynamic_dllm_cache/             # Core caching package
│   ├── cache/
│   │   ├── Cache.py                # DynamicDLLMCache singleton
│   │   └── Config.py               # DynamicDLLMCacheConfig dataclass
│   └── hooks/
│       └── cache_hook_LLaDA.py     # Window-budget cache hooks for LLaDA
├── eval_model/
│   └── LLaDA.py                    # lm-eval model wrapper (registered as "LLaDA")
├── utils/
│   ├── generate_function.py        # generate() and generate_pd() routines
│   └── utils.py                    # set_seed() helper
├── data/                           # Dataset download/verification scripts
├── lm_eval_tasks/                  # Custom lm-eval task definitions
├── metrics/                        # Accuracy and pass@1 computation
├── scripts/                        # Evaluation run scripts
└── assets/                         # Images for documentation
```

## Citation

```bibtex
@inproceedings{dynamic-dllm2026,
  title={Dynamic dLLM: Dynamic Cache-Budget and Adaptive Parallel Decoding for Training-Free Acceleration of Diffusion Large Language Models},
  author={...},
  booktitle={ICLR},
  year={2026}
}
```

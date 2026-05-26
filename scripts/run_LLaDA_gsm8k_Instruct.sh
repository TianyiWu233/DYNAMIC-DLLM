#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../data/env.sh"

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

MODEL="pretrained=/data1/wutianyi/models/LLaDA-8B-Instruct"
GEN_KWARGS="block_length=8,gen_length=256,steps=256,cfg_scale=0.0"
COMMON_ARGS="--model LLaDA --include_path ./lm_eval_tasks --tasks gsm8k_local --num_fewshot 4 --log_samples --apply_chat_template --fewshot_as_multiturn"

export CUDA_VISIBLE_DEVICES=6

echo "=== default generate, window_budget cache (select_from=v) ==="
python evaluation_script.py -m lm_eval \
  ${COMMON_ARGS} \
  --batch_size 1 \
  --output_path ./gsm8k_log/default_window_budget_v \
  --model_args "${MODEL},is_feature_cache=True,
window_size=32,layer_budget=32,select_from=v,prompt_interval_steps=5000,gen_interval_steps=5000,generate_mode=default" \
  --gen_kwargs "${GEN_KWARGS}"

echo "=== Done ==="

# Source this file (`source data/env.sh`) before running any
# scripts/run_LLaDA_*.sh so that `datasets.load_dataset(...)` (called from
# lm_eval) reads from the local cache populated by data/download_all.sh
# instead of re-downloading to `~/.cache/huggingface/datasets`.

LLADA_DATA_ROOT="${LLADA_DATA_ROOT:-/data1/wutianyi/data/llada}"

export HF_HOME="${LLADA_DATA_ROOT}/hf_home"
export HF_DATASETS_CACHE="${LLADA_DATA_ROOT}/hf_datasets"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_ALLOW_CODE_EVAL=1

# ---- HuggingFace mirror & accelerated transfer --------------------------------
# Use the hf-mirror.com mirror (中国大陆友好) and enable hf_transfer for high-
# bandwidth downloads.  These have to be exported BEFORE any Python process
# imports `datasets` / `huggingface_hub`, because huggingface_hub caches the
# endpoint URL on first import.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HF_HUB_ENABLE_EMERGENCY_RETRY="${HF_HUB_ENABLE_EMERGENCY_RETRY:-1}"
export HF_HUB_EMERGENCY_RETRY_WAIT_TIME="${HF_HUB_EMERGENCY_RETRY_WAIT_TIME:-10}"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

echo "[llada/data/env.sh] HF_ENDPOINT       = ${HF_ENDPOINT}"
echo "[llada/data/env.sh] HF_DATASETS_CACHE = ${HF_DATASETS_CACHE}"
echo "[llada/data/env.sh] hf_transfer       = ${HF_HUB_ENABLE_HF_TRANSFER}"

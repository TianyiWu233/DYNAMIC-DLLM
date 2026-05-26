#!/usr/bin/env bash
# One-shot downloader for every dataset referenced by scripts/run_LLaDA_*.sh.
#
# Usage:
#   bash data/download_all.sh                 # download everything
#   bash data/download_all.sh --skip gpqa_main # skip gated GPQA
#   bash data/download_all.sh --only gsm8k bbh # only the listed tasks
#
# Auxiliary env knobs you may set before invoking this script:
#   LLADA_DATA_ROOT  – override the local cache root
#                      (default: /data1/wutianyi/data/llada)
#   HF_TOKEN         – exported automatically as HUGGING_FACE_HUB_TOKEN so the
#                      gated `Idavidrein/gpqa` dataset can be fetched without
#                      a manual `huggingface-cli login`.

set -euo pipefail

LLADA_DATA_ROOT="${LLADA_DATA_ROOT:-/data1/wutianyi/data/llada}"
mkdir -p "${LLADA_DATA_ROOT}/hf_datasets" "${LLADA_DATA_ROOT}/hf_home"

export HF_HOME="${LLADA_DATA_ROOT}/hf_home"
export HF_DATASETS_CACHE="${LLADA_DATA_ROOT}/hf_datasets"
export HF_DATASETS_TRUST_REMOTE_CODE=1
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_BIN="${PYTHON:-python}"

echo "==================================================================="
echo "  HF_HOME           = ${HF_HOME}"
echo "  HF_DATASETS_CACHE = ${HF_DATASETS_CACHE}"
echo "  python            = $(${PY_BIN} -c 'import sys; print(sys.executable)')"
echo "==================================================================="

"${PY_BIN}" "${SCRIPT_DIR}/download_all.py" --root "${LLADA_DATA_ROOT}" "$@"

echo
echo "All requested datasets are cached under:"
echo "  ${HF_DATASETS_CACHE}"
echo "Source data/env.sh before launching scripts/run_LLaDA_*.sh so that"
echo "lm_eval reuses the cache instead of re-downloading."

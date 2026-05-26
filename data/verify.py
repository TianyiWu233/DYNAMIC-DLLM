"""Quick sanity-check that every benchmark referenced by the run scripts
loads from the local HF cache populated by ``download_all.py``.

Usage:
    python data/verify.py
    LLADA_DATA_ROOT=/some/other/path python data/verify.py
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

ROOT = os.environ.get("LLADA_DATA_ROOT", "/data1/wutianyi/data/llada")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(ROOT, "hf_datasets"))
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

CHECKS: list[Tuple[str, str, Optional[str], str, bool]] = [
    ("gsm8k",              "openai/gsm8k",                 "main",                "test",       False),
    ("humaneval",          "openai/openai_humaneval",      None,                  "test",       False),
    ("mbpp",               "google-research-datasets/mbpp", "full",               "test",       False),
    ("bbh",                "SaylorTwift/bbh",              "boolean_expressions", "test",       False),
    ("gpqa_main",          "Idavidrein/gpqa",              "gpqa_main",           "train",      False),
    ("hendrycks_math",     "EleutherAI/hendrycks_math",    "algebra",             "test",       False),
    ("mmlu",               "cais/mmlu",                    "abstract_algebra",    "test",       False),
    ("mmlu_pro",           "TIGER-Lab/MMLU-Pro",           None,                  "test",       False),
    ("longbench_hotpotqa", "Xnhyacinth/LongBench",         "hotpotqa",            "test",       True),
]


def main() -> int:
    import datasets

    print(f"HF_DATASETS_CACHE = {os.environ['HF_DATASETS_CACHE']}\n")
    failures: list[Tuple[str, str]] = []
    for label, repo, cfg, split, trust_remote in CHECKS:
        try:
            ds = datasets.load_dataset(
                path=repo,
                name=cfg,
                split=split,
                trust_remote_code=trust_remote,
            )
            print(f"  OK  {label:25s} (rows={len(ds)})")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {label:25s} {type(e).__name__}: {e}")
            failures.append((label, str(e)))

    print()
    if failures:
        print(f"{len(failures)} dataset(s) failed to load. "
              f"Re-run data/download_all.sh and check HF auth for gated repos.")
        return 1
    print("All datasets loadable from local cache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Download every dataset referenced by scripts/run_LLaDA_*.sh.

All data is materialised into the HuggingFace datasets cache under
``/data1/wutianyi/data/llada/hf_datasets`` so that, at evaluation time, a single
``export HF_DATASETS_CACHE=/data1/wutianyi/data/llada/hf_datasets`` (see
``env.sh``) is enough for ``lm_eval`` (which calls ``datasets.load_dataset``
internally) to find the data offline.

Mapping between ``--tasks`` value used in the shell scripts and the underlying
HuggingFace dataset (resolved from the YAML configs in ``lm_eval/tasks/``):

    --tasks                          dataset_path                     dataset_name(s)
    -------------------------------  -------------------------------  ----------------------------------
    gsm8k                            openai/gsm8k (alias: gsm8k)      main
    humaneval                        openai/openai_humaneval          (default)
    mbpp                             google-research-datasets/mbpp    full
    bbh                              SaylorTwift/bbh                  27 BBH sub-tasks
    gpqa_main_generative_n_shot      Idavidrein/gpqa                  gpqa_main         (GATED)
    minerva_math                     EleutherAI/hendrycks_math        7 math sub-domains
    mmlu_generative                  cais/mmlu                        57 MMLU subjects
    mmlu_pro                         TIGER-Lab/MMLU-Pro               (default)
    longbench_hotpotqa               Xnhyacinth/LongBench             hotpotqa          (trust_remote_code)

Run with ``bash download_all.sh`` (recommended) or ``python download_all.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from typing import Iterable, List, Optional, Tuple

DEFAULT_ROOT = "/data1/wutianyi/data/llada"


def _set_default_env(root: str) -> None:
    """Point HuggingFace caches at ``root`` *before* importing ``datasets``."""
    cache_dir = os.path.join(root, "hf_datasets")
    hf_home = os.path.join(root, "hf_home")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(hf_home, exist_ok=True)

    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")


def _bbh_configs() -> List[str]:
    return [
        "boolean_expressions",
        "causal_judgement",
        "date_understanding",
        "disambiguation_qa",
        "dyck_languages",
        "formal_fallacies",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "logical_deduction_three_objects",
        "movie_recommendation",
        "multistep_arithmetic_two",
        "navigate",
        "object_counting",
        "penguins_in_a_table",
        "reasoning_about_colored_objects",
        "ruin_names",
        "salient_translation_error_detection",
        "snarks",
        "sports_understanding",
        "temporal_sequences",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects",
        "web_of_lies",
        "word_sorting",
    ]


def _hendrycks_math_configs() -> List[str]:
    return [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]


def _mmlu_subjects() -> List[str]:
    return [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology",
        "high_school_statistics", "high_school_us_history",
        "high_school_world_history", "human_aging", "human_sexuality",
        "international_law", "jurisprudence", "logical_fallacies",
        "machine_learning", "management", "marketing", "medical_genetics",
        "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
        "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology",
        "us_foreign_policy", "virology", "world_religions",
    ]


# (label_for_log, dataset_path, dataset_name_or_None, trust_remote_code)
DatasetSpec = Tuple[str, str, Optional[str], bool]


def collect_specs(
    only: Optional[Iterable[str]] = None,
    skip: Optional[Iterable[str]] = None,
) -> List[DatasetSpec]:
    """Return every (label, repo, config, trust_remote_code) tuple to download."""
    specs: List[DatasetSpec] = []

    specs.append(("gsm8k", "openai/gsm8k", "main", False))

    specs.append(("humaneval", "openai/openai_humaneval", None, False))

    specs.append(("mbpp", "google-research-datasets/mbpp", "full", False))

    for cfg in _bbh_configs():
        specs.append((f"bbh/{cfg}", "SaylorTwift/bbh", cfg, False))

    specs.append(("gpqa_main", "Idavidrein/gpqa", "gpqa_main", False))

    for cfg in _hendrycks_math_configs():
        specs.append((f"hendrycks_math/{cfg}", "EleutherAI/hendrycks_math", cfg, False))

    for sub in _mmlu_subjects():
        specs.append((f"mmlu/{sub}", "cais/mmlu", sub, False))

    specs.append(("mmlu_pro", "TIGER-Lab/MMLU-Pro", None, False))

    specs.append(("longbench_hotpotqa", "Xnhyacinth/LongBench", "hotpotqa", True))

    if only:
        only_set = set(only)
        specs = [s for s in specs if s[0].split("/")[0] in only_set]
    if skip:
        skip_set = set(skip)
        specs = [s for s in specs if s[0].split("/")[0] not in skip_set]
    return specs


def download_one(spec: DatasetSpec, cache_dir: str, retries: int = 2) -> Tuple[bool, str]:
    """Force HuggingFace to materialise one dataset config into ``cache_dir``."""
    label, repo, config, trust_remote_code = spec
    import datasets

    last_err = ""
    for attempt in range(1, retries + 2):
        try:
            datasets.load_dataset(
                path=repo,
                name=config,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
            )
            return True, ""
        except Exception as e:  # noqa: BLE001 - we want every failure logged
            last_err = f"{type(e).__name__}: {e}"
            print(f"  [attempt {attempt}/{retries + 1}] {label} failed: {last_err}",
                  flush=True)
            if attempt <= retries:
                time.sleep(3 * attempt)
    return False, last_err


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="Root directory for the local HF cache "
                             "(default: %(default)s)")
    parser.add_argument(
        "--only", nargs="*", default=None,
        choices=[
            "gsm8k", "humaneval", "mbpp", "bbh", "gpqa_main",
            "hendrycks_math", "mmlu", "mmlu_pro", "longbench_hotpotqa",
        ],
        help="Only download these benchmarks.")
    parser.add_argument(
        "--skip", nargs="*", default=None,
        choices=[
            "gsm8k", "humaneval", "mbpp", "bbh", "gpqa_main",
            "hendrycks_math", "mmlu", "mmlu_pro", "longbench_hotpotqa",
        ],
        help="Skip these benchmarks (e.g. gpqa_main if you don't have HF access).")
    parser.add_argument("--retries", type=int, default=2,
                        help="Re-attempt count per dataset on transient failures.")
    args = parser.parse_args(argv)

    _set_default_env(args.root)
    cache_dir = os.environ["HF_DATASETS_CACHE"]
    print(f"[*] HF_HOME           = {os.environ['HF_HOME']}")
    print(f"[*] HF_DATASETS_CACHE = {cache_dir}")

    specs = collect_specs(only=args.only, skip=args.skip)
    print(f"[*] Will fetch {len(specs)} dataset configs in total.\n")

    successes: List[str] = []
    failures: List[Tuple[str, str]] = []
    for i, spec in enumerate(specs, 1):
        label, repo, config, _ = spec
        cfg_part = f" (config={config})" if config else ""
        print(f"[{i:02d}/{len(specs)}] {label} <- {repo}{cfg_part}", flush=True)
        ok, err = download_one(spec, cache_dir, retries=args.retries)
        if ok:
            successes.append(label)
        else:
            failures.append((label, err))
            traceback.print_exc()

    print("\n=== Summary ===")
    print(f"  succeeded : {len(successes)}/{len(specs)}")
    if failures:
        print(f"  failed    : {len(failures)}")
        for label, err in failures:
            print(f"    - {label}: {err}")
        print("\nHints:")
        print("  * `Idavidrein/gpqa` is a GATED dataset on the Hub. Run "
              "`huggingface-cli login` first.")
        print("  * Datasets that require trust_remote_code (e.g. LongBench) "
              "need network access to the *.py loader script.")
        print("  * Re-running this script will resume from the cache.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

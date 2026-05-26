from dataclasses import dataclass


@dataclass
class DynamicDLLMCacheConfig:
    prompt_interval_steps: int = 1
    gen_interval_steps: int = 1
    cfg_interval_steps: int = 1
    window_size: int = 0
    layer_budget: int = 0
    select_from: str = "x"  # "x" (hidden states) or "v" (value projections)
    gen_length: int = 256

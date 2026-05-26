import torch
from collections import defaultdict


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DynamicDLLMCache(metaclass=Singleton):
    gen_interval_steps: int
    prompt_interval_steps: int
    cfg_interval_steps: int
    prompt_length: int
    window_size: int
    layer_budget: int
    select_from: str
    gen_length: int
    __cache: defaultdict
    __step_counter: defaultdict

    @classmethod
    def new_instance(
        cls,
        prompt_interval_steps: int = 1,
        gen_interval_steps: int = 1,
        cfg_interval_steps: int = 1,
        window_size: int = 0,
        layer_budget: int = 0,
        select_from: str = "x",
        gen_length: int = 256,
    ) -> "DynamicDLLMCache":
        ins = cls()
        setattr(ins, "prompt_interval_steps", prompt_interval_steps)
        setattr(ins, "gen_interval_steps", gen_interval_steps)
        setattr(ins, "cfg_interval_steps", cfg_interval_steps)
        setattr(ins, "window_size", window_size)
        setattr(ins, "layer_budget", layer_budget)
        setattr(ins, "select_from", select_from)
        setattr(ins, "gen_length", gen_length)
        ins.init()
        return ins

    def init(self) -> None:
        self.__cache = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        )
        self.__step_counter = defaultdict(lambda: defaultdict(lambda: 0))
        self.__cur_prompt = torch.empty(0, dtype=torch.int)
        self.temp_features = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: 0))
        )

    def reset_cache(self, prompt_length: int = 0) -> None:
        self.init()
        torch.cuda.empty_cache()
        self.prompt_length = prompt_length
        self.cache_type = "no_cfg"

    def set_cache(
        self, layer_id: int, feature_name: str, features: torch.Tensor, cache_type: str
    ) -> None:
        self.__cache[self.cache_type][cache_type][layer_id][feature_name] = {
            0: features
        }

    def get_cache(
        self, layer_id: int, feature_name: str, cache_type: str
    ) -> torch.Tensor:
        output = self.__cache[self.cache_type][cache_type][layer_id][feature_name][0]
        return output

    def update_step(self, layer_id: int) -> None:
        self.__step_counter[self.cache_type][layer_id] += 1

    def get_current_step(self, layer_id: int) -> int:
        return self.__step_counter[self.cache_type][layer_id]

    def refresh_gen(self, layer_id: int = 0) -> bool:
        return (self.current_step - 1) % self.gen_interval_steps == 0

    def refresh_prompt(self, layer_id: int = 0) -> bool:
        return (self.current_step - 1) % self.prompt_interval_steps == 0

    def refresh_cfg(self, layer_id: int = 0) -> bool:
        return (
            self.current_step - 1
        ) % self.cfg_interval_steps == 0 or self.current_step <= 5

    # --- window_budget mode helpers ---

    def set_parameter(self, cache_para: tuple) -> None:
        self.cache_para = cache_para

    def get_parameter(self, layer_id: int) -> tuple:
        return self.cache_para

    def set_cur_prompt(self, cur_prompt: torch.Tensor) -> None:
        if self.__cur_prompt.numel() == 0:
            self.__cur_prompt = cur_prompt
        else:
            self.__cur_prompt = torch.cat([self.__cur_prompt, cur_prompt])

    def get_cur_prompt(self) -> torch.Tensor:
        return self.__cur_prompt

    @property
    def current_step(self) -> int:
        return max(list(self.__step_counter[self.cache_type].values()), default=1)

    def __repr__(self):
        return f"USE DynamicDLLMCache"

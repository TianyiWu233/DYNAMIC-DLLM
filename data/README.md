# `data/` — 一键拉取 `scripts/run_LLaDA_*.sh` 所需的全部测评数据

本目录里的脚本会把 `scripts/` 下所有 `run_LLaDA_*.sh` 用到的 HuggingFace 数据集
全部缓存到 `/data1/wutianyi/data/llada/`，并提供一个供测评脚本 `source` 的环境
变量片段，使得 `lm_eval` 在运行时直接命中本地缓存、不再联网下载。

## 1. 数据集清单

下面 9 类（共 **101** 个 dataset config）覆盖 `scripts/` 中出现过的所有
`--tasks ...` 取值。`dataset_path / dataset_name` 来自 `lm_eval/tasks/<task>/...yaml`。

| `--tasks` 参数 | 实际 HF 仓库 | 子集 / dataset_name | 备注 |
|---|---|---|---|
| `gsm8k` | `openai/gsm8k` | `main` | |
| `humaneval` | `openai/openai_humaneval` | — | |
| `mbpp` | `google-research-datasets/mbpp` | `full` | |
| `bbh`（=`bbh_cot_fewshot_*`，27 项） | `SaylorTwift/bbh` | 27 个子任务 | |
| `gpqa_main_generative_n_shot` | `Idavidrein/gpqa` | `gpqa_main` | **gated**，需先 `huggingface-cli login` |
| `minerva_math` | `EleutherAI/hendrycks_math` | 7 个 (algebra / counting_and_probability / geometry / intermediate_algebra / number_theory / prealgebra / precalculus) | |
| `mmlu_generative` | `cais/mmlu` | 57 个学科 | |
| `mmlu_pro` | `TIGER-Lab/MMLU-Pro` | — | |
| `longbench_hotpotqa` | `Xnhyacinth/LongBench` | `hotpotqa` | 需 `trust_remote_code` |

> Mapping 来源（可自行核对）：
> - `lm_eval/tasks/gsm8k/gsm8k.yaml`
> - `lm_eval/tasks/humaneval/humaneval.yaml`
> - `lm_eval/tasks/mbpp/mbpp.yaml`
> - `lm_eval/tasks/bbh/cot_fewshot/_cot_fewshot_template_yaml`
> - `lm_eval/tasks/gpqa/generative/_gpqa_generative_n_shot_yaml`
> - `lm_eval/tasks/minerva_math/minerva_math_*.yaml`
> - `lm_eval/tasks/mmlu/generative/_default_template_yaml`
> - `lm_eval/tasks/mmlu_pro/_default_template_yaml`
> - `lm_eval/tasks/longbench/hotpotqa.yaml`

## 2. 一键下载

```bash
cd /home/wutianyi/notebooks/llada/dLLM-cache-main

# (一次性) gpqa 是 gated，先做认证：
huggingface-cli login              # 或 export HF_TOKEN=hf_xxx

bash data/download_all.sh
```

下载完成后所有 Arrow 缓存都会落在
`/data1/wutianyi/data/llada/hf_datasets/`，HF Hub 元数据落在
`/data1/wutianyi/data/llada/hf_home/`。

常用选项：

```bash
bash data/download_all.sh --skip gpqa_main           # 跳过 gated
bash data/download_all.sh --only gsm8k bbh           # 只下载这两个基准
LLADA_DATA_ROOT=/data1/wutianyi/data/llada \
HF_TOKEN=hf_xxx bash data/download_all.sh            # 自定义路径 + 透传 token
```

如需手动调用 Python：

```bash
python data/download_all.py --help
python data/download_all.py --root /data1/wutianyi/data/llada
```

## 3. 在测评脚本中复用本地缓存

`scripts/run_LLaDA_*.sh` 调用的是 `datasets.load_dataset(...)`（在
`lm_eval/api/task.py:926`），只要进程的 `HF_DATASETS_CACHE` 指向我们刚才下好
的目录即可。已经准备好可直接 `source` 的环境片段：

```bash
cd /home/wutianyi/notebooks/llada/dLLM-cache-main
source data/env.sh                       # 注入 HF_HOME / HF_DATASETS_CACHE
bash scripts/run_LLaDA_gsm8k_Instruct.sh
```

如果你想让所有 `scripts/run_LLaDA_*.sh` 默认走本地缓存而不每次手动 `source`，
也可以在脚本最前面加一行 `source $(dirname "$0")/../data/env.sh`。

## 4. 数据是否需要二次处理？

**不需要在文件层面做二次处理**。每个任务的 `process_docs / utils.py /
filter_list` 都是 `lm_eval` 在评测的运行时阶段做的（例如 minerva_math 的
`process_docs` 抽 `\\boxed{...}`、GPQA 的随机化选项、HumanEval 的
`build_predictions` 拼测试用例…），它们都吃 HF 原生数据 schema，不要求把数据
预先转成新格式。

下面把每个数据集的 “处理在哪一步发生” 标清楚，方便你 review：

| 数据集 | 是否需要离线二次处理 | 在哪里做 | 入口函数 |
|---|---|---|---|
| `gsm8k` | 否 | 运行时 | `lm_eval/tasks/gsm8k/gsm8k.yaml` 里的 `doc_to_text` + `filter_list`（regex 抽 `#### <ans>`） |
| `humaneval` | 否 | 运行时 | `lm_eval/tasks/humaneval/utils.py: build_predictions` + `pass_at_k` |
| `mbpp` | 否 | 运行时 | `lm_eval/tasks/mbpp/utils.py: pass_at_1` + `list_fewshot_samples` |
| `bbh` | 否 | 运行时 | `_cot_fewshot_template_yaml` 的 `filter_list: get-answer`（regex `the answer is ...`） |
| `gpqa_main_generative_n_shot` | 否 | 运行时 | `lm_eval/tasks/gpqa/generative/utils.py: process_docs`（随机打乱选项） |
| `minerva_math` | 否 | 运行时 | `lm_eval/tasks/minerva_math/utils.py: process_docs / process_results`（抽 `\\boxed{}` + math_verify） |
| `mmlu_generative` | 否 | 运行时 | `_default_template_yaml` 的 `filter_list: get_response` |
| `mmlu_pro` | 否 | 运行时 | `lm_eval/tasks/mmlu_pro/utils.py: doc_to_text / fewshot_to_text` |
| `longbench_hotpotqa` | 否 | 运行时 | `lm_eval/tasks/longbench/metrics.py: get_qa_f1_with_score` |

> 也就是说，本目录只负责**把 HF 原始数据搬下来**；任何“格式转换 / 选项打乱 /
> 答案抽取 / 评分”都由 `lm_eval` 在评测期间自动完成。如果将来有新基准确实需要
> 离线预处理（例如把数据转成自定义 jsonl、生成新的 fewshot 池），可以另外在
> 本目录新建一个 `process_<task>.py`，遵循同样的 “只负责把数据写到
> `${LLADA_DATA_ROOT}/processed/<task>/`” 的约定。

## 5. 文件清单

```
data/
├── README.md           ← 本文件
├── download_all.py     ← Python 下载脚本（核心实现）
├── download_all.sh     ← Bash 入口（设好 env 后调 download_all.py）
├── env.sh              ← 测评前 source 用的环境变量片段
└── verify.py           ← 快速自检：所有数据集都能从本地缓存里 load 吗？
```

## 6. 自检

```bash
source data/env.sh
python data/verify.py
```

输出形如：

```
HF_DATASETS_CACHE = /data1/wutianyi/data/llada/hf_datasets

  OK  gsm8k                     (rows=1319)
  OK  humaneval                 (rows=164)
  OK  mbpp                      (rows=500)
  OK  bbh                       (rows=250)
  OK  gpqa_main                 (rows=448)
  OK  hendrycks_math            (rows=1187)
  OK  mmlu                      (rows=100)
  OK  mmlu_pro                  (rows=12032)
  OK  longbench_hotpotqa        (rows=200)
All datasets loadable from local cache.
```

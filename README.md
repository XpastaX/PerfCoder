**This repository is no longer maintained.**

**Please refer to [PerfCoder](https://github.com/BiSheng-Compiler-Agents/PerfCoder) for the latest updates.**


# PerfCoder

**PerfCoder** is a framework for training large language models to optimize C/C++ code for execution speed. It is built on top of [PIE](https://github.com/LearningOpt/pie), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (for SFT) and [Open-R1 / TRL](https://github.com/huggingface/open-r1) (for GRPO). The framework is composed of three stages:

1. **Data generation** — collect (slow → fast) code pairs from PIE, then use a strong teacher model (Qwen2.5-32B-Instruct) to *extract* category-aware optimization strategies that act as rationales for SFT.
2. **Supervised Fine-Tuning (SFT)** — train PerfCoder (Qwen2.5-Coder / CodeLlama backbones) to first produce optimization *strategies* (`[SUGG/]…[/SUGG]`) and then the *optimized code* (`[OPT/]…[/OPT]`), a Chain-of-Strategy (CoS) format.
3. **GRPO** — on top of the SFT planner (PerfCoder-1.5B), run online RL with TRL+vLLM. A frozen large optimizer (Qwen2.5-32B-Instruct) follows the planner's strategies to produce code, then `gem5` is used as the reward provider (correctness + measured speedup on a simulated Skylake CPU).

> The training of PerfCoder relies on **LLaMA-Factory** (SFT) and **TRL/Open-R1** (GRPO). The code in this repository covers data preparation, custom inference, the GRPO recipe (rewards, optimizer client, gem5 reward server) and the evaluation harness.

---

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [Installation](#installation)
3. [Datasets](#datasets)
4. [Pipeline Overview](#pipeline-overview)
5. [Step 1 — Data Generation](#step-1--data-generation)
6. [Step 2 — SFT Training (PerfCoder)](#step-2--sft-training-perfcoder)
7. [Step 3 — GRPO Training](#step-3--grpo-training)
8. [Inference](#inference)
9. [Evaluation with gem5](#evaluation-with-gem5)
10. [Bash Scripts Index](#bash-scripts-index)
11. [Acknowledgements](#acknowledgements)

---

## Repository Layout

```
PerfCoder/
├── bash/                       # All launch scripts (training / inference / GRPO / gem5 eval)
│   ├── PerfCoder/              #   PerfCoder SFT + 2-step inference + strategy extraction
│   ├── GRPO/                   #   GRPO base-model SFT, vLLM servers, GRPO launch, inference
│   └── gem5/                   #   Speed-test wrappers around speed_test.py
├── data/                       # Datasets and LlamaFactory dataset_info.json
│   ├── PIE/                    #   PIE training/test set + extracted strategies
│   ├── ds_config/              #   DeepSpeed configs (ZeRO-2 / ZeRO-3)
│   └── problem_dict.jsonl      #   PIE problem → (user, [solutions]) index
├── dataset/                    # Released SFT dataset (data_hq_selected_5k.jsonl)
├── prefix/                     # All prompt templates (extraction, follow, classification, …)
│   └── decompose/              #   Strategy-extraction & CoS templates
├── recipes/                    # GRPO YAML config + accelerate / DeepSpeed configs
│   ├── PerfCoder-1.5B/         #   GRPO recipe for the 1.5B planner
│   └── accelerate_configs/
├── script/                     # Data preparation utilities (sample / classify / format …)
├── src/
│   ├── open_r1/                # GRPO trainer (grpo.py, rewards.py, configs.py)
│   └── pie/                    # PIE-specific glue: gem5 reward server, vLLM optimizer client
├── gem5/                       # gem5 simulator wrapper (Docker, Skylake config)
├── utils/                      # I/O helpers (load_data / save_data / read_file / …)
├── extract_refine.py           # ⚙️  Strategy-extraction script (Qwen-32B teacher → suggestions)
├── inference_single.py         # ⚙️  Unified vLLM inference (stage1 / stage2 / 2-step CoS)
├── speed_test.py               # gem5-based speedup benchmarking
├── requirements.txt
└── setup.py
```

---

## Installation

> The code base targets CUDA 12.4 and PyTorch 2.6 (compatible with vLLM 0.8.x). Use a fresh Python 3.11 environment.

### 1. Create the environment

```bash
# with uv (recommended)
uv venv perfcoder --python 3.11 && source perfcoder/bin/activate && uv pip install --upgrade pip

# install GRPO/inference deps
uv pip install vllm==0.8.4
uv pip install setuptools && uv pip install flash-attn --no-build-isolation
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]"
```

### 2. Install LLaMA-Factory (only for SFT)

PerfCoder's SFT stage is launched through LLaMA-Factory. Clone it separately:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[torch,metrics,deepspeed]"
```

Then **symlink** this repo's `data/` into LLaMA-Factory so that the dataset entries in `data/dataset_info.json` are visible to `src/train.py`:

```bash
# inside LLaMA-Factory/
mv data data.bak
ln -s /absolute/path/to/PerfCoder/data ./data
```

### 3. (GRPO only) Install gem5 docker image

The gem5 wrapper under `gem5/` will pull the Skylake docker image automatically the first time `simulator.make()` is invoked (Docker Engine must be installed and accessible to the user). See [`gem5/README.md`](gem5/README.md) for details.

---

## Datasets

| Path | Description |
|------|-------------|
| `data/PIE/full.rar`                                    | Full PIE-derived (src, tgt) pairs with runtime / speedup metadata. |
| `data/PIE/SFT.rar`                                     | Plain (src_code, tgt_code) pairs after de-duplication. |
| `data/PIE/decompose/data_hq_selected_5k.json`          | **5K high-quality** (slow, fast, strategy) triples used to train PerfCoder (CoS). |
| `data/PIE/decompose/data_hq_rand_5k.json`              | 5K random baseline (no strategy curation). |
| `data/PIE/direct.rar` / `data/PIE/decompose/direct/…`  | Direct-optimization SFT data (no strategy), used for the *w/o Strategy* ablation. |
| `data/PIE/test_c20_eval.jsonl`                         | PIE test set, capped at 20 test cases per problem (used by `speed_test.py`). |
| `data/PIE/strategy/strategy_classified_Qwen32B-Inst.json` | Strategies bucketed into 15 categories by `script/classify.py`. |
| `dataset/data_hq_selected_5k.jsonl`                    | Released, ready-to-train CoS dataset (instruction/response). |

The dataset names registered in `data/dataset_info.json` are:
`PIE_SFT_DECOMPOSE_HQ_SELECTED_TOP5K` and `PIE_SFT_DECOMPOSE_HQ_SELECTED_TOP5K_DIRECT`.

> ⚠️ The big files are stored as `*.rar`. Extract them in place before training (e.g. `unrar x data/PIE/full.rar data/PIE/`).

---

## Pipeline Overview

```text
                ┌───────────────────────┐
   PIE raw ───▶ │ script/data_collection│ ───▶ data/PIE/SFT.jsonl + problem_dict.jsonl
                └──────────┬────────────┘
                           │   (collect (src, tgt) per user)
                           ▼
                ┌───────────────────────┐    Qwen2.5-32B-Inst
                │   extract_refine.py    │◀─── teacher LLM (vLLM, fp8)
                │   prefix=extract_stra  │
                └──────────┬────────────┘
                           │   per-pair optimization suggestions
                           ▼
                ┌───────────────────────┐
                │  script/filter_samples │── strategy_all.json
                │  script/classify.py    │── strategy_classified_Qwen32B-Inst.json
                │  script/sample.py      │── data_hq_selected_5k.json (CoS-formatted)
                └──────────┬────────────┘
                           │
                           ▼
              ┌────────────────────────────────┐
              │  SFT  (LLaMA-Factory, DeepSpeed)│  bash/PerfCoder/finetune/all_PerfCoder.sh
              └──────────┬─────────────────────┘
                         │  → ckpt/CoS/PerfCoder-{QC1.5B|QC7b|CL7b}
                         ▼
              ┌────────────────────────────────┐
              │  GRPO   (TRL + vLLM + gem5)    │  bash/GRPO/start_GRPO/*.sh
              │  planner = PerfCoder-1.5B      │
              │  optimizer = Qwen2.5-32B-Inst  │
              │  reward = gem5 speedup×correct │
              └────────────────────────────────┘
```

---

## Step 1 — Data Generation

### 1.1 Collect raw (src, tgt) pairs from PIE

```bash
# Requires the original PIE train/test jsonl + CodeNet metadata under data/raw/
python script/data_collection.py
# → data/PIE/SFT.jsonl, data/problem_dict.jsonl, data/PIE/test_c20.jsonl
```

`script/format_data.py` and `script/fetch_best.py` extend the pairs with cross-user
"best solution" augmentation, producing `data/PIE/full.jsonl`.

### 1.2 Extract optimization strategies with `extract_refine.py`

`extract_refine.py` is the **core data-generation script**. It loads a teacher LLM
(by default Qwen2.5-32B-Instruct quantized to fp8) with vLLM, applies the
template in `prefix/decompose/extract_stra.txt`, and asks the teacher to produce
**category-tagged optimization suggestions** for every (src_code, tgt_code) pair.
The script supports:

* **multi-process / multi-GPU sharding** — one vLLM worker per GPU group of size `--parallel`;
* **resume from cache** — partial outputs are dumped under `cache/<model_name>/temp_cache_process_*.json`;
* **fp8 / bf16** — toggled by `--use_8bit`.

Key arguments:

| Flag | Purpose |
|------|---------|
| `--input_path`            | jsonl with `src_code` / `tgt_code` fields. |
| `--save_path`             | output jsonl (will contain `<model_name>_resp`). |
| `--model_path`            | path to the teacher model (e.g. Qwen2.5-32B-Instruct). |
| `--model_name`            | response-key prefix (also used as cache directory). |
| `--prefix_path`           | prompt template; default `prefix/decompose/extract_stra.txt`. |
| `--replace_key`           | comma-sep `template_var|sample_field` mapping. Default `src_code|src_code,tgt_code|tgt_code`. |
| `--gpu_ids`               | e.g. `0,1,2,3,6,7`. |
| `--parallel`              | TP size per worker. |
| `--use_8bit`              | enable fp8 quantization. |
| `--gpu_memory_utilization`| default `0.85`. |
| `--extract_code`          | if set, also runs `extract_code_chunk()` on the response. |

Reference launcher (already shipped):

```bash
# PIE training pairs (use 6 GPUs, 2-way TP):
bash bash/PerfCoder/strategy_ext/extraction.sh
```

After extraction, regularize / classify / sample to produce the final SFT data:

```bash
# 1. collect raw strategies into one file
python script/filter_samples.py

# 2. bucket strategies into 15 categories with a 32B classifier
python script/classify.py \
    --model_path /path/to/Qwen2.5-32B-Instruct \
    --strategies_path data/PIE/strategy/strategy_all.json \
    --prefix_path prefix/classification.txt \
    --category_path data/PIE/strategy/category.json

# 3. score / dedup / sample 5K HQ pairs (data_hq_selected_5k.json),
#    formatted to LLaMA-Factory instruction/response jsonl
python script/sample.py
```

---

## Step 2 — SFT Training (PerfCoder)

All SFT runs are issued through **LLaMA-Factory** (`src/train.py`). The two LlamaFactory dataset names registered in [`data/dataset_info.json`](data/dataset_info.json) — `PIE_SFT_DECOMPOSE_HQ_SELECTED_TOP5K` and `PIE_SFT_DECOMPOSE_HQ_SELECTED_TOP5K_DIRECT` — point to files under `data/PIE/decompose/{SFT,direct}/`.

### 2.1 Train PerfCoder-7B / CL-7B on PIE (CoS)

Edit the paths at the top of [`bash/PerfCoder/finetune/all_PerfCoder.sh`](bash/PerfCoder/finetune/all_PerfCoder.sh) (`base_model_path`, `gpu_ids`, …) and run:

```bash
bash bash/PerfCoder/finetune/all_PerfCoder.sh
```

This script trains, sequentially, on `HQ-RAND5K` and `HQ-SELECTED-T5K`,
both for `qwen2.5-coder-7b` (`QC7b`) and `CodeLlama-7b-hf` (`CL7b`),
plus the corresponding **direct-optimization** baselines (`*_DIRECT`).

### 2.2 Train the 1.5B planner used by GRPO

The 1.5B planner uses four extra special tokens (`[SUGG/]`, `[/SUGG]`, `[OPT/]`, `[/OPT]`) that delimit the strategy and code blocks:

```bash
bash bash/GRPO/prepare_base_model_SFT.sh
```

This produces `ckpt/CoS/PerfCoder-QC1.5B`, the input checkpoint for the GRPO stage.

---

## Step 3 — GRPO Training

GRPO is launched with TRL + vLLM. **Three side processes must be running before training**:

### 3.1 gem5 reward server (Docker)

```bash
bash bash/GRPO/start_GRPO/start_gem5.sh
# → nohup python src/pie/gem5_launcher.py >logs/gem5_PIE.log 2>&1 &
```

The gem5 reward provider runs the candidate code on a simulated Skylake CPU and returns `(compile_ok, accuracy, runtime)`.

### 3.2 Optimizer vLLM server (Qwen2.5-32B-Instruct)

The planner only emits a *strategy*; a stronger frozen LLM acts as the **optimizer** that turns the strategy into code. Launch it on 4×V100/4×A100:

```bash
bash bash/GRPO/start_GRPO/start_optimizer.sh
# vllm openai server on port 8001, fp16, TP=4
```

### 3.3 TRL inference vLLM server (the planner)

```bash
bash bash/GRPO/start_GRPO/start_trl.sh
# CUDA_VISIBLE_DEVICES=4 trl vllm-serve --model ckpt/CoS/GRPO/PerfCoder-QC1.5B-S5K
```

### 3.4 Launch GRPO

```bash
bash bash/GRPO/start_GRPO/start_GRPO.sh
# → accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
#       --num_processes 2 src/open_r1/grpo.py --config recipes/PerfCoder-1.5B/config.yaml
```

The relevant recipe file lives under `recipes/`:

* [`recipes/PerfCoder-1.5B/config.yaml`](recipes/PerfCoder-1.5B/config.yaml) — main GRPO recipe (`reward_funcs: [cos_optimizer_code]`, points to gem5 + optimizer endpoints).

Reward composition (see `src/open_r1/rewards.py` and `src/pie/cos_optimizer.py`):

```
reward = compile_ok · accuracy · clip(speedup, 0, R_max)
```

> ⚠️ **Remember to terminate gem5, the optimizer server and the TRL server after training.** They are detached processes and will not be cleaned up automatically.

---

## Inference

The unified entry-point is [`inference_single.py`](inference_single.py) — it runs vLLM on one or more GPUs and supports a generic `replace_key` mechanism that fills the prompt template with arbitrary fields of the input jsonl.

### Single-pass (direct optimization or strategy extraction)

```bash
bash bash/PerfCoder/inference/PerfCoder.sh
```

### Two-step CoS inference

The CoS pipeline first asks the planner for strategies (stage 1) and then asks an optimizer LLM to follow them (stage 2):

```bash
# stage 1: planner produces [SUGG/]…[/SUGG]
bash bash/GRPO/inference/perfcoder_stra.sh
# stage 2: 32B optimizer follows each strategy and emits the optimized code
bash bash/GRPO/inference/2step.sh
```

The 2-step pipeline for the regular PerfCoder-7B (without GRPO) lives at:

```bash
bash bash/PerfCoder/inference/2step-extract.sh
bash bash/PerfCoder/inference/2step-follow.sh
```

---

## Evaluation with gem5

Speedup is measured by [`speed_test.py`](speed_test.py), which wraps the gem5 simulator (`gem5/simulator.py`). The wrappers iterate over every `*.jsonl` in a result folder:

```bash
# Evaluate decompose / single-pass results
bash bash/gem5/eval_new.sh

# Evaluate 2-step CoS results
bash bash/gem5/eval_cos.sh
```

The reference test set is `data/PIE/test_c20_eval.jsonl`. Outputs (per-problem accuracy, runtime, speedup, and compilation status) are written to `result/eval/...`.

---

## Bash Scripts Index

| Script | Stage | Purpose |
|--------|-------|---------|
| `bash/PerfCoder/strategy_ext/extraction.sh`        | data-gen | Run `extract_refine.py` on `data/PIE/full.jsonl` & `test_c20_eval.jsonl`. |
| `bash/PerfCoder/finetune/all_PerfCoder.sh`         | SFT | LLaMA-Factory full-FT of QC7b / CL7b on PIE (CoS + Direct). |
| `bash/PerfCoder/inference/PerfCoder.sh`            | inference | Single-pass strategy extraction with PerfCoder-7B. |
| `bash/PerfCoder/inference/2step-extract.sh`        | inference | Stage-1 strategy extraction (Qwen32B-Inst). |
| `bash/PerfCoder/inference/2step-follow.sh`         | inference | Stage-2 strategy following + code generation. |
| `bash/GRPO/prepare_base_model_SFT.sh`              | SFT | Train PerfCoder-1.5B planner with `[SUGG/]` / `[OPT/]` tokens. |
| `bash/GRPO/start_GRPO/start_gem5.sh`               | GRPO | Start gem5 reward server. |
| `bash/GRPO/start_GRPO/start_optimizer.sh`          | GRPO | Start Qwen2.5-32B-Inst optimizer (vLLM OpenAI server, port 8001). |
| `bash/GRPO/start_GRPO/start_trl.sh`                | GRPO | Start TRL vLLM-serve for the planner. |
| `bash/GRPO/start_GRPO/start_GRPO.sh`               | GRPO | Launch GRPO training (`src/open_r1/grpo.py`). |
| `bash/GRPO/inference/perfcoder_stra.sh`            | inference | Stage-1 strategy with the GRPO-trained planner. |
| `bash/GRPO/inference/2step.sh`                     | inference | Stage-2 follow with optimizer (greedy). |
| `bash/gem5/eval_new.sh` / `eval_cos.sh`            | eval | Run `speed_test.py` on every result jsonl in a folder. |

---

## Acknowledgements

PerfCoder's data, evaluation harness and gem5 wrapper are derived from
[**PIE** (Performance-Improving Edits)](https://github.com/LearningOpt/pie). The
SFT stage uses [**LLaMA-Factory**](https://github.com/hiyouga/LLaMA-Factory) and
the GRPO stage is built on top of [**Open-R1**](https://github.com/huggingface/open-r1)
and [**TRL**](https://github.com/huggingface/trl). We thank all the open-source
authors for making this work possible.

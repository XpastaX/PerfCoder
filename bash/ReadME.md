# bash scripts

This directory holds all launch scripts used to reproduce the experiments in the PerfCoder paper. Adjust the paths at the top of each script to match your environment before running.

The SFT stage of PerfCoder is driven by [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory); install it separately and symlink this repo's `data/` folder into LLaMA-Factory's `data/` so that the entries in `data/dataset_info.json` are picked up by `src/train.py`.

## `PerfCoder/`

Scripts for the 7B SFT models reported in the paper (PerfCoder-QC and PerfCoder-CL, on top of Qwen2.5-Coder-7B and CodeLlama-7B respectively):

* `PerfCoder/finetune/all_PerfCoder.sh` — full-fine-tune QC-7B and CL-7B on the PIE high-quality 5K (CoS) and on the random-5K / direct baselines.
* `PerfCoder/strategy_ext/extraction.sh` — call `extract_refine.py` with Qwen2.5-32B-Instruct as the teacher to generate per-pair optimization strategies for the PIE training set.
* `PerfCoder/inference/PerfCoder.sh`    — single-pass inference (CoS-style) with a fine-tuned PerfCoder-7B.
* `PerfCoder/inference/2step-extract.sh` and `2step-follow.sh` — two-stage CoS inference: stage 1 produces the strategies, stage 2 has a stronger optimizer LLM follow them.

## `GRPO/`

Scripts for the GRPO stage that turns PerfCoder-1.5B into the planner used in the paper.

* `GRPO/prepare_base_model_SFT.sh` — SFT a Qwen2.5-Coder-1.5B with the four extra special tokens `[SUGG/] [/SUGG] [OPT/] [/OPT]` to obtain the GRPO base checkpoint.
* `GRPO/start_GRPO/start_gem5.sh`     — launch the gem5 reward server (Docker, Skylake config) used by the `cos_optimizer_code` reward.
* `GRPO/start_GRPO/start_optimizer.sh` — start the frozen Qwen2.5-32B-Instruct optimizer as a vLLM OpenAI-compatible server on port 8001 (4×V100 / 4×A100).
* `GRPO/start_GRPO/start_trl.sh`       — start the TRL `vllm-serve` instance that the planner queries during rollout.
* `GRPO/start_GRPO/start_GRPO.sh`      — launch GRPO training (`src/open_r1/grpo.py` + `recipes/PerfCoder-1.5B/config.yaml`). Run **after** the three side servers above are up.
* `GRPO/inference/perfcoder_stra.sh`   — stage-1 strategy generation with the GRPO-trained planner.
* `GRPO/inference/2step.sh`            — stage-2 strategy-following with the 32B optimizer (greedy).

> ⚠️ The gem5, optimizer and TRL servers are detached processes; remember to `kill` them after training finishes — they will not exit on their own.

## `gem5/`

* `gem5/eval_new.sh` / `gem5/eval_cos.sh` — wrap `speed_test.py` to evaluate every `*.jsonl` produced by the inference scripts on the gem5 (Skylake) simulator. The reference test set is `data/PIE/test_c20_eval.jsonl` (978 programs × 20 test cases, as in PIE).

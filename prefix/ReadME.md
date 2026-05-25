# prefix/

Prompt templates used throughout the PerfCoder pipeline. Every file in this
directory is referenced by either a Python script, a launcher under `bash/`,
or a GRPO recipe under `recipes/`.

| File | Used by | Purpose |
|------|---------|---------|
| `decompose/extract_stra.txt` | `extract_refine.py` (default `--prefix_path`) | Asks Qwen2.5-32B-Instruct to compare a (slow, fast) pair and emit per-pair, category-tagged optimization **strategies**. This is the core teacher prompt used to build the CoS supervision data. |
| `decompose/ext_fo_test.txt`  | `bash/PerfCoder/inference/PerfCoder.sh`, `bash/GRPO/inference/perfcoder_stra.sh` | Inference-time stage-1 prompt for PerfCoder / the GRPO planner: given the source code, emit the strategy block (`[SUGG/]…[/SUGG]`). |
| `extraction.txt`             | `bash/PerfCoder/inference/2step-extract.sh` | Two-step pipeline, stage-1 strategy extraction with a non-CoS model (used to ablate the planner). |
| `follow_template.txt`        | `bash/PerfCoder/inference/2step-follow.sh`, `bash/GRPO/inference/2step.sh` | Two-step pipeline, stage-2 prompt: given the source code and the strategies, ask the optimizer LLM to **follow** them and emit the optimized code. |
| `classification.txt`         | `script/classify.py` (default `--prefix_path`) | Asks the 32B classifier to map every raw strategy into one of the 15 categories defined in `data/PIE/strategy/category.json`. |
| `direct_opt.txt`             | `inference_single.py` (default `--prefix`), `script/format_data.py` | Plain "rewrite this code to run faster" prompt, used by the *direct optimization* (no-strategy) baselines. |
| `base_optimizer.txt`         | `recipes/PerfCoder-1.5B/config.yaml` (`cos_optimizer_prompt_template_file`) | System prompt loaded by `src/pie/cos_optimizer.py` when querying Qwen2.5-32B-Instruct as the strategy follower during GRPO. |

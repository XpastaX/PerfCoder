# script/

Helper scripts used to **build the training data** of PerfCoder. They are not invoked during training itself, only as a one-off preprocessing pipeline (run them in the order below).

| Script | Purpose |
|--------|---------|
| `data_collection.py`       | Build `data/PIE/SFT.jsonl` and `data/problem_dict.jsonl` from the original PIE/CodeNet metadata: collect all (slow → fast) (`src_code`, `tgt_code`) pairs grouped by user/problem. |
| `format_data.py`           | Re-package the raw PIE training set into deduplicated slow/fast pairs (one pair per (problem, user) bucket). |
| `fetch_best.py`            | Augment with cross-user "best solution" pairs — for every problem, link slow submissions to the fastest accepted submission across users. Output: `data/PIE/full.jsonl`. |
| `extract_refine.py` (root) | Call Qwen2.5-32B-Instruct via vLLM with the prompt template `prefix/decompose/extract_stra.txt` to extract per-pair optimization *strategies*. |
| `filter_samples.py`        | Aggregate all raw strategies produced by `extract_refine.py` into `data/PIE/strategy/strategy_all.json`. |
| `classify.py`              | Use Qwen2.5-32B-Instruct + `prefix/classification.txt` to bucket every strategy into one of the **15 categories** in `data/PIE/strategy/category.json`. Output: `strategy_classified_Qwen32B-Inst.json`. |
| `sample.py`                | Score / dedup / **balanced-sample 5K** (slow, fast, strategy) triples across the 15 categories — produces `data/PIE/decompose/data_hq_selected_5k.json`. The matching random baseline `data_hq_rand_5k.json` is also generated here. |

The end product, `dataset/data_hq_selected_5k.jsonl`, is the high-quality CoS dataset *D_b* used to train PerfCoder-QC / PerfCoder-CL in the paper.
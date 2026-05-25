# data/

Static datasets used to (re-)train PerfCoder. The training driver is **LLaMA-Factory** (`src/train.py`), which reads dataset descriptors from `data/dataset_info.json`.

## `dataset_info.json`

Maps logical dataset names to the underlying jsonl files (relative to this directory). The four datasets used by the paper's SFT recipes are:

| Name | File | Use |
|------|------|-----|
| `PIE_SFT_DECOMPOSE_HQ_SELECTED_TOP5K`        | `PIE/decompose/SFT/data_hq_selected_5k.jsonl`  | 5K balanced (slow, fast, strategy) triples — the **D_b** CoS training set used to train PerfCoder-QC / PerfCoder-CL. |
| `PIE_SFT_DECOMPOSE_HQ_SELECTED_TOP5K_DIRECT` | `PIE/decompose/direct/data_hq_selected_5k.jsonl` | The same 5K pairs **without** the strategy block — used for the *w/o Strategy* ablation in Table 3. |

When using LLaMA-Factory, symlink this directory to `LLaMA-Factory/data/`:

```bash
ln -s /absolute/path/to/PerfCoder/data /path/to/LLaMA-Factory/data
```

## `PIE/`

| Path | Description |
|------|-------------|
| `PIE/SFT.rar`                                          | Plain (slow, fast) pairs after deduplication. |
| `PIE/full.rar`                                         | Full PIE-derived pairs with runtime / speedup metadata produced by `script/data_collection.py` + `script/fetch_best.py`. |
| `PIE/direct.rar`                                       | Direct-optimization variant (no strategy field) of the 5K HQ pairs. |
| `PIE/decompose/data_hq_selected_5k.json`               | Raw 5K HQ (slow, fast, strategy) triples — output of `script/sample.py`. |
| `PIE/decompose/data_hq_rand_5k.json`                   | 5K random baseline (no balanced sampling). |
| `PIE/decompose/data_hq.rar`                            | Full HQ archive before sub-sampling. |
| `PIE/decompose/SFT/data_hq_selected_5k.jsonl`          | LLaMA-Factory-formatted version of the HQ 5K. |
| `PIE/decompose/direct/data_hq_selected_5k.jsonl`       | LLaMA-Factory-formatted version of the *direct* (no-strategy) baseline. |
| `PIE/strategy/strategy_all.json`                       | All raw strategies extracted by `extract_refine.py`. |
| `PIE/strategy/strategy_classified_Qwen32B-Inst.json`   | Strategies bucketed into 15 categories by `script/classify.py`. |
| `PIE/strategy/strategy_unrecognized_Qwen32B-Inst.json` | Strategies the classifier could not place into any of the 15 categories. |
| `PIE/strategy/category.json`                           | The 15 category definitions used both at extraction and classification time. |
| `PIE/test_c20_eval.jsonl`                              | PIE test set capped at 20 test cases per problem (978 programs, 41 problems) — the evaluation set used by `speed_test.py` and Table 2 of the paper. |

## `ds_config/`

DeepSpeed configurations consumed by LLaMA-Factory during SFT:

* `d2.json`       — ZeRO-2.
* `d3_offload.json` / `d3_offload_ori.json` — ZeRO-3 with CPU offload (used for full-FT of the 7B backbones in the paper).

## `problem_dict.jsonl`

A flat index mapping each PIE problem id to `{user → list of accepted solutions}`. Built by `script/data_collection.py` from the original CodeNet metadata; consumed by `script/fetch_best.py` to materialise the cross-user "best solution" augmentation.

> ⚠️ Big files are stored as `.rar` archives (`SFT.rar`, `full.rar`, `direct.rar`, `data_hq.rar`). Extract them in place before training, e.g. `unrar x data/PIE/SFT.rar data/PIE/`.

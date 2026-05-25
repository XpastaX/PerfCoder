# Recipes

YAML configs for the **GRPO** stage of PerfCoder.

* [`PerfCoder-1.5B/config.yaml`](PerfCoder-1.5B/config.yaml) — the recipe used in the paper. It selects the `cos_optimizer_code` reward, points at the gem5 reward server (`src/pie/gem5_client.py`) and at the frozen Qwen2.5-32B-Instruct optimizer server (`start_optimizer.sh`), and trains the PerfCoder-1.5B planner with TRL's GRPO trainer.

`accelerate_configs/` and `ds_config/` provide the matching 🤗 Accelerate and DeepSpeed launch configurations:

* `accelerate_configs/zero3.yaml` — the default for the 1.5B planner in the paper (DeepSpeed ZeRO-3 across the training GPUs).
* `accelerate_configs/zero2.yaml`, `ddp.yaml`, `fsdp.yaml`, `ds_config/zero3.json` — alternative launch backends.

To reproduce the GRPO training run from the paper:

```bash
bash bash/GRPO/start_GRPO/start_gem5.sh        # gem5 reward server (Docker)
bash bash/GRPO/start_GRPO/start_optimizer.sh   # Qwen2.5-32B-Instruct optimizer (vLLM)
bash bash/GRPO/start_GRPO/start_trl.sh         # TRL vLLM serve (planner)
bash bash/GRPO/start_GRPO/start_GRPO.sh        # GRPO training
```

`start_GRPO.sh` ultimately calls

```bash
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes 2 src/open_r1/grpo.py \
    --config recipes/PerfCoder-1.5B/config.yaml
```
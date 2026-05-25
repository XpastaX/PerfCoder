# `src/pie/`

Glue code that connects PerfCoder's GRPO trainer (`src/open_r1/grpo.py`) to:

* the **gem5 reward provider** (`gem5_client.py`, `gem5_launcher.py`, `scoring.py`) — wraps the Skylake gem5 simulator under `src/pie/gem5/` so that the GRPO `cos_optimizer_code` reward can compile and benchmark candidate code in a deterministic environment;
* the **frozen optimizer LLM** (`vllm_client.py`, `cos_optimizer.py`) — talks to the Qwen2.5-32B-Instruct vLLM server started by `bash/GRPO/start_GRPO/start_optimizer.sh`, prompting it with the planner's `[SUGG/]…[/SUGG]` block and recovering the optimized code;
* dataset handling for GRPO (`create_dataset.py`).

## Reproducing the paper's GRPO run

The four side processes (gem5 reward server, optimizer vLLM, planner TRL serve, GRPO trainer) are normally launched through the wrappers under `bash/GRPO/start_GRPO/`. A minimal manual recipe is:

```bash
# 0. environment
export PYTHONPATH=src
source path/to/venv/bin/activate

# 1. gem5 reward server (Skylake docker)
nohup python src/pie/gem5_launcher.py >logs/gem5_PIE.log 2>&1 &

# 2. TRL inference server for the planner
CUDA_VISIBLE_DEVICES=4 trl vllm-serve --model ckpt/CoS/PerfCoder-QC1.5B

# 3. Qwen2.5-32B-Instruct optimizer (vLLM OpenAI server)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=^docker0,lo NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_NET_GDR_LEVEL=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen2.5-32B-Instruct \
  --quantization bitsandbytes --load-format bitsandbytes \
  --dtype half --trust-remote-code \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.7 \
  --port 8001 --max-model-len 8192

# 4. GRPO training
NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=5,6 ACCELERATE_LOG_LEVEL=info \
  accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes 2 src/open_r1/grpo.py \
    --config recipes/PerfCoder-1.5B/config.yaml
```

## File map

| File | Role |
|------|------|
| `gem5_launcher.py`   | Entry-point that brings up the gem5 docker container and listens for benchmarking jobs. |
| `gem5_client.py`     | Async client used inside the GRPO reward function to submit candidate code to gem5. |
| `gem5/`              | Vendored gem5 wrapper (Skylake config, simulator API). |
| `scoring.py`         | Aggregates per-test-case results into a single `(compile, accuracy, runtime)` triple. |
| `vllm_client.py`     | Thin async client for the optimizer's OpenAI-compatible vLLM endpoint. |
| `cos_optimizer.py`   | Chain-of-Strategy optimizer: feed the planner's `[SUGG/]` block to the 32B model and recover the rewritten code. |
| `create_dataset.py`  | `load_dataset` / `format_grpo_example` helpers consumed by `src/open_r1/grpo.py`. |

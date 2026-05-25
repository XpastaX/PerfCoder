export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m vllm.entrypoints.openai.api_server \
  --model /path/to/qwen2.5-32b-Inst \
  --dtype float16 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.7 \
  --port 8001 \
  --max-model-len 8192
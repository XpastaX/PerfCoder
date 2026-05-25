# export VLLM_WORKER_MULTIPROC_METHOD=spawn
# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=^docker0,lo
# export NCCL_P2P_LEVEL=NVL
# export NCCL_SHM_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_NET_GDR_LEVEL=0

gpu_ids="0,1"
prefix="prefix/decompose/ext_fo_test.txt"

# model_name="CL7b-HQ-SELECTED-T5K"
# model_path=ckpt/${model_name}

replace_key="src_code|src_code"
model_name=CoS-QC1.5B-S5K
model_path=ckpt/CoS/GRPO/${model_name}
save_path=result/cos/${model_name}_sugg.jsonl
python inference_single.py \
--prefix $prefix \
--parse "" \
--replace_key ${replace_key} \
--model_path $model_path \
--model_name $model_name \
--save_path $save_path \
--gpu_ids $gpu_ids 

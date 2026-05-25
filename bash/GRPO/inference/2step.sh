# export VLLM_WORKER_MULTIPROC_METHOD=spawn
# # export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=^docker0,lo
# export NCCL_P2P_LEVEL=NVL
# export NCCL_SHM_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_NET_GDR_LEVEL=0

gpu_ids="0,1,2,3"
prefix="prefix/follow_template.txt"

# for model_name in PerfCoder-1.5B-GRPO-500 1 #PerfCoder-QC1.5B
# do
model_name=CoS-QC1.5B-S5K-ADV
# model_name=PerfCoder-1.5B-GRPO
replace_key="src_code|src_code,suggestion|${model_name}_resp"
model_path=/path/to/qwen2.5-32b-Inst
save_path=result/cos/2step/${model_name}.jsonl
input_path=result/cos/${model_name}_sugg.jsonl
python inference_single.py \
--prefix $prefix \
--parse "" \
--replace_key ${replace_key} \
--model_path $model_path \
--model_name $model_name \
--save_path $save_path \
--gpu_ids $gpu_ids \
--input_path $input_path \
--use_8bit
# --use_8bit 
# done
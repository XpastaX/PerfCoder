export VLLM_WORKER_MULTIPROC_METHOD=spawn
gpu_ids="0,1,2,3"
prefix="prefix/decompose/ext_fo_test.txt"

replace_key="src_code|src_code"

for model_nickname in QC7b CL7b
do
model_name=${model_nickname}-HQ-SELECTED-T5K-b64
model_path=ckpt/${model_name}
save_path=result/decompose/all/${model_name}.jsonl
python inference_single.py \
--prefix $prefix \
--parse "" \
--replace_key ${replace_key} \
--model_path $model_path \
--model_name $model_name \
--save_path $save_path \
--gpu_ids $gpu_ids
done

for model_nickname in QC7b CL7b
do
model_name=${model_nickname}-HQ-SELECTED-T5K-D-b64
model_path=ckpt/${model_name}
save_path=result/decompose/all/${model_name}.jsonl
python inference_single.py \
--model_path $model_path \
--model_name $model_name \
--save_path $save_path \
--gpu_ids $gpu_ids \
--repetition_penalty 1.05 \
--max_tokens 2000
done


for model_nickname in QC7b CL7b
do
model_name=${model_nickname}-HQ-RAND5K-b64
model_path=ckpt/${model_name}
save_path=result/decompose/all/${model_name}.jsonl
python inference_single.py \
--prefix $prefix \
--parse "" \
--replace_key ${replace_key} \
--model_path $model_path \
--model_name $model_name \
--save_path $save_path \
--gpu_ids $gpu_ids
done
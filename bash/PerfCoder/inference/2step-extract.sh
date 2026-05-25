model_path=/path/to/Qwen_32B_Inst
model_name="Qwen32B-Inst"
save_path=result/2step/strategy/${model_name}_strategies.jsonl
gpu_ids="0,1"
prefix="prefix/extraction.txt"

python inference_single.py \
--prefix $prefix \
--parse "" \
--model_path $model_path \
--model_name $model_name \
--save_path $save_path \
--gpu_ids $gpu_ids \
--use_8bit

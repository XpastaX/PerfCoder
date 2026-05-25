export VLLM_WORKER_MULTIPROC_METHOD=spawn
model_path=/path/to/qwen2.5-32b-Inst
model_name="Qwen_Refine"
input_path="data/PIE/full.jsonl"
save_path=result/strategy/${model_name}_F_RS.jsonl
gpu_ids="0,1,2,3,6,7"
python extract_refine.py \
--model_path $model_path \
--model_name $model_name \
--input_path $input_path \
--save_path $save_path \
--use_8bit \
--parallel 2 \
--gpu_ids $gpu_ids 



model_name="strategy"
input_path="data/PIE/test_c20_eval.jsonl"
save_path=result/strategy/${model_name}_F_RS_test.jsonl
gpu_ids="0,1,2,3,6,7"
python extract_refine.py \
--model_path $model_path \
--model_name $model_name \
--input_path $input_path \
--save_path $save_path \
--use_8bit \
--parallel 2 \
--gpu_ids $gpu_ids 
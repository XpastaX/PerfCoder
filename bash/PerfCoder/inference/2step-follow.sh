export VLLM_WORKER_MULTIPROC_METHOD=spawn
gpu_ids="0,1"
prefix="prefix/follow_template.txt"

model_name=Qwen32B-Inst
model_path=/path/to/Qwen_32B_Inst
input_path=result/2step/strategy/Qwen32B-Inst_strategies.jsonl
# input_path=result/strategy/Qwen_32B_Inst_test_raw_strategy_test.jsonl
replace_key="src_code|src_code,suggestion|Qwen32B-Inst_resp"
save_path=result/2step/${model_name}_${model_name}.jsonl

python inference_single.py \
--prefix $prefix \
--input_path $input_path \
--parse "#### **Optimized Code**:" \
--replace_key ${replace_key} \
--model_path $model_path \
--model_name $model_name \
--save_path $save_path \
--gpu_ids $gpu_ids \
--use_8bit

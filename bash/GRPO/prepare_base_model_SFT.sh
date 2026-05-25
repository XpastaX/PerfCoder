data_SFT=PIE_SFT_DECOMPOSE_HQ_SELECTED_TOP5K
project_root=CodeOpt
trainer_root=/path/to/LLaMA-Factory
ds_config=data/ds_config/d2.json
gpu_ids=0,1,2,3
sepcial_tokens="[SUGG/],[/SUGG],[OPT/],[/OPT]"

cd $trainer_root

model_nickname=QC1.5B
model_name_or_path=/path/to/Qwen2.5-Coder-1.5B
output_dir=ckpt/CoS/PerfCoder-${model_nickname}
deepspeed --include=localhost:$gpu_ids --master_port 10086 \
src/train.py \
  --deepspeed $ds_config \
  --model_name_or_path $model_name_or_path \
  --stage sft \
  --do_train \
  --finetuning_type full \
  --dataset ${data_SFT} \
  --cutoff_len 8192 \
  --max_samples 100000000000 \
  --overwrite_cache \
  --preprocessing_num_workers 16 \
  --output_dir $output_dir \
  --logging_steps 10 \
  --save_strategy no \
  --plot_loss \
  --overwrite_output_dir \
  --save_only_model \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1.0e-5 \
  --num_train_epochs 2.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --new_special_tokens ${sepcial_tokens} \
  --bf16




model_nickname=QC3B
model_name_or_path=/path/to/Qwen2.5-Coder-3B
output_dir=ckpt/CoS/PerfCoder-${model_nickname}
deepspeed --include=localhost:$gpu_ids --master_port 10086 \
src/train.py \
  --deepspeed $ds_config \
  --model_name_or_path $model_name_or_path \
  --stage sft \
  --do_train \
  --finetuning_type full \
  --dataset ${data_SFT} \
  --cutoff_len 8192 \
  --max_samples 100000000000 \
  --overwrite_cache \
  --preprocessing_num_workers 16 \
  --output_dir $output_dir \
  --logging_steps 10 \
  --save_strategy no \
  --plot_loss \
  --overwrite_output_dir \
  --save_only_model \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1.0e-5 \
  --num_train_epochs 2.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --new_special_tokens ${sepcial_tokens} \
  --bf16

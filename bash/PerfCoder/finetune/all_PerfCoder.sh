cd /path/to/LlamaFactory
ds_config=data/ds_config/d3_offload_ori.json
base_model_path=path/to/base/model/collections
gpu_ids=0,1,2,3


model_name=CodeLlama-7b-hf
model_nickname=CL7b
model_name_or_path=${base_model_path}/${model_name}
output_dir=ckpt/${model_nickname}-HQ-RAND5K-b64
deepspeed --include=localhost:$gpu_ids --master_port 10085 \
src/train.py \
  --template llama2 \
  --deepspeed $ds_config \
  --model_name_or_path $model_name_or_path \
  --stage sft \
  --do_train \
  --finetuning_type full \
  --dataset PIE_SFT_DECOMPOSE_HQ_RAND5K \
  --cutoff_len 8192 \
  --max_samples 100000000000 \
  --overwrite_cache \
  --preprocessing_num_workers 16 \
  --output_dir $output_dir \
  --logging_steps 10 \
  --save_strategy epoch \
  --plot_loss \
  --overwrite_output_dir \
  --save_only_model \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2.0e-5 \
  --num_train_epochs 2.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16



model_name=qwen2.5-coder-7b
model_nickname=QC7b
model_name_or_path=${base_model_path}/${model_name}
output_dir=ckpt/${model_nickname}-HQ-RAND5K-b64
deepspeed --include=localhost:$gpu_ids --master_port 10085 \
src/train.py \
  --deepspeed $ds_config \
  --model_name_or_path $model_name_or_path \
  --stage sft \
  --do_train \
  --finetuning_type full \
  --dataset PIE_SFT_DECOMPOSE_HQ_RAND5K \
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
  --gradient_accumulation_steps 16 \
  --learning_rate 1.0e-5 \
  --num_train_epochs 1.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16


model_name=CodeLlama-7b-hf
model_nickname=CL7b
model_name_or_path=${base_model_path}/${model_name}
output_dir=ckpt/${model_nickname}-HQ-SELECTED-T5K-b64
deepspeed --include=localhost:$gpu_ids --master_port 10085 \
src/train.py \
  --template llama2 \
  --deepspeed $ds_config \
  --model_name_or_path $model_name_or_path \
  --stage sft \
  --do_train \
  --finetuning_type full \
  --dataset PIE_SFT_DECOMPOSE_HQ_SELECTED_TOP5K \
  --cutoff_len 8192 \
  --max_samples 100000000000 \
  --overwrite_cache \
  --preprocessing_num_workers 16 \
  --output_dir $output_dir \
  --logging_steps 10 \
  --save_strategy epoch \
  --plot_loss \
  --overwrite_output_dir \
  --save_only_model \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2.0e-5 \
  --num_train_epochs 2.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16



model_name=qwen2.5-coder-7b
model_nickname=QC7b
model_name_or_path=${base_model_path}/${model_name}
output_dir=ckpt/${model_nickname}-HQ-SELECTED-T5K-b64
deepspeed --include=localhost:$gpu_ids --master_port 10085 \
src/train.py \
  --deepspeed $ds_config \
  --model_name_or_path $model_name_or_path \
  --stage sft \
  --do_train \
  --finetuning_type full \
  --dataset PIE_SFT_DECOMPOSE_HQ_SELECTED_TOP5K \
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
  --gradient_accumulation_steps 16 \
  --learning_rate 1.0e-5 \
  --num_train_epochs 1.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16



model_name=qwen2.5-coder-7b
model_nickname=QC7b
model_name_or_path=${base_model_path}/${model_name}
output_dir=ckpt/${model_nickname}-HQ-SELECTED-T5K-D-b64
deepspeed --include=localhost:$gpu_ids --master_port 10085 \
src/train.py \
  --deepspeed $ds_config \
  --model_name_or_path $model_name_or_path \
  --stage sft \
  --do_train \
  --finetuning_type full \
  --dataset PIE_SFT_DECOMPOSE_HQ_SELECTED_TOP5K_DIRECT \
  --cutoff_len 2000 \
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
  --gradient_accumulation_steps 16 \
  --learning_rate 2.0e-5 \
  --num_train_epochs 2.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16



model_name=CodeLlama-7b-hf
model_nickname=CL7b
model_name_or_path=${base_model_path}/${model_name}
output_dir=ckpt/${model_nickname}-HQ-SELECTED-T5K-D-b64
deepspeed --include=localhost:$gpu_ids --master_port 10085 \
src/train.py \
  --template llama2 \
  --deepspeed $ds_config \
  --model_name_or_path $model_name_or_path \
  --stage sft \
  --do_train \
  --finetuning_type full \
  --dataset PIE_SFT_DECOMPOSE_HQ_SELECTED_TOP5K_DIRECT \
  --cutoff_len 2000 \
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
  --gradient_accumulation_steps 16 \
  --learning_rate 2.0e-5 \
  --num_train_epochs 2.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16

# model_path=ckpt/CoS/PerfCoder-QC1.5B
model_path=ckpt/CoS/GRPO/CoS-QC1.5B-S5K
CUDA_VISIBLE_DEVICES=4 trl vllm-serve --model $model_path
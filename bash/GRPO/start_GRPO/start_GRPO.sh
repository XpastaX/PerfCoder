export NCCL_IB_DISABLE=1 
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export ACCELERATE_LOG_LEVEL=info 
export WANDB_DISABLED=1
export no_proxy=127.0.0.1
export PYTHONPATH=src
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# config_path="recipes/CoS-1.5B/selected-5K.yaml"
# config_path="recipes/CoS-1.5B/selected-5K-advanced.yaml"
config_path="recipes/PerfCoder-1.5B/config.yaml"

# torchrun --nproc_per_node=2 \
#     src/open_r1/grpo.py \
#     --config $config_path

accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
--num_processes 2 \
src/open_r1/grpo.py --config $config_path
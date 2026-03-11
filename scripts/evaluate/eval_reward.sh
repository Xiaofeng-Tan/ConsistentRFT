#!/bin/bash
# NCCL environment variables
#export PYTHONPATH=\$PYTHONPATH:$(pwd)
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
#export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
#export NCCL_SOCKET_IFNAME=bond1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export UCX_NET_DEVICES=eth0
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export NCCL_NVLS_ENABLE=0


# Input parameters 
# TODO: Modify these parameters as needed
reward_model_type="all" # "hpsv2", "clip_score" "image_reward", "pick_score", "unified_reward"

# network proxy settings and api url, when the value is "None" it means None
image_reward_http_proxy=None
image_reward_https_proxy=None
pick_score_http_proxy=None
pick_score_https_proxy=None
unified_reward_url=http://127.0.0.1:8080

# reward models paths
hps_path="./pretrained_ckpt/HPS_v2.1_compressed.pt"
hps_clip_path="./pretrained_ckpt/open_clip_pytorch_model.bin"
clip_score_path="hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"
image_reward_path="./pretrained_ckpt/imagerward/ImageReward.pt"
image_reward_med_config="./pretrained_ckpt/imagerward/med_config.json"
unified_reward_default_question_type="semantic"
unified_reward_num_workers=1

# input paths
input_dir="$1"
output_json="${input_dir}/sample_test_mix_0_50/eval_${reward_model_type}_test_mix_00_50.json"
prompt_file="${input_dir}/prompt_test_mix_0_50.json"

# batch size
batch_size=16

CUDA_VISIBLE_DEVICES=$2 torchrun --standalone --nnodes=1 --nproc-per-node=$3 \
    fastvideo/eval/eval_reward.py \
    --batch_size $batch_size \
    --output_json $output_json \
    --prompt_file $prompt_file \
    --reward_model $reward_model_type \
    --hps_path $hps_path \
    --hps_clip_path $hps_clip_path \
    --clip_score_path $clip_score_path \
    --image_reward_path $image_reward_path \
    --image_reward_med_config $image_reward_med_config \
    --image_reward_http_proxy $image_reward_http_proxy \
    --image_reward_https_proxy $image_reward_https_proxy \
    --pick_score_http_proxy $pick_score_http_proxy \
    --pick_score_https_proxy $pick_score_https_proxy \
    --unified_reward_url $unified_reward_url \
    --unified_reward_default_question_type $unified_reward_default_question_type \
    --unified_reward_num_workers $unified_reward_num_workers
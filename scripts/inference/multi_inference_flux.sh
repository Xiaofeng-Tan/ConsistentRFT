#!/bin/bash
# ==============================================================================
# Multi-Checkpoint Inference Script for Flux Models (Full Fine-tuning)
# ==============================================================================
# Description:
#   This script performs batch inference across multiple checkpoint directories.
#   It automatically discovers and processes all valid checkpoints, skipping
#   already-processed checkpoints.
#
# Usage:
#   bash scripts/inference/multi_inference_flux.sh <checkpoint_base_dir> <gpu_ids> <num_gpus>
#
# Arguments:
#   $1 - Base directory containing checkpoint subdirectories (e.g., ./outputs/exp1)
#   $2 - CUDA visible devices (e.g., "0,1,2,3")
#   $3 - Number of GPUs per node (e.g., 4)
#
# Example:
#   bash scripts/inference/multi_inference_flux.sh ./outputs/grpo_exp 0,1,2,3 4
# ==============================================================================

set -e  # Exit on error

# ==============================================================================
# NCCL Configuration
# ==============================================================================
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=eth0
export UCX_NET_DEVICES=eth0
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export NCCL_NVLS_ENABLE=0

# ==============================================================================
# Environment Setup
# ==============================================================================
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# ==============================================================================
# Inference Parameters
# ==============================================================================
FLUX_BASELINE_MODEL_DIR="./pretrained_ckpt/flux"
BASE_CKPT_DIR="$1"
MIX_SAMPLING_STEPS=0
TOTAL_SAMPLING_STEPS=50
PROMPT_TYPE="test"
PROMPT_FILE="./data/prompts_${PROMPT_TYPE}.txt"

# ==============================================================================
# Argument Validation
# ==============================================================================
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Error: Missing required arguments."
    echo "Usage: bash $0 <checkpoint_base_dir> <gpu_ids> <num_gpus>"
    exit 1
fi

if [ ! -d "$BASE_CKPT_DIR" ]; then
    echo "Error: Checkpoint base directory does not exist: $BASE_CKPT_DIR"
    exit 1
fi

if [ ! -f "$PROMPT_FILE" ]; then
    echo "Error: Prompt file does not exist: $PROMPT_FILE"
    exit 1
fi

# ==============================================================================
# Main Inference Loop
# ==============================================================================
echo "============================================================"
echo "Starting Multi-Checkpoint Inference"
echo "============================================================"
echo "Base checkpoint directory: $BASE_CKPT_DIR"
echo "GPU devices: $2"
echo "Number of GPUs: $3"
echo "Prompt file: $PROMPT_FILE"
echo "============================================================"

for ckpt_dir in "$BASE_CKPT_DIR"/checkpoint-*; do
    # Validate checkpoint directory
    if [ -d "$ckpt_dir" ] && [ -f "${ckpt_dir}/diffusion_pytorch_model.safetensors" ]; then
        output_dir="${ckpt_dir}/sample_${PROMPT_TYPE}_mix_${MIX_SAMPLING_STEPS}_${TOTAL_SAMPLING_STEPS}"
        output_json="${ckpt_dir}/prompt_${PROMPT_TYPE}_mix_${MIX_SAMPLING_STEPS}_${TOTAL_SAMPLING_STEPS}.json"

        # Skip if already processed
        if [ -f "$output_json" ]; then
            echo "[SKIP] Already processed: $ckpt_dir"
            continue
        fi

        echo "------------------------------------------------------------"
        echo "[PROCESSING] Checkpoint: $ckpt_dir"
        echo "------------------------------------------------------------"
        
        CUDA_VISIBLE_DEVICES="$2" torchrun \
            --standalone \
            --nnodes=1 \
            --nproc-per-node="$3" \
            fastvideo/sample/sample_flux.py \
            --model_path "${ckpt_dir}/diffusion_pytorch_model.safetensors" \
            --prompts_file "$PROMPT_FILE" \
            --output_dir "$output_dir" \
            --output_json "$output_json" \
            --seed 617 \
            --mix_sampling_steps "$MIX_SAMPLING_STEPS" \
            --total_sampling_steps "$TOTAL_SAMPLING_STEPS" \
            --flux_baseline_model_dir "$FLUX_BASELINE_MODEL_DIR"
            # --baseline  # Uncomment to use baseline model
        
        echo "[DONE] Checkpoint: $ckpt_dir"
    else
        echo "[SKIP] No valid model file: $ckpt_dir"
    fi
done

echo "============================================================"
echo "Multi-Checkpoint Inference Complete"
echo "============================================================"

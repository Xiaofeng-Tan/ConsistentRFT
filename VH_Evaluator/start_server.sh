#!/bin/bash
# Usage: bash start_server.sh [NUM_GPUS] [PORT]
#   NUM_GPUS: number of GPUs for tensor parallelism (default: 8)
#   PORT:     server port (default: 8000)

NUM_GPUS=${1:-8}
PORT=${2:-8000}
LOCAL_IP=127.0.0.1

# Build CUDA_VISIBLE_DEVICES string: 0,1,...,NUM_GPUS-1
GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))

echo "Starting VLM server on ${LOCAL_IP}:${PORT} with ${NUM_GPUS} GPUs (${GPU_IDS})"

CUDA_VISIBLE_DEVICES=${GPU_IDS} vllm serve ../pretrained_ckpt/Qwen2.5-VL-72B-Instruct \
    --host ${LOCAL_IP} \
    --trust-remote-code \
    --served-model-name QwenVL \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size ${NUM_GPUS} \
    --limit-mm-per-prompt '{"image": 99}' \
    --port ${PORT}

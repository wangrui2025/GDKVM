#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export CUDA_VISIBLE_DEVICES=4,5

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(($RANDOM % 40000 + 20000))

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1

export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

IFS=',' read -ra _cuda_devices <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#_cuda_devices[@]}

echo "🚀 Starting training on ${NUM_GPUS} GPUs (IDs: ${CUDA_VISIBLE_DEVICES}) on Port ${MASTER_PORT}..."
echo "📂 Project Root: ${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

uv run torchrun \
  --standalone \
  --nproc_per_node=${NUM_GPUS} \
  train.py \
  hydra.run.dir=outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}
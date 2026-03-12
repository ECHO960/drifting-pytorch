#!/usr/bin/env bash
# Single-node multi-GPU training
# Usage: bash scripts/train_1node.sh configs/imagenet_l2.yaml [--resume checkpoints/last.pt]

set -e

CONFIG=${1:-configs/imagenet_l2.yaml}
shift || true   # remaining args forwarded to train.py

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Launching on ${NGPU} GPUs with config: ${CONFIG}"

torchrun \
    --standalone \
    --nproc_per_node=${NGPU} \
    train.py --config "${CONFIG}" "$@"

#!/usr/bin/env bash
# Single-node MNIST training for quick sanity checks.
#
# Usage:
#   bash scripts/train_mnist.sh
#   bash scripts/train_mnist.sh --resume checkpoints/mnist/last.pt

set -e

CONFIG=${1:-configs/mnist.yaml}
shift || true

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

echo "[mnist] gpus=${NGPU} master_port=${MASTER_PORT} config=${CONFIG}"

torchrun \
    --nproc_per_node  ${NGPU} \
    --master_port     ${MASTER_PORT} \
    "${PROJECT_ROOT}/train.py" --config "${CONFIG}" "$@"

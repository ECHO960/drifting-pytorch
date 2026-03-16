#!/usr/bin/env bash
# Multi-node training via torchrun (NCCL backend).
# Run this script on EVERY node with matching NNODES / NODE_RANK.
#
# Required env vars (set before calling or export in your cluster job):
#   MASTER_ADDR  — hostname/IP of rank-0 node
#   MASTER_PORT  — free port (e.g. 29500)
#   NNODES       — total number of nodes
#   NODE_RANK    — this node's rank (0-based)
#
# Example (2 nodes, 8 GPUs each):
#   # On node 0:
#   MASTER_ADDR=node0 MASTER_PORT=29500 NNODES=2 NODE_RANK=0 \
#       bash scripts/train_multinode.sh configs/imagenet_l2.yaml
#
#   # On node 1:
#   MASTER_ADDR=node0 MASTER_PORT=29500 NNODES=2 NODE_RANK=1 \
#       bash scripts/train_multinode.sh configs/imagenet_l2.yaml

set -e

CONFIG=${1:-configs/imagenet_l2.yaml}
shift || true

MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-54993}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
NGPU_PER_NODE=${NGPU_PER_NODE:-$(python -c "import torch; print(torch.cuda.device_count())")}

echo "node ${NODE_RANK}/${NNODES}, ${NGPU_PER_NODE} GPUs, master=${MASTER_ADDR}:${MASTER_PORT}"

torchrun \
    --nproc_per_node=${NGPU_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train.py --config "${CONFIG}" "$@"

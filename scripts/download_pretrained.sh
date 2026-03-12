#!/usr/bin/env bash
# Download pretrained DiT weights for warm-start initialization.
# Weights are from the original DiT repo (Peebles & Xie 2022).
#
# Usage: bash scripts/download_pretrained.sh

set -e
mkdir -p checkpoints/pretrained

BASE="https://dl.fbaipublicfiles.com/DiT/models"

echo "Downloading DiT-B/2 (256x256)..."
wget -q --show-progress -O checkpoints/pretrained/DiT-B-2-256x256.pt \
    "${BASE}/DiT-B-2-256x256.pt"

echo "Downloading DiT-L/2 (256x256)..."
wget -q --show-progress -O checkpoints/pretrained/DiT-L-2-256x256.pt \
    "${BASE}/DiT-L-2-256x256.pt"

echo "Downloading DiT-XL/2 (256x256)..."
wget -q --show-progress -O checkpoints/pretrained/DiT-XL-2-256x256.pt \
    "${BASE}/DiT-XL-2-256x256.pt"

echo "Done. Weights saved to checkpoints/pretrained/"
echo ""
echo "To use in config, set:  model.pretrained: checkpoints/pretrained/DiT-L-2-256x256.pt"

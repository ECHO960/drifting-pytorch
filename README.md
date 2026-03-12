# Generative Modeling via Drifting

> ⚠️ **Work in Progress** — implementation is complete but not yet validated experimentally. Results may differ from the paper.

Unofficial PyTorch reproduction of [**Generative Modeling via Drifting**](https://arxiv.org/abs/2602.04770) (Deng et al., arXiv 2026), with a [DiT](https://arxiv.org/abs/2212.09748) backbone and support for both image generation and robot action generation.


---

## Repo Structure

```
├── configs/
│   ├── imagenet_b2.yaml     # DiT-B/2, ImageNet latent space
│   ├── imagenet_l2.yaml     # DiT-L/2, ImageNet latent space
│   └── robotics.yaml        # small DiT, robot action space
├── kernels/
│   ├── base.py              # abstract DriftKernel
│   ├── exponential.py       # k(x,y) = exp(−‖φ(x)−φ(y)‖ / τ)
│   └── gaussian.py          # k(x,y) = exp(−‖φ(x)−φ(y)‖² / 2τ²)
├── losses/
│   └── drifting.py          # compute_V(), drifting_loss(), CFG variant
├── models/
│   ├── dit.py               # DiT backbone (pretrained-weight compatible)
│   ├── vae.py               # SD VAE wrapper (encode/decode only)
│   └── feature_encoder.py   # DINOv2 / Identity encoder registry
├── data/
│   ├── imagenet.py          # ClassStratifiedSampler + ImageNet loader
│   ├── robotics.py          # Diffusion Policy Zarr loader
│   └── README.md            # download instructions
├── train.py                 # torchrun DDP entry point
├── sample.py                # 1-NFE inference
└── evaluate.py              # FID evaluation (clean-fid)
```

---

## Installation

```bash
git clone https://github.com/ECHO960/drifting-pytorch.git
cd drifting-pytorch
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.2, CUDA 12.1.

---

## Data

See [`data/README.md`](data/README.md) for full download instructions.

**ImageNet** — download from [image-net.org](https://www.image-net.org/download.php) and place under `data/imagenet/`:
```
data/imagenet/
├── train/   # 1000 synset folders
└── val/
```

**Robotics (Diffusion Policy Zarr)** — example with PushT:
```bash
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zarr.zip
unzip pusht.zarr.zip -d data/robotics/
```

**Pretrained DiT weights** (optional warm-start):
```bash
bash scripts/download_pretrained.sh
# → checkpoints/pretrained/DiT-{B,L,XL}-2-256x256.pt
```

SD VAE and DINOv2 are downloaded automatically on first run.

---

## Training

### Single node (all GPUs)
```bash
bash scripts/train_1node.sh configs/imagenet_l2.yaml
```

### Multi-node
```bash
# Run on every node (set env vars accordingly)
MASTER_ADDR=node0 MASTER_PORT=29500 NNODES=2 NODE_RANK=0 \
    bash scripts/train_multinode.sh configs/imagenet_l2.yaml

MASTER_ADDR=node0 MASTER_PORT=29500 NNODES=2 NODE_RANK=1 \
    bash scripts/train_multinode.sh configs/imagenet_l2.yaml
```

### Resume
```bash
bash scripts/train_1node.sh configs/imagenet_l2.yaml --resume checkpoints/imagenet_l2/last.pt
```

### Smoke test (CPU, 2 steps)
```bash
torchrun --standalone --nproc_per_node=1 train.py \
    --config configs/imagenet_b2.yaml --smoke-test
```

---

## Sampling

```bash
python sample.py \
    --config  configs/imagenet_l2.yaml \
    --ckpt    checkpoints/imagenet_l2/last.pt \
    --n       16 \
    --class-id 207 \
    --out     samples/
```

---

## Evaluation (FID)

```bash
python evaluate.py \
    --config configs/imagenet_l2.yaml \
    --ckpt   checkpoints/imagenet_l2/last.pt \
    --n      50000 \
    --out    eval_samples/
```

---

## Kernel Modularity

The drift kernel is fully pluggable. To swap kernel or encoder, change two lines in the yaml:

```yaml
encoder:
  type: dinov2_l    # dinov2_b | dinov2_l | dinov2_s | identity

kernel:
  type: exponential # exponential | gaussian
  tau: 1.0
```

For robotics, `encoder.type: identity` computes the kernel directly in action space without any vision encoder.

To implement a custom kernel, subclass `DriftKernel`:

```python
from kernels.base import DriftKernel

class MyKernel(DriftKernel):
    def pairwise_weights(self, x, y):
        # x: [N, *], y: [M, *]  →  return [N, M]
        ...
```

---

## Key Design Choices

- **Anti-symmetry by construction**: `V_{p,q} = −V_{q,p}` holds because positive and negative terms are subtracted symmetrically — no special loss term needed.
- **DiT pretrained warm-start**: timestep embedding is retained (fixed at t=0) so official DiT weights load cleanly via `model.pretrained` in config.
- **Class-stratified batching**: N_pos ≥ 64 per class per batch is critical (FID degrades 75% with N_pos=1). `ClassStratifiedSampler` guarantees this.
- **Frozen DINOv2**: kernel similarity is computed in DINOv2 feature space, never fine-tuned.


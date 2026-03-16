"""
Generate samples from a trained Drifting model.

Usage:
    python sample.py --config configs/imagenet_l2.yaml \
                     --ckpt   checkpoints/imagenet_l2/last.pt \
                     --n      16 \
                     --class-id 207 \
                     --out    samples/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torchvision.utils as vutils
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from models.dit import DiT
from models.vae import VAEWrapper


@torch.no_grad()
def sample(
    model: DiT,
    vae: VAEWrapper | None,
    n: int,
    class_id: int | None,
    device: torch.device,
    latent_size: int = 32,
    in_channels: int = 4,
) -> torch.Tensor:
    """
    Single-step generation: x = f_θ(ε).

    Returns:
        images: [n, 3, H, W] in [-1, 1] if vae provided, else raw latents
    """
    model.eval()

    eps = torch.randn(n, in_channels, latent_size, latent_size, device=device)
    labels = None
    if class_id is not None:
        labels = torch.full((n,), class_id, dtype=torch.long, device=device)

    x = model(eps, label=labels)   # [n, C, H, W]  — 1 NFE

    if vae is not None:
        images = vae.decode(x)   # [n, 3, 256, 256]
        return images.clamp(-1, 1)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--ckpt",     type=str, required=True)
    parser.add_argument("--n",        type=int, default=16, help="number of samples")
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--out",      type=str, default="samples")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(args.config)

    # Load model
    model = DiT.from_config(cfg.model).to(device)
    state = torch.load(args.ckpt, map_location=device)
    key   = "ema" if "ema" in state else "model"
    model.load_state_dict(state[key], strict=False)
    model.eval()

    # Load VAE (ImageNet only)
    vae = None
    if cfg.data.type == "imagenet":
        vae = VAEWrapper.from_config(cfg.vae).to(device)

    # Generate
    images = sample(
        model, vae,
        n=args.n,
        class_id=args.class_id,
        device=device,
        latent_size=cfg.model.get("input_size", 32),
        in_channels=cfg.model.in_channels,
    )

    # Save
    Path(args.out).mkdir(parents=True, exist_ok=True)
    # Normalize [-1,1] → [0,1] for saving
    grid = vutils.make_grid((images + 1) / 2, nrow=4, padding=2)
    vutils.save_image(grid, os.path.join(args.out, "samples.png"))
    print(f"Saved {args.n} samples to {args.out}/samples.png")


if __name__ == "__main__":
    main()

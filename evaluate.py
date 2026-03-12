"""
FID / IS evaluation for the Drifting model.

Generates 50k samples (or as specified) and computes FID against
real ImageNet statistics using clean-fid.

Usage:
    python evaluate.py --config configs/imagenet_l2.yaml \
                       --ckpt   checkpoints/imagenet_l2/last.pt \
                       --n      50000 \
                       --batch  128 \
                       --out    eval_samples/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torchvision.utils as vutils
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models.dit import DiT
from models.vae import VAEWrapper
from sample import sample as generate_batch


def generate_all(
    model, vae, n_total, batch_size, class_ids, device, latent_size, in_channels, out_dir
):
    """Generate n_total images, saving them to out_dir as individual PNGs."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    generated = 0
    pbar = tqdm(total=n_total, desc="Generating")

    while generated < n_total:
        bs = min(batch_size, n_total - generated)
        cid = class_ids[generated % len(class_ids)] if class_ids else None
        imgs = generate_batch(
            model, vae, bs, cid, device, latent_size, in_channels
        )
        imgs = ((imgs + 1) / 2).clamp(0, 1)  # → [0,1]
        for i, img in enumerate(imgs):
            vutils.save_image(img, os.path.join(out_dir, f"{generated + i:06d}.png"))
        generated += bs
        pbar.update(bs)

    pbar.close()
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    type=str, required=True)
    parser.add_argument("--ckpt",      type=str, required=True)
    parser.add_argument("--n",         type=int, default=50_000)
    parser.add_argument("--batch",     type=int, default=128)
    parser.add_argument("--out",       type=str, default="eval_samples")
    parser.add_argument("--dataset",   type=str, default="imagenet_train",
                        help="clean-fid dataset name for reference stats")
    parser.add_argument("--seed",      type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = OmegaConf.load(args.config)

    # Load model (use EMA weights)
    model = DiT.from_config(cfg.model).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state.get("ema", state.get("model")), strict=False)
    model.eval()

    vae = None
    if cfg.data.type == "imagenet":
        vae = VAEWrapper.from_config(cfg.vae).to(device)

    # Generate all samples
    class_ids = list(range(1000)) if cfg.data.type == "imagenet" else None
    n = generate_all(
        model, vae,
        n_total=args.n,
        batch_size=args.batch,
        class_ids=class_ids,
        device=device,
        latent_size=cfg.model.get("input_size", 32),
        in_channels=cfg.model.in_channels,
        out_dir=args.out,
    )
    print(f"Generated {n} images in {args.out}/")

    # FID via clean-fid
    try:
        from cleanfid import fid
        score = fid.compute_fid(args.out, dataset_name=args.dataset, dataset_split="train")
        print(f"FID: {score:.3f}")
    except ImportError:
        print("clean-fid not installed. Run: pip install clean-fid")
    except Exception as e:
        print(f"FID computation failed: {e}")


if __name__ == "__main__":
    main()

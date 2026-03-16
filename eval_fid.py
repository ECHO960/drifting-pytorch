"""
FID evaluation and image sampling for trained Drifting checkpoints.

Usage:
    # Evaluate a single checkpoint
    python eval_fid.py --ckpt checkpoints/imagenet_l2/last.pt --config configs/imagenet_l2.yaml

    # Multiple checkpoints (prints FID table)
    python eval_fid.py --ckpt checkpoints/imagenet_l2/ckpt_epoch0009.pt \\
                               checkpoints/imagenet_l2/ckpt_epoch0018.pt \\
                       --config configs/imagenet_l2.yaml

Options:
    --n-samples     Number of images to generate for FID (default 10000)
    --batch-size    Generation batch size per GPU (default 64)
    --cfg-scale     CFG guidance scale alpha (default 2.0)
    --out-dir       Where to save samples and results (default eval_out/)
    --no-fid        Skip FID, only save image grid
    --n-grid        Images in the grid (default 64)
    --seed          RNG seed (default 42)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torchvision.utils as vutils
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).parent))
from models.dit import DiT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_dist():
    if "RANK" not in os.environ:
        return 0, 1, 0          # single-GPU fallback
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def is_main(rank): return rank == 0


def load_ema(ckpt_path: str, model: DiT, device):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    key   = "ema" if "ema" in state else "model"
    model.load_state_dict(state[key])
    step  = state.get("step", "?")
    return model.to(device).eval(), step


@torch.no_grad()
def generate_batch(model, n: int, num_classes: int, cfg_scale: float,
                   image_size: int, device, dtype):
    labels = torch.randint(0, num_classes, (n,), device=device)
    eps    = torch.randn(n, 3, image_size, image_size, device=device, dtype=dtype)
    alpha  = torch.full((n,), cfg_scale, device=device, dtype=dtype)
    imgs   = model(eps, y=labels, alpha=alpha)   # [N, 3, H, W]  in [-1, 1]
    return imgs.float().clamp(-1, 1)


def denorm(t: torch.Tensor) -> torch.Tensor:
    """[-1,1] → [0,1]"""
    return (t + 1) / 2


# ---------------------------------------------------------------------------
# Real image extraction (for FID reference)
# ---------------------------------------------------------------------------

def _extract_real_images(cfg, n: int, image_size: int, out_dir: Path):
    import glob, torchvision.transforms as T
    from torchvision.transforms.functional import to_pil_image
    from data.imagenet import ParquetImageNet

    files = sorted(glob.glob(os.path.join(cfg.data.root, "data", "train-*.parquet")))
    ds    = ParquetImageNet(files, image_size)
    step  = max(1, len(ds) // n)
    saved = 0
    for i in range(0, len(ds), step):
        if saved >= n:
            break
        img, _ = ds[i]
        pil = to_pil_image((img * 0.5 + 0.5).clamp(0, 1))
        pil.save(out_dir / f"{saved:06d}.png")
        saved += 1
    print(f"  extracted {saved} real images")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(ckpt_path: str, cfg, args, rank: int, world_size: int, device):
    # ---- Build model ----
    from omegaconf import OmegaConf as OC
    model_cfg = OC.merge(cfg.model, {"cfg_dropout_prob": cfg.cfg.get("uncond_prob", 0.1)})
    model = DiT.from_config(model_cfg)
    model, step = load_ema(ckpt_path, model, device)

    dtype = torch.bfloat16 if cfg.training.precision == "bf16" else torch.float32

    tag     = Path(ckpt_path).stem
    out_dir = Path(args.out_dir) / tag
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {tag}  (step {step}) ===")

    # ---- Image grid ----
    if is_main(rank):
        with torch.autocast("cuda", dtype=dtype):
            grid_imgs = generate_batch(
                model, args.n_grid, cfg.model.num_classes,
                args.cfg_scale, cfg.model.input_size, device, dtype
            )
        grid = vutils.make_grid(denorm(grid_imgs), nrow=8, padding=2)
        vutils.save_image(grid, out_dir / "grid.png")
        print(f"  grid saved → {out_dir / 'grid.png'}")

    if args.no_fid:
        return

    # ---- Generate images for FID ----
    n_per_rank  = args.n_samples // world_size
    fid_dir     = out_dir / "fid_imgs"
    if is_main(rank):
        fid_dir.mkdir(exist_ok=True)

    if world_size > 1:
        dist.barrier()

    generated = 0
    img_idx   = rank * n_per_rank
    bs        = args.batch_size
    while generated < n_per_rank:
        n = min(bs, n_per_rank - generated)
        with torch.autocast("cuda", dtype=dtype):
            imgs = generate_batch(
                model, n, cfg.model.num_classes,
                args.cfg_scale, cfg.model.input_size, device, dtype
            )
        imgs_u8 = (denorm(imgs) * 255).byte().cpu()
        for img in imgs_u8:
            from torchvision.transforms.functional import to_pil_image
            to_pil_image(img).save(fid_dir / f"{img_idx:06d}.png")
            img_idx += 1
        generated += n

    if world_size > 1:
        dist.barrier()

    # ---- FID (rank 0 only) ----
    if is_main(rank):
        from cleanfid import fid

        # Extract real images from parquet on first call; reuse across checkpoints.
        real_dir = Path(args.out_dir) / "real_imgs"
        if not real_dir.exists() or len(list(real_dir.glob("*.png"))) < args.n_samples:
            print(f"  extracting {args.n_samples} real images → {real_dir}")
            real_dir.mkdir(exist_ok=True)
            _extract_real_images(cfg, args.n_samples, cfg.model.input_size, real_dir)

        score = fid.compute_fid(
            str(fid_dir),
            fdir2=str(real_dir),
            mode="clean",
            num_workers=8,
        )
        print(f"  FID = {score:.2f}  (n={args.n_samples}, cfg={args.cfg_scale})")
        with open(out_dir / "fid.txt", "w") as f:
            f.write(f"step={step}  FID={score:.2f}  n={args.n_samples}  cfg={args.cfg_scale}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       nargs="+", required=True)
    parser.add_argument("--config",     required=True)
    parser.add_argument("--n-samples",  type=int,   default=10_000)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--cfg-scale",  type=float, default=1.0)
    parser.add_argument("--out-dir",    default="eval_out")
    parser.add_argument("--no-fid",     action="store_true")
    parser.add_argument("--n-grid",     type=int,   default=64)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = OmegaConf.load(args.config)

    rank, world_size, local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")

    for ckpt in args.ckpt:
        evaluate(ckpt, cfg, args, rank, world_size, device)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

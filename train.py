"""
Drifting Model — Training Entry Point.

Launch with torchrun:
    # Single node
    torchrun --nproc_per_node=8 train.py --config configs/imagenet_l2.yaml

    # Multi-node (see scripts/train_multinode.sh)
    torchrun --nproc_per_node=8 --nnodes=N --node_rank=R \
             --master_addr=ADDR --master_port=PORT \
             train.py --config configs/imagenet_l2.yaml
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from models.dit import DiT
from models.vae import VAEWrapper
from models.feature_encoder import build_encoder
from kernels import build_kernel
from losses import drifting_loss, drifting_loss_cfg


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def setup_dist():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_dist():
    dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def log(rank: int, msg: str):
    if is_main(rank):
        print(msg, flush=True)


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p.detach(), alpha=1 - decay)


def save_checkpoint(out_dir: str, step: int, model, ema, optimizer, rank: int):
    if not is_main(rank):
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    state = {
        "step":      step,
        "model":     model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "ema":       ema.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, os.path.join(out_dir, f"ckpt_{step:07d}.pt"))
    torch.save(state, os.path.join(out_dir, "last.pt"))   # always keep latest
    log(rank, f"[ckpt] saved step {step}")


def load_checkpoint(path: str, model, ema, optimizer, device):
    state = torch.load(path, map_location=device)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(state["model"])
    ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])
    return state["step"]


# ---------------------------------------------------------------------------
# Mixed precision scaler helper
# ---------------------------------------------------------------------------

def make_scaler(precision: str):
    if precision == "bf16":
        return None   # bf16 doesn't need GradScaler
    return torch.cuda.amp.GradScaler()


def autocast(precision: str):
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    return torch.autocast("cuda", dtype=dtype)


# ---------------------------------------------------------------------------
# Build everything from config
# ---------------------------------------------------------------------------

def build_model(cfg, device):
    model = DiT.from_config(cfg.model)
    if cfg.model.get("pretrained"):
        model.load_pretrained(cfg.model.pretrained)
    return model.to(device)


def build_optimizer(model: nn.Module, cfg):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.999),
    )


def build_scheduler(optimizer, cfg):
    warmup = cfg.training.warmup_steps
    total  = cfg.training.steps

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        # cosine decay after warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# ImageNet training step
# ---------------------------------------------------------------------------

def train_step_imagenet(
    batch,
    model,
    vae,
    encoder,
    kernel,
    cfg,
    device,
    precision,
    scaler,
):
    images, labels = batch
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    # Encode images → latents (frozen VAE, no grad)
    with torch.no_grad():
        latents = vae.encode(images)    # [N, 4, 32, 32]

    # CFG: sample alpha or use unconditional step
    use_cfg    = cfg.cfg.enabled
    uncond_prob = cfg.cfg.get("uncond_prob", 0.1)
    alpha      = 1.0
    if use_cfg:
        alpha_max = cfg.cfg.alpha_max
        alpha = torch.empty(1).uniform_(1.0, alpha_max).item()

    # Generate: sample noise → model forward
    eps = torch.randn_like(latents)

    with autocast(precision):
        x_gen = model(eps, y=labels)   # [N, 4, 32, 32]

        # Positives = real latents of same classes in batch
        y_pos = latents.detach()

        if use_cfg and alpha > 1.0:
            # Need unconditional positives for CFG mixed negatives
            # Use a shuffled subset of the same batch as "unconditional" real data
            idx_uncond = torch.randperm(y_pos.shape[0], device=device)
            y_pos_uncond = y_pos[idx_uncond]
            # Flatten spatial dims for kernel: [N, 4*32*32]
            x_gen_flat = x_gen.flatten(1)
            y_pos_flat = y_pos.flatten(1)
            y_pos_uncond_flat = y_pos_uncond.flatten(1)
            loss, info = drifting_loss_cfg(
                x_gen_flat, y_pos_flat, y_pos_uncond_flat, kernel, alpha
            )
        else:
            x_gen_flat = x_gen.flatten(1)
            y_pos_flat = y_pos.flatten(1)
            loss, info = drifting_loss(x_gen_flat, y_pos_flat, kernel)

    return loss, info


# ---------------------------------------------------------------------------
# Robotics training step
# ---------------------------------------------------------------------------

def train_step_robotics(batch, model, kernel, device, precision):
    obs     = batch["obs"].to(device, non_blocking=True)
    actions = batch["action"].to(device, non_blocking=True)  # [N, H, action_dim]
    labels  = batch["task_id"].to(device, non_blocking=True)

    # Actions as 1-D "latents": flatten [N, H*action_dim]
    actions_flat = actions.flatten(1)
    # Add spatial dims for DiT (treat as 1D sequence via 1×D "image")
    action_dim  = actions_flat.shape[-1]
    x_2d        = actions_flat.unsqueeze(-1).unsqueeze(-1)  # [N, D, 1, 1]
    eps         = torch.randn_like(x_2d)

    with autocast(precision):
        x_gen   = model(eps, y=labels)            # [N, D, 1, 1]
        x_gen_f = x_gen.flatten(1)                # [N, D]
        y_pos_f = actions_flat.detach()
        loss, info = drifting_loss(x_gen_f, y_pos_f, kernel)

    return loss, info


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg_path: str, smoke_test: bool = False, resume: str | None = None):
    cfg = OmegaConf.load(cfg_path)
    if smoke_test:
        cfg.training.steps       = 2
        cfg.training.log_every   = 1
        cfg.training.ckpt_every  = 2
        cfg.training.warmup_steps = 0

    # Distributed setup
    rank, world_size, local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")
    log(rank, f"[init] rank={rank}/{world_size}, device={device}")

    # ---- Models ----
    model  = build_model(cfg, device)
    ema    = copy.deepcopy(model).eval()

    encoder = build_encoder(cfg.encoder)
    if encoder is not None:
        encoder = encoder.to(device)

    kernel  = build_kernel(cfg.kernel, encoder=encoder)

    # VAE (ImageNet only)
    vae = None
    if cfg.data.type == "imagenet":
        vae = VAEWrapper.from_config(cfg.vae).to(device)

    # ---- Optimizer ----
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler    = make_scaler(cfg.training.precision)

    # ---- DDP ----
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    dist.barrier()

    # ---- Data ----
    if cfg.data.type == "imagenet":
        from data.imagenet import build_imagenet_loader
        loader, num_classes = build_imagenet_loader(cfg.data, rank=rank, world_size=world_size)
    elif cfg.data.type == "robotics":
        from data.robotics import build_robotics_loader
        loader, action_dim, num_tasks = build_robotics_loader(cfg.data)
    else:
        raise ValueError(f"Unknown data type: {cfg.data.type}")

    # ---- Resume ----
    start_step = 0
    if resume:
        start_step = load_checkpoint(resume, model, ema, optimizer, device)
        log(rank, f"[resume] from step {start_step}")

    # ---- Training loop ----
    precision = cfg.training.precision
    out_dir   = cfg.output_dir
    t0        = time.time()
    data_iter = iter(loader)

    for step in range(start_step, cfg.training.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch     = next(data_iter)

        optimizer.zero_grad(set_to_none=True)

        if cfg.data.type == "imagenet":
            loss, info = train_step_imagenet(
                batch, model, vae, encoder, kernel, cfg, device, precision, scaler
            )
        else:
            loss, info = train_step_robotics(batch, model, kernel, device, precision)

        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()

        scheduler.step()
        update_ema(ema, model.module, cfg.training.ema_decay)

        # Logging
        if step % cfg.training.log_every == 0 and is_main(rank):
            elapsed = time.time() - t0
            lr      = scheduler.get_last_lr()[0]
            print(
                f"step={step:7d} | loss={info['loss']:.4f} | "
                f"V_norm={info['V_norm']:.4f} | lr={lr:.2e} | "
                f"t={elapsed:.1f}s",
                flush=True,
            )

        # Checkpoint
        if (step + 1) % cfg.training.ckpt_every == 0:
            save_checkpoint(out_dir, step + 1, model, ema, optimizer, rank)

    dist.barrier()
    save_checkpoint(out_dir, cfg.training.steps, model, ema, optimizer, rank)
    log(rank, "Training complete.")
    cleanup_dist()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--smoke-test", action="store_true", help="2-step smoke test")
    parser.add_argument("--resume",     type=str, default=None, help="path to checkpoint")
    args = parser.parse_args()

    train(args.config, smoke_test=args.smoke_test, resume=args.resume)

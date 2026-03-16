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
        ema_p.mul_(decay).add_(p.detach().cpu(), alpha=1 - decay)


def save_checkpoint(out_dir: str, epoch: int, step: int, model, ema, optimizer, rank: int):
    if not is_main(rank):
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    state = {
        "epoch":     epoch,
        "step":      step,
        "model":     model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "ema":       ema.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, os.path.join(out_dir, f"ckpt_step{step:07d}.pt"))
    torch.save(state, os.path.join(out_dir, "last.pt"))
    log(rank, f"[ckpt] saved step {step}")


def load_checkpoint(path: str, model, ema, optimizer, device):
    state = torch.load(path, map_location=device, weights_only=False)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(state["model"])
    ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])
    return state.get("epoch", 0), state.get("step", 0)


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
    # Wire cfg_dropout_prob from training cfg into model cfg before from_config
    model_cfg = cfg.model
    if cfg.get("cfg") and cfg.cfg.get("uncond_prob") is not None:
        model_cfg = OmegaConf.merge(model_cfg, {"cfg_dropout_prob": cfg.cfg.uncond_prob})
    model = DiT.from_config(model_cfg)
    if cfg.model.get("pretrained"):
        model.load_pretrained(cfg.model.pretrained)
    return model.to(device)


def build_optimizer(model: nn.Module, cfg):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.9998),
    )


def sample_alpha(alpha_max: float) -> float:
    """
    Sample CFG guidance scale α per paper:
      50%  → α = 1  (no guidance)
      50%  → α ~ p(α) ∝ α⁻³  on [1, α_max]  (inverse-CDF method)
    """
    if torch.rand(1).item() < 0.5:
        return 1.0
    u = torch.rand(1).item()
    # Inverse CDF of p(α) ∝ α⁻³: α = 1/sqrt(1 - u·(1 - 1/α_max²))
    return 1.0 / (1.0 - u * (1.0 - 1.0 / alpha_max ** 2)) ** 0.5


def build_scheduler(optimizer, cfg):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup = cfg.training.warmup_steps
    total  = cfg.training.steps
    warmup_sched = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=total - warmup, eta_min=0)
    return SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup])


# ---------------------------------------------------------------------------
# ImageNet training step
# ---------------------------------------------------------------------------

def train_step_imagenet(
    batch,
    model,
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

    # First N_pos = positives, remaining = uncond negatives.
    N = cfg.data.n_samples_per_class
    images_pos, labels_pos = images[:N], labels[:N]
    images_unc = images[N:]

    # CFG: sample alpha
    use_cfg = cfg.cfg.enabled
    alpha   = 1.0
    if use_cfg:
        alpha = sample_alpha(cfg.cfg.alpha_max)

    eps     = torch.randn_like(images_pos)
    alpha_t = torch.full((N,), alpha, device=device)

    with autocast(precision):
        x_gen = model(eps, y=labels_pos, alpha=alpha_t)
        if use_cfg:
            loss, info = drifting_loss_cfg(x_gen, images_pos.detach(), images_unc.detach(), kernel, alpha, labels=labels_pos)
        else:
            loss, info = drifting_loss(x_gen, images_pos.detach(), kernel, labels=labels_pos)

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
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(eval_loader, model, kernel, cfg, device, precision, eval_batches: int):
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    losses, v_norms = [], []
    for i, batch in enumerate(eval_loader):
        if i >= eval_batches:
            break
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        N = cfg.data.n_samples_per_class
        images_pos, labels_pos = images[:N], labels[:N]
        images_unc = images[N:]
        # Test set has no labels (all -1); fall back to random classes for eval diagnostics.
        if (labels_pos < 0).any():
            labels_pos = torch.randint(0, 1000, (N,), device=device)
        alpha = sample_alpha(cfg.cfg.alpha_max) if cfg.cfg.enabled else 1.0
        eps     = torch.randn_like(images_pos)
        alpha_t = torch.full((N,), alpha, device=device)
        with autocast(precision):
            x_gen = raw_model(eps, y=labels_pos, alpha=alpha_t)
            if cfg.cfg.enabled:
                _, info = drifting_loss_cfg(x_gen, images_pos, images_unc, kernel, alpha, labels=labels_pos)
            else:
                _, info = drifting_loss(x_gen, images_pos, kernel, labels=labels_pos)
        losses.append(info["loss"])
        v_norms.append(info["V_norm"])
    raw_model.train()
    n = max(len(losses), 1)
    return {"eval_loss": sum(losses) / n, "eval_V_norm": sum(v_norms) / n}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg_path: str, smoke_test: bool = False, resume: str | None = None):
    cfg = OmegaConf.load(cfg_path)
    if smoke_test:
        cfg.training.steps        = 4
        cfg.training.log_every    = 1
        cfg.training.ckpt_every   = 4
        cfg.training.eval_every   = 4
        cfg.training.eval_batches = 2
        cfg.training.warmup_steps = 0

    # Distributed setup
    rank, world_size, local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")
    log(rank, f"[init] rank={rank}/{world_size}, device={device}")

    # ---- Models ----
    model  = build_model(cfg, device)
    ema    = copy.deepcopy(model).cpu().eval()  # keep EMA on CPU, saves ~1.5GB for L/16

    encoder = build_encoder(cfg.encoder)
    if encoder is not None:
        encoder = encoder.to(device)

    kernel = build_kernel(cfg.kernel, encoder=encoder)

    # ---- Data ----
    if cfg.data.type == "imagenet":
        from data.imagenet import build_imagenet_loader, build_imagenet_eval_loader
        loader, train_sampler, num_classes = build_imagenet_loader(cfg.data, rank=rank, world_size=world_size)
        eval_loader = build_imagenet_eval_loader(cfg.data, rank=rank, world_size=world_size)
    elif cfg.data.type == "robotics":
        from data.robotics import build_robotics_loader
        loader, action_dim, num_tasks = build_robotics_loader(cfg.data)
        train_sampler, eval_loader = None, None
    else:
        raise ValueError(f"Unknown data type: {cfg.data.type}")

    # ---- Optimizer / Scheduler ----
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler    = make_scaler(cfg.training.precision)

    # ---- DDP ----
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    dist.barrier()

    # ---- Resume ----
    start_step = 0
    if resume:
        _, start_step = load_checkpoint(resume, model, ema, optimizer, device)
        log(rank, f"[resume] from step {start_step}")

    # ---- Wandb (rank 0 only) ----
    use_wandb = cfg.get("logging", {}).get("wandb", False)
    if use_wandb and is_main(rank):
        import wandb  # noqa: PLC0415 — optional dependency, imported once here
        wandb.init(
            project=cfg.get("logging", {}).get("project", "drifting"),
            name=cfg.get("name", "run"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ---- Training loop ----
    precision  = cfg.training.precision
    out_dir    = cfg.output_dir
    t0         = time.time()
    data_iter  = iter(loader)
    epoch      = 0
    loss_ema   = None   # exponential moving avg of loss for spike detection

    for step in range(start_step, cfg.training.steps):
        # Restart loader when exhausted; update sampler epoch for DDP shuffling
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            data_iter = iter(loader)
            batch     = next(data_iter)

        optimizer.zero_grad(set_to_none=True)

        if cfg.data.type == "imagenet":
            loss, info = train_step_imagenet(
                batch, model, encoder, kernel, cfg, device, precision, scaler
            )
        else:
            loss, info = train_step_robotics(batch, model, kernel, device, precision)

        # Loss spike detection — skip step if loss is NaN or >10× the running avg
        loss_val = info["loss"]
        is_spike = (not torch.isfinite(loss)) or \
                   (loss_ema is not None and loss_val > 10.0 * loss_ema)
        if is_spike:
            log(rank, f"[warn] step={step} skipping spike loss={loss_val:.4f} (ema={loss_ema:.4f})")
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            continue
        loss_ema = loss_val if loss_ema is None else 0.99 * loss_ema + 0.01 * loss_val

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
                f"V_norm={info['V_norm']:.4f} | lr={lr:.2e} | t={elapsed:.1f}s",
                flush=True,
            )
            if use_wandb:
                wandb.log({"loss": info["loss"], "V_norm": info["V_norm"], "lr": lr}, step=step)

        # Checkpoint
        if (step + 1) % cfg.training.ckpt_every == 0:
            save_checkpoint(out_dir, epoch, step + 1, model, ema, optimizer, rank)

        # Eval
        if eval_loader is not None and (step + 1) % cfg.training.eval_every == 0:
            eval_info = evaluate(
                eval_loader, model, kernel, cfg, device, precision,
                eval_batches=cfg.training.eval_batches,
            )
            if is_main(rank):
                print(
                    f"[eval] step={step+1} | eval_loss={eval_info['eval_loss']:.4f} | "
                    f"eval_V_norm={eval_info['eval_V_norm']:.4f}",
                    flush=True,
                )
                if use_wandb:
                    wandb.log({"eval_loss": eval_info["eval_loss"], "eval_V_norm": eval_info["eval_V_norm"]}, step=step)

    dist.barrier()
    save_checkpoint(out_dir, epoch, cfg.training.steps, model, ema, optimizer, rank)
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

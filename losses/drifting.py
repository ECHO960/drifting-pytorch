"""
Drifting loss implementation.

Core idea (arXiv 2602.04770):
  1. Generate x = f_θ(ε)
  2. Compute drift field V(x) = V⁺_p(x) − V⁻_q(x)
       V⁺: attraction toward real data  y_pos ~ p_data
       V⁻: repulsion from generated     y_neg = x (current batch)
  3. Loss = MSE(x,  stopgrad(x + V(x)))
  4. Anti-symmetry is guaranteed by construction: V_{p,q} = -V_{q,p}
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from kernels import DriftKernel


# ---------------------------------------------------------------------------
# Core field computation
# ---------------------------------------------------------------------------

def compute_drift_field(
    x_gen: Tensor,
    y_pos: Tensor,
    y_neg: Tensor,
    kernel: DriftKernel,
) -> Tensor:
    """
    Compute V_{p,q}(x) = V⁺_p(x) − V⁻_q(x).

    Args:
        x_gen: [N, D] generated samples (query points)
        y_pos: [P, D] real data samples  (positives from p_data)
        y_neg: [Q, D] generated samples  (negatives from q_θ; usually = x_gen)
        kernel: DriftKernel instance

    Returns:
        V: [N, D] drift vector for each generated sample
    """
    x_flat = x_gen.flatten(1)   # [N, D]
    p_flat = y_pos.flatten(1)   # [P, D]
    q_flat = y_neg.flatten(1)   # [Q, D]

    # Unnormalized weights
    w_pos = kernel(x_gen, y_pos)        # [N, P]
    w_neg = kernel(x_gen, y_neg)        # [N, Q]

    # Row-normalize (softmax-style but with explicit sum)
    w_pos = w_pos / (w_pos.sum(dim=-1, keepdim=True) + 1e-8)
    w_neg = w_neg / (w_neg.sum(dim=-1, keepdim=True) + 1e-8)

    # V⁺(x) = Σ_j w_pos[i,j] * (y_pos[j] - x[i])
    #        = (w_pos @ p_flat) - x_flat
    V_plus  = (w_pos @ p_flat) - x_flat   # [N, D]  attraction

    # V⁻(x) = Σ_j w_neg[i,j] * (y_neg[j] - x[i])
    #        = (w_neg @ q_flat) - x_flat
    V_minus = (w_neg @ q_flat) - x_flat   # [N, D]  repulsion

    V = V_plus - V_minus                   # [N, D]
    return V.view_as(x_gen)


# ---------------------------------------------------------------------------
# Training loss
# ---------------------------------------------------------------------------

def drifting_loss(
    x_gen: Tensor,
    y_pos: Tensor,
    kernel: DriftKernel,
    y_neg: Tensor | None = None,
) -> tuple[Tensor, dict]:
    """
    Compute the drifting training loss.

    L = ‖x − stopgrad(x + V(x))‖²

    Args:
        x_gen:  [N, *] generated samples from f_θ(ε)
        y_pos:  [P, *] real data samples (same class / task as x_gen)
        kernel: DriftKernel
        y_neg:  [Q, *] negatives; defaults to x_gen (self-repulsion)

    Returns:
        loss:   scalar
        info:   dict with diagnostic scalars
    """
    if y_neg is None:
        y_neg = x_gen

    V = compute_drift_field(x_gen, y_pos, y_neg, kernel)   # [N, *]

    target = (x_gen + V).detach()   # stopgrad — no gradient through target
    loss = F.mse_loss(x_gen, target)

    with torch.no_grad():
        v_norm = V.flatten(1).norm(dim=-1).mean()

    return loss, {"loss": loss.item(), "V_norm": v_norm.item()}


# ---------------------------------------------------------------------------
# CFG: mixed-negative drifting loss
# ---------------------------------------------------------------------------

def drifting_loss_cfg(
    x_gen_cond: Tensor,
    y_pos_cond: Tensor,
    y_pos_uncond: Tensor,
    kernel: DriftKernel,
    alpha: float,
) -> tuple[Tensor, dict]:
    """
    Drifting loss with classifier-free guidance training (Section 4, paper).

    Mixed negatives: q̃ = (1−γ)·q_θ(·|c) + γ·p_data(·|∅)
    where γ = 1 − 1/α.

    Args:
        x_gen_cond:    [N, *] samples generated with class label
        y_pos_cond:    [P, *] real data for this class
        y_pos_uncond:  [U, *] unconditional real data (random classes)
        kernel:        DriftKernel
        alpha:         CFG scale (≥ 1)
    """
    gamma = 1.0 - 1.0 / alpha

    # Build mixed negatives: gamma fraction from unconditional real data,
    # (1-gamma) fraction from the generated batch.
    N = x_gen_cond.shape[0]
    n_uncond = max(1, int(N * gamma))
    n_cond   = N - n_uncond

    idx_uncond = torch.randperm(y_pos_uncond.shape[0], device=x_gen_cond.device)[:n_uncond]
    y_neg_mixed = torch.cat([
        x_gen_cond[:n_cond],
        y_pos_uncond[idx_uncond],
    ], dim=0)

    return drifting_loss(x_gen_cond, y_pos_cond, kernel, y_neg=y_neg_mixed)

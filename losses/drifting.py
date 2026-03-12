"""
Drifting loss implementation.

Matches the official Colab demo (lambertae/lambertae.github.io).

Key differences vs. naive row-norm:
  - Joint kernel matrix over [gen; pos] — one cdist call
  - Symmetric normalization: k[i,j] / sqrt(row_sum[i] * col_sum[j])
    (graph Laplacian style, not softmax)
  - Self-masking: gen[i]->gen[i] distance set to inf
  - Confidence-weighted V: V = (nk_pos * s_neg) @ pos - (nk_neg * s_pos) @ gen
    drift magnitude scales with neighborhood density
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from kernels import DriftKernel


# ---------------------------------------------------------------------------
# Core field computation  (matches official implementation)
# ---------------------------------------------------------------------------

def compute_drift_field(
    x_gen: Tensor,
    y_pos: Tensor,
    kernel: DriftKernel,
) -> Tensor:
    """
    Compute drift field V(x_gen) toward y_pos and away from x_gen itself.

    Algorithm:
      1. targets = [gen; pos]
      2. dist = cdist(encode(gen), encode(targets))  [G, G+P]
      3. Mask diagonal (gen[i] vs gen[i]) -> inf
      4. k = exp(-dist / tau)
      5. Symmetric norm: nk = k / sqrt(row_sum x col_sum)
      6. s_neg = sum_j nk_neg[i,j],  s_pos = sum_j nk_pos[i,j]
      7. V = (nk_pos * s_neg) @ pos - (nk_neg * s_pos) @ gen

    Args:
        x_gen:  [N, *] generated samples
        y_pos:  [P, *] real data samples (same class / task)
        kernel: DriftKernel -- encoder phi accessed via kernel.encoder, tau via kernel.tau
    """
    G = x_gen.shape[0]

    x_flat = x_gen.flatten(1).float()   # [G, D]
    p_flat = y_pos.flatten(1).float()   # [P, D]

    # Feature encoding
    encoder = getattr(kernel, "encoder", None)
    tau     = getattr(kernel, "tau", 0.05)

    if encoder is not None:
        with torch.no_grad():
            fx = encoder(x_gen)   # [G, F]
            fy = encoder(y_pos)   # [P, F]
    else:
        fx = x_flat
        fy = p_flat

    # Joint distance: gen -> [gen, pos]
    feat_targets = torch.cat([fx, fy], dim=0)             # [G+P, F]
    dist = torch.cdist(fx, feat_targets, p=2)             # [G, G+P]

    # Self-masking
    dist[:, :G].fill_diagonal_(float("inf"))

    k = torch.exp(-dist / tau)                            # [G, G+P]

    # Symmetric normalization
    row_sum = k.sum(dim=-1, keepdim=True)                 # [G, 1]
    col_sum = k.sum(dim=0,  keepdim=True)                 # [1, G+P]
    nk = k / (row_sum * col_sum).clamp_min(1e-12).sqrt()  # [G, G+P]

    nk_neg = nk[:, :G]                                    # [G, G]
    nk_pos = nk[:, G:]                                    # [G, P]

    s_neg = nk_neg.sum(dim=-1, keepdim=True)              # [G, 1]
    s_pos = nk_pos.sum(dim=-1, keepdim=True)              # [G, 1]

    V = (nk_pos * s_neg) @ p_flat - (nk_neg * s_pos) @ x_flat   # [G, D]

    return V.view_as(x_gen)


# ---------------------------------------------------------------------------
# Training loss
# ---------------------------------------------------------------------------

def drifting_loss(
    x_gen: Tensor,
    y_pos: Tensor,
    kernel: DriftKernel,
) -> tuple[Tensor, dict]:
    """
    L = ||x - stopgrad(x + V(x))||^2

    Args:
        x_gen:  [N, *] generated samples from f_theta(eps)
        y_pos:  [P, *] real data samples (same class / task as x_gen)
        kernel: DriftKernel
    """
    with torch.no_grad():
        V      = compute_drift_field(x_gen, y_pos, kernel)
        target = x_gen + V

    loss = F.mse_loss(x_gen, target)

    with torch.no_grad():
        v_norm = V.flatten(1).norm(dim=-1).mean()

    return loss, {"loss": loss.item(), "V_norm": v_norm.item()}


# ---------------------------------------------------------------------------
# CFG: mixed-positive drifting loss
# ---------------------------------------------------------------------------

def drifting_loss_cfg(
    x_gen_cond: Tensor,
    y_pos_cond: Tensor,
    y_pos_uncond: Tensor,
    kernel: DriftKernel,
    alpha: float,
) -> tuple[Tensor, dict]:
    """
    CFG training: mixed positives q~ = (1-gamma)*p(|c) + gamma*p(|empty), gamma = 1 - 1/alpha.

    Appends a gamma-fraction of unconditional real data to y_pos_cond so the
    kernel is attracted toward both conditional and unconditional real data.
    """
    gamma    = 1.0 - 1.0 / alpha
    n_uncond = max(1, int(x_gen_cond.shape[0] * gamma))

    idx         = torch.randperm(y_pos_uncond.shape[0], device=x_gen_cond.device)[:n_uncond]
    y_pos_mixed = torch.cat([y_pos_cond, y_pos_uncond[idx]], dim=0)

    return drifting_loss(x_gen_cond, y_pos_mixed, kernel)

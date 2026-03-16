"""
Drifting loss — paper-accurate implementation (§A.5–A.6).

Feature normalization (Eq. 18–21):
  S_j    = mean_batch(‖φ_j(x)−φ_j(y)‖) / √C_j          [stop-grad]
  φ̃_j   = φ_j / S_j          →  E[dist] ≈ √C_j
  τ̃_j   = τ · √C_j

Drift normalization (Eq. 23–25):
  λ_j   = √( E[‖V_j‖² / C_j] )                         [stop-grad]
  Ṽ_j   = V_j / λ_j

Multiple temperatures (§A.6):
  Ṽ_j  ← Σ_τ  Ṽ_j,τ

Loss (Eq. 26):
  L_j  = MSE( φ̃_j(x),  sg(φ̃_j(x) + Ṽ_j) )
  Gradient flows through the frozen encoder back to x_gen.

CFG (§3.5 / App. A.7):  k_pos × α,  k_unc × (α−1)
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from kernels import DriftKernel


# ---------------------------------------------------------------------------
# Distributed gather — gradients flow back to local rank only
# ---------------------------------------------------------------------------

class _GatherLayer(torch.autograd.Function):
    """All-gather with gradient support for the local rank's slice."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        if not (dist.is_available() and dist.is_initialized()):
            return (x,)
        out = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(out, x.contiguous())
        return tuple(out)

    @staticmethod
    def backward(ctx, *grads):
        x, = ctx.saved_tensors
        if not (dist.is_available() and dist.is_initialized()):
            return grads[0]
        grad = torch.stack(grads)
        dist.all_reduce(grad)
        return grad[dist.get_rank()]


def _all_gather(x: Tensor) -> Tensor:
    """Gather x from all ranks → [world*N, C]. Grad flows to local slice."""
    return torch.cat(_GatherLayer.apply(x), dim=0)


def _all_gather_nograd(x: Tensor) -> Tensor:
    """Gather x from all ranks → [world*N, C]. No gradient."""
    if not (dist.is_available() and dist.is_initialized()):
        return x
    out = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(out, x.contiguous())
    return torch.cat(out, dim=0)


def _rank() -> int:
    return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0


def _get_rank_slice(tensor: Tensor, local_size: int) -> Tensor:
    r = _rank()
    return tensor[r * local_size:(r + 1) * local_size]


# ---------------------------------------------------------------------------
# Core: drift field for one temperature
# ---------------------------------------------------------------------------

def _drift_one_tau(
    fx_norm:         Tensor,
    fy_norm:         Tensor,
    fu_norm:         Tensor | None,
    tau_tilde:       float,
    alpha:           float = 1.0,
    same_class_mask: Tensor | None = None,
) -> Tensor:
    """
    Returns V_tilde [G, C] (stop-grad) for one temperature.
    All inputs are already feature-normalised.

    same_class_mask: [G, G] bool, True where labels_i == labels_j.
    When provided, same-class pairs are excluded from the negative set so only
    cross-class generated samples serve as negatives (matching paper's batching).
    """
    G, C = fx_norm.shape[0], fx_norm.shape[1]
    use_cfg = fu_norm is not None

    all_targets = torch.cat(
        [fx_norm, fy_norm] + ([fu_norm] if use_cfg else []), dim=0
    )
    dist_mat = torch.cdist(fx_norm, all_targets, p=2)     # [G, G+P(+U)]

    if same_class_mask is not None:
        dist_mat[:, :G][same_class_mask] = float("inf")
    else:
        dist_mat[:, :G].fill_diagonal_(float("inf"))

    k_raw = torch.exp(-dist_mat / tau_tilde)

    # Apply CFG weights before normalisation.
    # At α=1: k_unc *= 0 → unc columns vanish cleanly.
    k_neg = k_raw[:, :G]
    k_pos = k_raw[:, G:G + fy_norm.shape[0]] * alpha
    if use_cfg:
        k_unc = k_raw[:, G + fy_norm.shape[0]:] * (alpha - 1.0)
        k_w = torch.cat([k_neg, k_pos, k_unc], dim=1)
    else:
        k_w = torch.cat([k_neg, k_pos], dim=1)

    row_sum = k_w.sum(-1, keepdim=True)
    col_sum = k_w.sum(0,  keepdim=True)
    nk = k_w / (row_sum * col_sum).clamp_min(1e-12).sqrt()

    P      = fy_norm.shape[0]
    nk_neg = nk[:, :G]
    nk_pos = nk[:, G:G + P]

    if use_cfg:
        nk_unc = nk[:, G + P:]
        s_neg  = (nk_neg.sum(-1) + nk_unc.sum(-1)).unsqueeze(-1)
    else:
        s_neg = nk_neg.sum(-1, keepdim=True)

    s_pos = nk_pos.sum(-1, keepdim=True)

    V_j = (nk_pos * s_neg) @ fy_norm - (nk_neg * s_pos) @ fx_norm
    if use_cfg:
        V_j = V_j - (nk_unc * s_pos) @ fu_norm

    lambda_j = ((V_j.pow(2).sum(-1) / C).mean()).sqrt().clamp_min(1e-6)
    return V_j / lambda_j


# ---------------------------------------------------------------------------
# Encode + feature-normalise
# ---------------------------------------------------------------------------

def _encode_normalise(encoder, x_gen: Tensor, y_pos: Tensor,
                      y_unc: Tensor | None = None):
    """
    Returns (fx_norm, fy_norm, fu_norm, sqrt_C).
    fx_norm has gradients; fy_norm/fu_norm are stop-grad.
    """
    if encoder is not None:
        with torch.no_grad():
            fy = encoder(y_pos).float()
            fu = encoder(y_unc).float() if y_unc is not None else None
        fx = encoder(x_gen).float()
    else:
        fx = x_gen.flatten(1).float()
        fy = y_pos.flatten(1).float()
        fu = y_unc.flatten(1).float() if y_unc is not None else None

    C      = fx.shape[-1]
    sqrt_C = C ** 0.5

    with torch.no_grad():
        all_ref  = torch.cat([fx.detach(), fy] + ([fu] if fu is not None else []), dim=0)
        dist_all = torch.cdist(fx.detach(), all_ref, p=2)
        G        = fx.shape[0]
        dist_all[:, :G].fill_diagonal_(float("inf"))
        finite = dist_all[dist_all.isfinite()]
        S_j    = (finite.mean() / sqrt_C).clamp_min(1e-6)

    fx_norm = fx / S_j
    fy_norm = fy / S_j
    fu_norm = fu / S_j if fu is not None else None

    return fx_norm, fy_norm, fu_norm, sqrt_C


# ---------------------------------------------------------------------------
# Public loss functions
# ---------------------------------------------------------------------------

def drifting_loss(
    x_gen:  Tensor,
    y_pos:  Tensor,
    kernel: DriftKernel,
    labels: Tensor | None = None,
) -> tuple[Tensor, dict]:
    encoder = getattr(kernel, "encoder", None)
    taus    = getattr(kernel, "taus", [getattr(kernel, "tau", 0.05)])

    fx_norm, fy_norm, _, sqrt_C = _encode_normalise(encoder, x_gen, y_pos)
    G_local = fx_norm.shape[0]

    fx_norm_all = _all_gather(fx_norm)
    fy_norm_all = _all_gather_nograd(fy_norm)
    labels_all  = _all_gather_nograd(labels) if labels is not None else None

    same_class_mask = (labels_all.unsqueeze(0) == labels_all.unsqueeze(1)) if labels_all is not None else None
    V_sum = sum(
        _drift_one_tau(fx_norm_all.detach(), fy_norm_all, None, tau * sqrt_C,
                       same_class_mask=same_class_mask)
        for tau in taus
    )

    V_local       = _get_rank_slice(V_sum, G_local)
    fx_norm_local = _get_rank_slice(fx_norm_all, G_local)
    target = fx_norm_local.detach() + V_local
    loss   = F.mse_loss(fx_norm_local, target)

    with torch.no_grad():
        v_norm = V_local.norm(dim=-1).mean()

    return loss, {"loss": loss.item(), "V_norm": v_norm.item()}


def drifting_loss_cfg(
    x_gen_cond:   Tensor,
    y_pos_cond:   Tensor,
    y_pos_uncond: Tensor,
    kernel:       DriftKernel,
    alpha:        float,
    labels:       Tensor | None = None,
) -> tuple[Tensor, dict]:
    encoder = getattr(kernel, "encoder", None)
    taus    = getattr(kernel, "taus", [getattr(kernel, "tau", 0.05)])

    fx_norm, fy_norm, fu_norm, sqrt_C = _encode_normalise(
        encoder, x_gen_cond, y_pos_cond, y_pos_uncond
    )
    G_local = fx_norm.shape[0]

    fx_norm_all = _all_gather(fx_norm)
    fy_norm_all = _all_gather_nograd(fy_norm)
    fu_norm_all = _all_gather_nograd(fu_norm)
    labels_all  = _all_gather_nograd(labels) if labels is not None else None

    same_class_mask = (labels_all.unsqueeze(0) == labels_all.unsqueeze(1)) if labels_all is not None else None
    V_sum = sum(
        _drift_one_tau(fx_norm_all.detach(), fy_norm_all, fu_norm_all, tau * sqrt_C, alpha,
                       same_class_mask=same_class_mask)
        for tau in taus
    )

    V_local       = _get_rank_slice(V_sum, G_local)
    fx_norm_local = _get_rank_slice(fx_norm_all, G_local)
    target = fx_norm_local.detach() + V_local
    loss   = F.mse_loss(fx_norm_local, target)

    with torch.no_grad():
        v_norm = V_local.norm(dim=-1).mean()

    return loss, {"loss": loss.item(), "V_norm": v_norm.item()}

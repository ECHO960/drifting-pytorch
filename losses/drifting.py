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

Per-class structure (matching reference implementation):
  For each class c, we preselect:
    feat_gen_c  — generated images conditioned on class c
    feat_pos_c  — real images of class c (strict same-class positives)
    feat_neg_c  — generated images of class c + uncond images (negatives)
  This guarantees V_pos only attracts toward same-class real images.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from kernels import DriftKernel
from losses.dist_utils import all_gather, all_gather_nograd, get_rank_slice


# ---------------------------------------------------------------------------
# Core: drift field for one temperature, pre-selected features
# ---------------------------------------------------------------------------

def _drift_one_tau(
    feat_gen: Tensor,   # [G, C] generated images of class c  (detached)
    feat_pos: Tensor,   # [P, C] real images of class c       (stop-grad)
    feat_neg: Tensor,   # [N, C] cross-class generated [+ uncond] (stop-grad)
    tau_tilde: float,
    alpha: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Returns (V/λ, V_pos/λ, V_neg/λ) for one temperature.

    feat_neg contains images from OTHER classes (disjoint from feat_gen),
    so no self-masking is needed.
    """
    G, C = feat_gen.shape
    N    = feat_neg.shape[0]

    all_ref  = torch.cat([feat_neg, feat_pos], dim=0)    # [N+P, C]
    dist_mat = torch.cdist(feat_gen, all_ref, p=2)        # [G, N+P]

    logits = -dist_mat / tau_tilde
    nk_row = torch.softmax(logits, dim=0)
    nk_col = torch.softmax(logits, dim=1)
    nk = torch.sqrt(nk_row * nk_col)

    nk_neg = nk[:, :N]
    nk_pos = nk[:, N:]
    s_neg  = nk_neg.sum(-1, keepdim=True)
    s_pos  = nk_pos.sum(-1, keepdim=True)

    V_pos = (nk_pos * s_neg) @ feat_pos   # scale after normalization so alpha is not absorbed
    V_neg = (nk_neg * s_pos) @ feat_neg

    V_j      = V_pos - V_neg
    lambda_j = V_j.pow(2).mean().sqrt().clamp_min(1e-6)
    # Always return float32 — matmul inside autocast (bf16) would otherwise produce bf16,
    # which mismatches the float32 V_sum accumulator.
    return (V_j / lambda_j).float(), (V_pos / lambda_j).float(), (V_neg / lambda_j).float()


# ---------------------------------------------------------------------------
# Encode + feature-normalise
# ---------------------------------------------------------------------------

def _encode_normalise(encoder, x_gen: Tensor, y_pos: Tensor,
                      y_unc: Tensor | None = None):
    """
    Returns (fx_norm, fy_norm, fu_norm, sqrt_C).
    fx_norm has gradients; fy_norm/fu_norm are stop-grad.
    """
    if encoder is None:
        fx_norm = F.normalize(x_gen.flatten(1).float(), p=2)
        fy_norm = F.normalize(y_pos.flatten(1).float(), p=2)
        fu_norm = F.normalize(y_unc.flatten(1).float(), p=2) if y_unc is not None else None
        return fx_norm, fy_norm, fu_norm, 1

    with torch.no_grad():
        fy = encoder(y_pos).float()
        fu = encoder(y_unc).float() if y_unc is not None else None
    fx = encoder(x_gen).float()

    C      = fx.shape[-1]
    sqrt_C = C ** 0.5

    with torch.no_grad():
        ref_only = torch.cat([fy] + ([fu] if fu is not None else []), dim=0)
        dist_ref = torch.cdist(fx.detach(), ref_only, p=2)   # [G, P+U], no self-distances
        S_j      = (dist_ref.mean() / sqrt_C).clamp_min(1e-6)

    fx_norm = fx / S_j
    fy_norm = fy / S_j
    fu_norm = fu / S_j if fu is not None else None

    return fx_norm, fy_norm, fu_norm, sqrt_C


# ---------------------------------------------------------------------------
# Public loss function
# ---------------------------------------------------------------------------

def drifting_loss(
    x_gen:      Tensor,
    y_pos:      Tensor,
    kernel:     DriftKernel,
    labels:     Tensor | None = None,
    labels_pos: Tensor | None = None,
    y_uncond:   Tensor | None = None,
    alpha:      float = 1.0,           # kept for API compatibility; not used in kernel
) -> tuple[Tensor, dict]:
    """
    Per-class drifting loss.

    For each class c present in the batch the loss selects:
      feat_gen_c  — generated images of class c
      feat_pos_c  — real images of class c (strict same-class positives)
      feat_neg_c  — generated images of class c + uncond images (if any)

    labels:     class labels for x_gen (required for per-class logic).
    labels_pos: class labels for y_pos. Defaults to labels when y_pos and
                x_gen share the same classes (the common case).
    y_uncond:   unconditional real images merged into the negative set.
    """
    encoder = getattr(kernel, "encoder", None)
    taus    = getattr(kernel, "taus", [getattr(kernel, "tau", 0.05)])

    fx_norm, fy_norm, fu_norm, sqrt_C = _encode_normalise(encoder, x_gen, y_pos, y_uncond)

    _lpos = labels_pos if labels_pos is not None else labels

    # Accumulators for V across classes — filled in the per-class loop below
    V_sum     = torch.zeros_like(fx_norm)
    V_pos_sum = torch.zeros_like(fx_norm)
    V_neg_sum = torch.zeros_like(fx_norm)

    if labels is None:
        classes = [None]
    else:
        classes = labels.unique().tolist()

    for c in classes:
        if c is None:
            mask_gen = torch.ones(fx_norm.shape[0], dtype=torch.bool, device=fx_norm.device)
            mask_pos = torch.ones(fy_norm.shape[0], dtype=torch.bool, device=fy_norm.device)
        else:
            mask_gen = labels == c
            mask_pos = _lpos == c
            if not mask_gen.any() or not mask_pos.any():
                continue

        feat_gen_c = fx_norm[mask_gen]
        feat_pos_c = fy_norm[mask_pos]

        # Negatives: cross-class generated images + uncond (if any).
        # Fall back to same-class when no other class exists in the batch.
        mask_neg   = ~mask_gen if labels is not None else torch.zeros_like(mask_gen)
        feat_neg_c = fx_norm[mask_neg].detach() if mask_neg.any() else feat_gen_c.detach()
        if fu_norm is not None:
            feat_neg_c = torch.cat([feat_neg_c, fu_norm], dim=0)

        n_pos = feat_pos_c.shape[0]
        n_neg = feat_neg_c.shape[0]
        auto_alpha = n_neg / max(n_pos, 1)   # balance kernel mass: N_neg pos-weighted cols vs N_neg neg cols

        V_c = V_pos_c = V_neg_c = None
        for tau in taus:
            V_tau, Vp_tau, Vn_tau = _drift_one_tau(
                feat_gen_c.detach(), feat_pos_c, feat_neg_c, tau * sqrt_C, auto_alpha,
            )
            V_c     = V_tau  if V_c     is None else V_c     + V_tau
            V_pos_c = Vp_tau if V_pos_c is None else V_pos_c + Vp_tau
            V_neg_c = Vn_tau if V_neg_c is None else V_neg_c + Vn_tau

        V_sum[mask_gen]     = V_c
        V_pos_sum[mask_gen] = V_pos_c
        V_neg_sum[mask_gen] = V_neg_c

    target = fx_norm.detach() + V_sum
    loss   = F.mse_loss(fx_norm, target)

    with torch.no_grad():
        v_norms = V_sum.norm(dim=-1)
    return loss, {
        "loss":      loss.item(),
        "V_norm":    v_norms.mean().item(),
        "V_std":     v_norms.std().item(),
        "V_max":     V_sum.abs().max().item(),
        "V_pos":     V_pos_sum.norm(dim=-1).mean().item(),
        "V_neg":     V_neg_sum.norm(dim=-1).mean().item(),
        "feat_norm": fx_norm.norm(dim=-1).mean().item(),
    }

def drifting_loss_all_gather(
    x_gen:      Tensor,
    y_pos:      Tensor,
    kernel:     DriftKernel,
    labels:     Tensor | None = None,
    labels_pos: Tensor | None = None,
    y_uncond:   Tensor | None = None,
    alpha:      float = 1.0,
) -> tuple[Tensor, dict]:
    """
    Same as drifting_loss but all-gathers fx_norm/fy_norm/fu_norm across ranks
    before the per-class loop, so each GPU sees the full cross-GPU feature pool.
    Gradient flows back to the local rank's slice of fx_norm only.
    """
    encoder = getattr(kernel, "encoder", None)
    taus    = getattr(kernel, "taus", [getattr(kernel, "tau", 0.05)])

    fx_norm, fy_norm, fu_norm, sqrt_C = _encode_normalise(encoder, x_gen, y_pos, y_uncond)
    G_local = fx_norm.shape[0]

    fx_norm_all = all_gather(fx_norm)                                         # grad → local slice
    fy_norm_all = all_gather_nograd(fy_norm)
    fu_norm_all = all_gather_nograd(fu_norm) if fu_norm is not None else None

    _lpos    = labels_pos if labels_pos is not None else labels
    lgen_all = all_gather_nograd(labels) if labels is not None else None
    lpos_all = all_gather_nograd(_lpos)  if _lpos  is not None else None

    V_sum_all     = torch.zeros_like(fx_norm_all)
    V_pos_sum_all = torch.zeros_like(fx_norm_all)
    V_neg_sum_all = torch.zeros_like(fx_norm_all)

    if lgen_all is None:
        classes = [None]
    else:
        classes = lgen_all.unique().tolist()

    for c in classes:
        if c is None:
            mask_gen = torch.ones(fx_norm_all.shape[0], dtype=torch.bool, device=fx_norm_all.device)
            mask_pos = torch.ones(fy_norm_all.shape[0], dtype=torch.bool, device=fy_norm_all.device)
        else:
            mask_gen = lgen_all == c
            mask_pos = lpos_all == c
            if not mask_gen.any() or not mask_pos.any():
                continue

        feat_gen_c = fx_norm_all[mask_gen]
        feat_pos_c = fy_norm_all[mask_pos]

        mask_neg   = ~mask_gen if lgen_all is not None else torch.zeros_like(mask_gen)
        feat_neg_c = fx_norm_all[mask_neg].detach() if mask_neg.any() else feat_gen_c.detach()
        if fu_norm_all is not None:
            feat_neg_c = torch.cat([feat_neg_c, fu_norm_all], dim=0)

        n_pos = feat_pos_c.shape[0]
        n_neg = feat_neg_c.shape[0]
        auto_alpha = n_neg / max(n_pos, 1)

        V_c = V_pos_c = V_neg_c = None
        for tau in taus:
            V_tau, Vp_tau, Vn_tau = _drift_one_tau(
                feat_gen_c.detach(), feat_pos_c, feat_neg_c, tau * sqrt_C, auto_alpha,
            )
            V_c     = V_tau  if V_c     is None else V_c     + V_tau
            V_pos_c = Vp_tau if V_pos_c is None else V_pos_c + Vp_tau
            V_neg_c = Vn_tau if V_neg_c is None else V_neg_c + Vn_tau

        V_sum_all[mask_gen]     = V_c
        V_pos_sum_all[mask_gen] = V_pos_c
        V_neg_sum_all[mask_gen] = V_neg_c

    target = fx_norm_all.detach() + V_sum_all
    loss   = F.mse_loss(fx_norm_all, target)

    with torch.no_grad():
        V_local  = get_rank_slice(V_sum_all,     G_local)
        Vp_local = get_rank_slice(V_pos_sum_all, G_local)
        Vn_local = get_rank_slice(V_neg_sum_all, G_local)
        fx_local = get_rank_slice(fx_norm_all,   G_local)
        v_norms  = V_local.norm(dim=-1)
    return loss, {
        "loss":      loss.item(),
        "V_norm":    v_norms.mean().item(),
        "V_std":     v_norms.std().item(),
        "V_max":     V_local.abs().max().item(),
        "V_pos":     Vp_local.norm(dim=-1).mean().item(),
        "V_neg":     Vn_local.norm(dim=-1).mean().item(),
        "feat_norm": fx_local.norm(dim=-1).mean().item(),
    }


# Alias kept for call-site clarity; both resolve to the same function.
drifting_loss_cfg = drifting_loss

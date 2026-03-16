# Paper vs. Implementation Discrepancy Report

**Paper:** "Generative Modeling via Drifting" (arXiv 2602.04770v2)
**Implementation root:** `/opt/tiger/drifting-pytorch/`
**Date:** 2026-03-13

---

## Critical Discrepancies

### ~~1. Kernel Normalization Formula~~ ✅ CONFIRMED CORRECT
**Author's official toy code (`toy_mean_drift.py`):**
```python
normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True)
nk = kernel / normalizer.clamp_min(1e-12).sqrt()
```
This is exactly `k / sqrt(row_sum × col_sum)` — identical to the implementation. The paper text loosely says "softmax" but the actual formula is simultaneous symmetric normalization. **Not a discrepancy.**

---

### 2. CFG Alpha Weighting
**Paper (§3.5):** Frames CFG as a change in *sampling proportions* of the target distribution: `q(c) = α·p_data(c) − (α−1)·p_data(∅)`. Does not say to multiply kernel columns by α before normalization.

**Code:**
```python
k_pos = k_raw[:, G:G + fy_norm.shape[0]] * alpha
k_unc = k_raw[:, G + fy_norm.shape[0]:] * (alpha - 1.0)
```
Explicitly scales kernel columns by α and (α−1) *before* the joint normalization, so the denominators absorb the weights. This is a specific implementation decision not described in the paper.

---

### 3. Feature Encoder
**Paper (Tables 3–5):** Best results use a **custom latent-space MAE encoder** (ResNet-based, width 640, trained 1280 epochs with classification fine-tuning). DINOv2 is never mentioned in the paper.

**Code:** Uses **DINOv2 ViT-L/14** (`dinov2_l`) or ViT-B/14 (`dinov2_b`). The paper's custom encoder is not implemented. Results are not directly comparable to the paper.

---

### 4. Latent Space vs. Pixel Space
**Paper:** Headline result (FID 1.54) uses **latent space** with a VAE (SD-VAE, 32×32×4 latents). The pixel-space experiments are secondary (FID ~1.6–1.8, Table 6).

**Code:** All configs (`imagenet_b2.yaml`, `imagenet_l2.yaml`) operate in **pixel space only** (`in_channels: 3`, `input_size: 256`). No VAE is present anywhere in the codebase. The latent-space pipeline cannot be reproduced.

---

### 5. Batch Size and Batch Construction
**Paper (Table 2 ablation):** Optimal batch: N_c=64 classes × N=64 samples per class = **B=4096 total**. Negative samples are all N_c×N_neg generated samples from the current batch.

**Code:** Flat DataLoader shuffle, not class-stratified. `imagenet_l2.yaml`: 96 positives + 48 uncond = **144 total**. Generated negatives = 96 samples. This is **~40× smaller** than the paper's batch, which Table 2 shows is critical for performance.

---

### 6. Multi-Scale Features (Eq. 14 / Appendix A.5)
**Paper (Appendix A.5, Eq. 14):** The loss is summed over **multiple encoder feature scales** (multiple ResNet stages):
```
L = Σ_j E[‖φ̃_j(x) − sg(φ̃_j(x) + Ṽ_j)‖²]
```

**Code:** Uses a **single feature vector** (DINOv2 CLS token — one global embedding). No multi-scale extraction. The `j` index in the code refers to temperatures, not feature scales.

---

### ~~7. Data Augmentation~~ ✅ FIXED
**Paper:** Augmentation not explicitly stated, but all competitive ImageNet generative models use random horizontal flip + RandomResizedCrop.

**Fix:** Training transform now uses `RandomResizedCrop(scale=(0.08,1.0)) + RandomHorizontalFlip`. Eval loader still uses `Resize + CenterCrop`.

---

### ~~8. Kernel Class Architecture (Dead Code)~~ ✅ FIXED
**Fix:** Removed `pairwise_weights` methods and the `_encode` helper with `torch.no_grad()` from all kernel classes. `DriftKernel` is now a plain `nn.Module` container (encoder + taus). The loss still computes the exponential kernel inline with proper feature normalization. Note: `GaussianKernel` config still maps to the exponential kernel in the loss — this is a known limitation (items 2/3/6 would need to be addressed to use the kernel type).

---

### 9. No FID Evaluation / Inference Pipeline
**Paper:** Reports FID on 50K generated images as the primary metric.

**Code:** `evaluate()` in `train.py` computes only training loss and V_norm on eval batches. No image generation at test time, no FID computation, no sampling script in the training pipeline. (`eval_fid.py` was added separately but is not part of the core pipeline.)

---

## Minor / Unclear Discrepancies

### ~~10. Drift Field Formula (Eq. 11)~~ ✅ CONFIRMED CORRECT
**Author's official toy code:**
```python
pos_coeff = nk[:, G:] * nk[:, :G].sum(dim=-1, keepdim=True)   # nk_pos * s_neg
neg_coeff = nk[:, :G] * nk[:, G:].sum(dim=-1, keepdim=True)   # nk_neg * s_pos
V = pos_coeff @ pos - neg_coeff @ gen
```
Exactly matches `(nk_pos * s_neg) @ fy_norm - (nk_neg * s_pos) @ fx_norm` in the code. **Not a discrepancy.**

---

### ~~11. Timestep Conditioning (Legacy)~~ ✅ FIXED
**Paper:** No diffusion timestep — the drifting generator maps ε → x in one forward pass.

**Fix:** `TimestepEmbedder` (t=0) replaced with a `nn.Parameter` learned bias `cond_bias` of shape `[hidden_size]`. Renamed `TimestepEmbedder` → `ScalarEmbedder` (still used for α CFG conditioning). Note: pretrained DiT checkpoints are no longer compatible.

---

### 12. Alpha Conditioning Mechanism
**Paper (§3.5):** States the generator is "conditioned on α" but does not specify the embedding method.

**Code:** Uses `TimestepEmbedder` (sinusoidal + 2-layer MLP) for α, added to the conditioning vector. Reasonable choice, not specified in the paper.

---

### 13. S_j Feature Normalization — Sample Set
**Paper (Appendix A.6):** `S_j = mean_batch(dist) / √C` [stop-grad]. Ambiguous which pairs of samples are included in the mean.

**Code:** Computes mean pairwise distance from x_gen to all of [x_gen, y_pos, y_unc] (excluding self-distances). Including uncond samples `y_unc` in S_j is an implementation choice that may not match the paper's intent.

---

### 14. Optimizer: beta2 and Weight Decay
**Paper (Appendix A.9):** Optimizer hyperparameters not accessible from HTML.

**Code:**
- `betas=(0.9, 0.9998)` — non-standard beta2 (PyTorch default is 0.999)
- `imagenet_b2.yaml`: `weight_decay: 0.0`
- `imagenet_l2.yaml`: `weight_decay: 1e-2`

Internal inconsistency across configs; cannot verify against paper.

---

### 15. CFG Uncond Negative Source
**Paper (§3.5):** Uncond negatives are real images from `p_data(∅)` (any class, unconditional).

**Code:** Uncond samples are simply the tail of the flat-shuffled DataLoader batch. Not class-stratified, so positives and uncond negatives may occasionally be from the same class.

---

### 16. EMA on CPU
**Paper:** EMA used for evaluation; decay not specified.

**Code:** EMA model kept on CPU to save GPU memory:
```python
ema_p.mul_(decay).add_(p.detach().cpu(), alpha=1 - decay)
```
Not described in paper; a memory optimization that introduces minor CPU↔GPU transfer overhead.

---

## Summary Table

| # | Topic | Paper | Code | Severity |
|---|-------|-------|------|----------|
| 1 | Kernel normalization | Two sequential softmaxes | Symmetric sqrt(row×col) | ✅ Correct |
| 2 | CFG α weighting | Distribution-level reweighting | Column scaling of k_raw before norm | Minor (principled) |
| 3 | Feature encoder | Custom latent-MAE (ResNet) | DINOv2 ViT-L/14 | **Critical** |
| 4 | Pixel vs. latent space | Latent (VAE) for headline result | Pixel space only | **Critical** |
| 5 | Batch size / structure | N_c=64 × N=64, B=4096 | Flat shuffle, B=144 | **Critical** |
| 6 | Multi-scale features | Σ_j loss over encoder stages | Single DINOv2 CLS token | **Critical** |
| 7 | Data augmentation | Not stated (standard expected) | ✅ RandomResizedCrop + HFlip | Fixed |
| 8 | Kernel class (dead code) | Unified kernel | ✅ Dead methods removed | Fixed |
| 9 | FID / sampling pipeline | 50K-image FID | No FID in training pipeline | **Critical** |
| 10 | Drift field V formula | Eq. 11 joint product | Factored marginal sums | ✅ Correct |
| 11 | Timestep t=0 | No timestep | ✅ Learned bias (cond_bias) | Fixed |
| 12 | α embedding mechanism | Not specified | TimestepEmbedder | Minor |
| 13 | S_j sample set | Ambiguous | Includes unc samples | Minor |
| 14 | beta2 / weight_decay | Not accessible | 0.9998; inconsistent across configs | Unclear |
| 15 | Uncond negative source | p_data(∅), any class | Tail of flat DataLoader batch | Minor |
| 16 | EMA on CPU | Not described | CPU EMA to save memory | Minor |

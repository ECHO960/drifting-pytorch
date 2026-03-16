"""
DiT backbone for the Drifting generator — modernized architecture.

Upgrades vs. original DiT (Peebles & Xie 2022):
  - RMSNorm    instead of LayerNorm (no affine params; adaLN provides scale/shift)
  - SwiGLU     instead of GELU MLP
  - 2D RoPE    applied to Q, K of image tokens (context tokens have no positional bias)
  - QK-Norm    (RMSNorm on Q and K before scaled dot-product attention)
  - adaLN-zero on all tokens (image + in-context conditioning)
  - In-context conditioning tokens prepended to the patch sequence

Maps noise ε [B, C, H, W] + class label → generated sample [B, C, H, W].
No diffusion timestep — a learned bias (cond_bias) anchors the conditioning.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


# ---------------------------------------------------------------------------
# RMSNorm (no learnable parameters — adaLN handles scale/shift)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


# ---------------------------------------------------------------------------
# 2D axial RoPE
# ---------------------------------------------------------------------------

def _rope_1d(
    seq_len: int, head_dim: int, base: float = 10000.0, device=None
) -> tuple[Tensor, Tensor]:
    """(cos, sin) each [seq_len, head_dim] for 1-D RoPE."""
    assert head_dim % 2 == 0
    half  = head_dim // 2
    theta = base ** (-torch.arange(0, half, dtype=torch.float32, device=device) / half)
    pos   = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, theta)                       # [seq_len, half]
    cos   = torch.cat([freqs.cos(), freqs.cos()], dim=-1) # [seq_len, head_dim]
    sin   = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def build_2d_rope(
    grid_h: int, grid_w: int, head_dim: int, device=None
) -> tuple[Tensor, Tensor]:
    """
    Axial 2D RoPE: first head_dim/2 dims encode row, last head_dim/2 encode col.
    Returns (cos, sin) each [grid_h * grid_w, head_dim].
    """
    half     = head_dim // 2
    cos_h, sin_h = _rope_1d(grid_h, half, device=device)  # [H, half]
    cos_w, sin_w = _rope_1d(grid_w, half, device=device)  # [W, half]
    row_ids  = torch.arange(grid_h, device=device).repeat_interleave(grid_w)
    col_ids  = torch.arange(grid_w, device=device).repeat(grid_h)
    cos      = torch.cat([cos_h[row_ids], cos_w[col_ids]], dim=-1)  # [T, head_dim]
    sin      = torch.cat([sin_h[row_ids], sin_w[col_ids]], dim=-1)
    return cos, sin


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: [B, heads, T, head_dim]; cos/sin: [T, head_dim] — broadcasts over B and heads."""
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Attention with QK-Norm and 2D RoPE
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head self-attention: QK-Norm + 2D RoPE on image tokens only."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.qkv  = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x:     Tensor,
        cos:   Tensor,
        sin:   Tensor,
        n_ctx: int,
    ) -> Tensor:
        """
        x:     [B, n_ctx + T, D]
        cos, sin: [T, head_dim] — applied to image tokens only
        n_ctx: number of prepended conditioning tokens
        """
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def _heads(t: Tensor) -> Tensor:
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = _heads(q), _heads(k), _heads(v)   # [B, heads, L, head_dim]

        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE on image tokens only (skip first n_ctx positions)
        if n_ctx < L:
            q_img = _apply_rope(q[:, :, n_ctx:], cos, sin)
            k_img = _apply_rope(k[:, :, n_ctx:], cos, sin)
            q = torch.cat([q[:, :, :n_ctx], q_img], dim=2)
            k = torch.cat([k[:, :, :n_ctx], k_img], dim=2)

        out = F.scaled_dot_product_attention(q, k, v)   # [B, heads, L, head_dim]
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU: hidden_dim ≈ 8/3 × dim keeps param count ≈ ratio-4 GELU MLP."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up   = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# Conditioning helpers
# ---------------------------------------------------------------------------

def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """adaLN: shift/scale [B, D] modulate sequence x [B, T, D]."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ScalarEmbedder(nn.Module):
    """Sinusoidal scalar → MLP embedding (used for α CFG conditioning)."""

    def __init__(self, hidden_size: int, freq_embed_size: int = 256):
        super().__init__()
        self.freq_embed_size = freq_embed_size
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def _sincos(t: Tensor, dim: int) -> Tensor:
        half  = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self._sincos(t, self.freq_embed_size))


class LabelEmbedder(nn.Module):
    """Class label → embedding."""

    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        self.embedding  = nn.Embedding(num_classes, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels: Tensor) -> Tensor:
        return self.embedding(labels)


# ---------------------------------------------------------------------------
# DiT block
# ---------------------------------------------------------------------------

def _round64(n: int) -> int:
    return ((n + 63) // 64) * 64


class DiTBlock(nn.Module):
    """
    DiT block: adaLN-zero (applied to all tokens) + Attention + SwiGLU.
    Conditioning enters via both adaLN modulation AND in-context tokens in the sequence.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        swiglu_hidden = _round64(int(mlp_ratio * 2 / 3 * hidden_size))
        self.norm1 = RMSNorm(hidden_size)
        self.attn  = Attention(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp   = SwiGLU(hidden_size, swiglu_hidden)
        # adaLN-zero: 6 outputs, initialized to 0 → identity at init
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(
        self,
        x:     Tensor,
        c:     Tensor,
        cos:   Tensor,
        sin:   Tensor,
        n_ctx: int,
    ) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            _modulate(self.norm1(x), shift_msa, scale_msa), cos, sin, n_ctx
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            _modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """Output projection with adaLN (image tokens only)."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm   = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN  = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return self.linear(_modulate(self.norm(x), shift, scale))


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

_CONFIGS = {
    "S/1":   dict(depth=12, hidden_size=384,  num_heads=6,  patch_size=1),
    "S/2":   dict(depth=12, hidden_size=384,  num_heads=6,  patch_size=2),
    "B/1":   dict(depth=12, hidden_size=768,  num_heads=12, patch_size=1),
    "B/2":   dict(depth=12, hidden_size=768,  num_heads=12, patch_size=2),
    "L/2":   dict(depth=24, hidden_size=1024, num_heads=16, patch_size=2),
    "XL/2":  dict(depth=28, hidden_size=1152, num_heads=16, patch_size=2),
    # Pixel-space variants: 256×256 input, patch_size=16 → 256 tokens
    "B/16":  dict(depth=12, hidden_size=768,  num_heads=12, patch_size=16),
    "L/16":  dict(depth=24, hidden_size=1024, num_heads=16, patch_size=16),
    "XL/16": dict(depth=28, hidden_size=1152, num_heads=16, patch_size=16),
}


# ---------------------------------------------------------------------------
# DiT model
# ---------------------------------------------------------------------------

class DiT(nn.Module):
    """
    Modernized DiT generator for the Drifting model.

    Conditioning enters via two paths simultaneously:
      1. adaLN-zero: combined c vector modulates shift/scale/gate of every norm layer.
      2. Separate in-context token banks: each conditioning variable (class label, α)
         has its own learnable token bank. The scalar embedding is broadcast-added to
         the bank and the result is prepended to the patch sequence, letting the
         transformer attend directly and independently to each conditioning signal.

    2D RoPE is applied only to image-patch tokens; context tokens are position-free.
    """

    def __init__(
        self,
        variant:          str  = "B/16",
        input_size:       int  = 256,
        in_channels:      int  = 3,
        num_classes:      int  = 1000,
        use_cfg:          bool = False,
        learn_sigma:      bool = False,
        num_class_tokens: int  = 8,
        num_cfg_tokens:   int  = 4,
    ):
        super().__init__()
        cfg = _CONFIGS[variant]
        self.patch_size       = cfg["patch_size"]
        self.hidden_size      = cfg["hidden_size"]
        self.num_heads        = cfg["num_heads"]
        self.depth            = cfg["depth"]
        self.in_channels      = in_channels
        self.out_channels     = in_channels * 2 if learn_sigma else in_channels
        self.input_size       = input_size
        self.grid_size        = input_size // self.patch_size
        self.num_patches      = self.grid_size ** 2
        self.num_class_tokens = num_class_tokens
        self.num_cfg_tokens   = num_cfg_tokens if use_cfg else 0
        self.n_ctx_tokens     = num_class_tokens + self.num_cfg_tokens

        # Patch embedding
        self.x_embedder = nn.Linear(
            self.patch_size * self.patch_size * in_channels, self.hidden_size, bias=True
        )

        # Conditioning embedders → scalar [B, D] used for adaLN and token bank bias
        self.cond_bias  = nn.Parameter(torch.zeros(self.hidden_size))
        self.y_embedder = (
            LabelEmbedder(num_classes, self.hidden_size) if num_classes > 0 else None
        )
        self.alpha_embedder = ScalarEmbedder(self.hidden_size) if use_cfg else None

        # Per-conditioning learnable token banks
        # Each bank: [num_tokens, D], init ~ N(0, 1/√D) following imfDiT
        std = self.hidden_size ** -0.5
        self.class_token_bank = nn.Parameter(torch.empty(num_class_tokens, self.hidden_size).normal_(std=std))
        self.cfg_token_bank   = (
            nn.Parameter(torch.empty(num_cfg_tokens, self.hidden_size).normal_(std=std))
            if use_cfg else None
        )

        # Transformer
        self.blocks = nn.ModuleList([
            DiTBlock(self.hidden_size, self.num_heads) for _ in range(self.depth)
        ])
        self.final_layer = FinalLayer(self.hidden_size, self.patch_size, self.out_channels)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic_init)

        # Patch embed
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))

        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding.weight, std=0.02)
        if self.alpha_embedder is not None:
            nn.init.normal_(self.alpha_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.alpha_embedder.mlp[2].weight, std=0.02)

    # ------------------------------------------------------------------
    def patchify(self, x: Tensor) -> Tensor:
        p = self.patch_size
        return rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)

    def unpatchify(self, x: Tensor, h: int, w: int) -> Tensor:
        p = self.patch_size
        return rearrange(
            x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=h // p, w=w // p, p1=p, p2=p
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x:     Tensor,
        label: Tensor | None = None,
        alpha: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x:     [B, C, H, W] input noise ε
            label: [B] integer class labels (None → unconditional)
            alpha: [B] CFG guidance scale α ≥ 1 (only used when use_cfg=True)
        Returns:
            [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Patch embed
        tokens = self.patchify(x)           # [B, T, p²C]
        tokens = self.x_embedder(tokens)    # [B, T, D]

        # Compute scalar embeddings for each conditioning variable
        cls_emb   = self.y_embedder(label) if (self.y_embedder is not None and label is not None) \
                    else torch.zeros(B, self.hidden_size, device=x.device)
        alpha_emb = self.alpha_embedder(1 - 1 / alpha.float()) \
                    if (self.alpha_embedder is not None and alpha is not None) \
                    else torch.zeros(B, self.hidden_size, device=x.device)

        # adaLN conditioning vector c [B, D]
        c = self.cond_bias.unsqueeze(0).expand(B, -1) + cls_emb + alpha_emb

        # In-context token banks: bank [num_tokens, D] + emb [B, 1, D] → [B, num_tokens, D]
        class_ctx = self.class_token_bank.unsqueeze(0) + cls_emb.unsqueeze(1)
        if self.cfg_token_bank is not None:
            cfg_ctx = self.cfg_token_bank.unsqueeze(0) + alpha_emb.unsqueeze(1)
            ctx = torch.cat([class_ctx, cfg_ctx], dim=1)   # [B, 8+4, D]
        else:
            ctx = class_ctx                                 # [B, 8, D]

        tokens = torch.cat([ctx, tokens], dim=1)            # [B, n_ctx + T, D]

        # 2D RoPE for image tokens
        head_dim = self.hidden_size // self.num_heads
        cos, sin = build_2d_rope(self.grid_size, self.grid_size, head_dim, device=x.device)
        # cos/sin: [T, head_dim] — broadcast over batch and heads in _apply_rope

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, c, cos, sin, self.n_ctx_tokens)

        # Strip context tokens, project image tokens to pixels
        img = tokens[:, self.n_ctx_tokens:]           # [B, T, D]
        img = self.final_layer(img, c)                # [B, T, p²·out_C]
        return self.unpatchify(img, H, W)             # [B, C, H, W]

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg) -> "DiT":
        return cls(
            variant=cfg.variant,
            input_size=cfg.get("input_size", 256),
            in_channels=cfg.in_channels,
            num_classes=cfg.get("num_classes", 1000),
            use_cfg=cfg.get("use_cfg", False),
            num_class_tokens=cfg.get("num_class_tokens", 8),
            num_cfg_tokens=cfg.get("num_cfg_tokens", 4),
        )

    def load_pretrained(self, path: str):
        state = torch.load(path, map_location="cpu")
        if "model" in state:
            state = state["model"]
        missing, unexpected = self.load_state_dict(state, strict=False)
        print(f"[DiT] Loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")

"""
DiT (Diffusion Transformer) backbone for the Drifting generator.

Architecture follows Peebles & Xie 2022 (arXiv 2212.09748).
Pretrained DiT weights can be loaded; the timestep embedding is kept
for weight compatibility but clamped to t=0 (unused in drifting).

Variants (patch_size refers to the spatial dimension in latent space):
  S/1, S/2, B/1, B/2, L/2, XL/2  — matching original DiT naming.
"""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep → MLP embedding (kept for weight compat)."""

    def __init__(self, hidden_size: int, freq_embed_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_embed_size = freq_embed_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: Tensor) -> Tensor:
        emb = self.timestep_embedding(t, self.freq_embed_size)
        return self.mlp(emb)


class LabelEmbedder(nn.Module):
    """Class label → embedding with optional dropout for CFG."""

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.0):
        super().__init__()
        use_cfg = dropout_prob > 0
        self.embedding = nn.Embedding(num_classes + (1 if use_cfg else 0), hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: Tensor) -> Tensor:
        drop = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        return torch.where(drop, torch.full_like(labels, self.num_classes), labels)

    def forward(self, labels: Tensor, force_drop: bool = False) -> Tensor:
        if self.training and self.dropout_prob > 0:
            labels = self.token_drop(labels)
        elif force_drop:
            labels = torch.full_like(labels, self.num_classes)
        return self.embedding(labels)


# ---------------------------------------------------------------------------
# DiT block
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """Single DiT block with adaLN-zero conditioning."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        # adaLN-zero: 6 parameters (shift/scale × 3) initialized to 0
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        # Self-attention
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """Output projection with adaLN conditioning."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ---------------------------------------------------------------------------
# DiT model
# ---------------------------------------------------------------------------

_CONFIGS = {
    # name       depth  hidden  heads  patch
    "S/1":   dict(depth=12, hidden_size=384,  num_heads=6,  patch_size=1),
    "S/2":   dict(depth=12, hidden_size=384,  num_heads=6,  patch_size=2),
    "B/1":   dict(depth=12, hidden_size=768,  num_heads=12, patch_size=1),
    "B/2":   dict(depth=12, hidden_size=768,  num_heads=12, patch_size=2),
    "L/2":   dict(depth=24, hidden_size=1024, num_heads=16, patch_size=2),
    "XL/2":  dict(depth=28, hidden_size=1152, num_heads=16, patch_size=2),
}


class DiT(nn.Module):
    """
    Diffusion Transformer generator for the Drifting model.

    Maps noise ε [B, C, H, W] + class label → generated sample x [B, C, H, W].
    Timestep conditioning is kept (t=0) for pretrained-weight compatibility.
    """

    def __init__(
        self,
        variant: str = "B/2",
        input_size: int = 32,        # spatial size of input (latent: 32, pixel: 256//patch)
        in_channels: int = 4,
        num_classes: int = 1000,
        cfg_dropout_prob: float = 0.0,
        learn_sigma: bool = False,   # False for drifting (direct x prediction)
    ):
        super().__init__()
        cfg = _CONFIGS[variant]
        self.patch_size  = cfg["patch_size"]
        self.hidden_size = cfg["hidden_size"]
        self.num_heads   = cfg["num_heads"]
        self.depth       = cfg["depth"]
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.input_size  = input_size
        self.num_patches = (input_size // self.patch_size) ** 2

        # Patch embedding
        self.x_embedder = nn.Linear(
            self.patch_size * self.patch_size * in_channels, self.hidden_size, bias=True
        )
        # Position embedding (fixed sinusoidal, learnable alt also works)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.hidden_size), requires_grad=False
        )

        # Conditioning
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        if num_classes > 0:
            self.y_embedder = LabelEmbedder(num_classes, self.hidden_size, cfg_dropout_prob)
        else:
            self.y_embedder = None

        # Transformer blocks
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

        # Sinusoidal pos embed
        pos = self._get_2d_sincos_pos_embed(
            self.hidden_size, int(self.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        # Patch embed std
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))

        # Timestep MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Label embed
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding.weight, std=0.02)

    @staticmethod
    def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int):
        import numpy as np
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid   = np.meshgrid(grid_w, grid_h)
        grid   = np.stack(grid, axis=0).reshape(2, -1)

        emb_h = DiT._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = DiT._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        return np.concatenate([emb_h, emb_w], axis=1)   # [T, D]

    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos):
        import numpy as np
        omega  = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim // 2)
        omega  = 1.0 / 10000 ** omega
        out    = np.einsum("m,d->md", pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)

    # ------------------------------------------------------------------
    def patchify(self, x: Tensor) -> Tensor:
        """[B, C, H, W] → [B, T, patch_size²·C]"""
        p = self.patch_size
        return rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)

    def unpatchify(self, x: Tensor, h: int, w: int) -> Tensor:
        """[B, T, patch_size²·C] → [B, C, H, W]"""
        p = self.patch_size
        return rearrange(
            x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=h // p, w=w // p, p1=p, p2=p
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        y: Tensor | None = None,
        force_drop_label: bool = False,
    ) -> Tensor:
        """
        Args:
            x: [B, C, H, W] input noise ε
            y: [B] integer class labels (or None for unconditional)
            force_drop_label: force unconditional (CFG inference)
        Returns:
            out: [B, C, H, W] generated sample
        """
        B, C, H, W = x.shape

        # Patch embed + pos
        tokens = self.patchify(x)              # [B, T, p²C]
        tokens = self.x_embedder(tokens)       # [B, T, D]
        tokens = tokens + self.pos_embed       # broadcast

        # Conditioning: t=0 (fixed) + class label
        t = torch.zeros(B, dtype=torch.long, device=x.device)
        c = self.t_embedder(t)                 # [B, D]
        if self.y_embedder is not None and y is not None:
            c = c + self.y_embedder(y, force_drop=force_drop_label)

        # Transformer
        for block in self.blocks:
            tokens = block(tokens, c)

        # Output projection
        tokens = self.final_layer(tokens, c)   # [B, T, p²·out_C]
        out    = self.unpatchify(tokens, H, W) # [B, out_C, H, W]
        return out

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg) -> "DiT":
        return cls(
            variant=cfg.variant,
            in_channels=cfg.in_channels,
            num_classes=cfg.get("num_classes", 1000),
            cfg_dropout_prob=0.0,
        )

    def load_pretrained(self, path: str):
        """Load pretrained DiT weights (official release format)."""
        state = torch.load(path, map_location="cpu")
        # Official checkpoint may be wrapped in 'model' key
        if "model" in state:
            state = state["model"]
        missing, unexpected = self.load_state_dict(state, strict=False)
        print(f"[DiT] Loaded pretrained: {len(missing)} missing, {len(unexpected)} unexpected keys")

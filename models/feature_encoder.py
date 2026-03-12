"""
Feature encoder registry for the drift kernel.

The encoder φ maps raw inputs (images / latents) to feature vectors
used to compute kernel distances.  All encoders are always frozen.

Usage:
    enc = build_encoder(cfg.encoder)   # cfg.encoder.type = "dinov2_b"
    feats = enc(images)                # [N, F]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Base wrapper — ensures frozen eval mode
# ---------------------------------------------------------------------------

class FrozenEncoder(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        # Always keep backbone in eval
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return self._forward(x)

    def _forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# DINOv2
# ---------------------------------------------------------------------------

class DINOv2Encoder(FrozenEncoder):
    """DINOv2 ViT — outputs CLS token features [N, F]."""

    def __init__(self, model_name: str = "dinov2_vitb14"):
        backbone = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True, verbose=False
        )
        super().__init__(backbone)
        self.out_dim = backbone.embed_dim

    def _forward(self, x: Tensor) -> Tensor:
        # DINOv2 expects images in [0,1] or [-1,1] (224×224 optimal, but flexible)
        return self.backbone(x)     # CLS token: [N, F]


# ---------------------------------------------------------------------------
# Identity (for low-dim inputs like robot actions)
# ---------------------------------------------------------------------------

class IdentityEncoder(nn.Module):
    """Pass-through — use raw input as features."""

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(1)


# ---------------------------------------------------------------------------
# Registry & builder
# ---------------------------------------------------------------------------

ENCODER_REGISTRY: dict[str, type] = {
    "dinov2_b":  lambda: DINOv2Encoder("dinov2_vitb14"),
    "dinov2_l":  lambda: DINOv2Encoder("dinov2_vitl14"),
    "dinov2_s":  lambda: DINOv2Encoder("dinov2_vits14"),
    "identity":  lambda: IdentityEncoder(),
}


def build_encoder(cfg) -> nn.Module | None:
    """
    Build encoder from OmegaConf config.
    Returns None if cfg.type == 'identity' (kernel operates in raw space).
    """
    enc_type = cfg.type
    if enc_type not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder type '{enc_type}'. "
                         f"Available: {list(ENCODER_REGISTRY)}")
    if enc_type == "identity":
        return None   # sentinel: kernel uses raw input
    return ENCODER_REGISTRY[enc_type]()

"""
Thin wrapper around Stable Diffusion VAE.
Used to encode images → latents (training) and decode latents → images (sampling).
Always frozen; not part of the trainable generator.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class VAEWrapper(nn.Module):
    """
    Wraps diffusers AutoencoderKL.

    encode: pixel image [B,3,H,W] → latent [B,4,H/8,W/8]
    decode: latent [B,4,h,w]      → pixel image [B,3,H,W]
    """

    def __init__(self, model_id: str = "stabilityai/sd-vae-ft-ema", scale: float = 0.18215):
        super().__init__()
        from diffusers import AutoencoderKL
        self.vae   = AutoencoderKL.from_pretrained(model_id)
        self.scale = scale
        for p in self.parameters():
            p.requires_grad_(False)
        self.vae.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()   # always keep frozen
        return self

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """x: [B, 3, H, W] in [-1, 1] → latent [B, 4, H/8, W/8]"""
        dist = self.vae.encode(x).latent_dist
        return dist.sample() * self.scale

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        """z: [B, 4, h, w] → image [B, 3, H, W] in [-1, 1]"""
        return self.vae.decode(z / self.scale).sample

    @classmethod
    def from_config(cls, cfg) -> "VAEWrapper":
        return cls(model_id=cfg.model_id, scale=cfg.latent_scale)

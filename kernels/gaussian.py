"""Gaussian (RBF) kernel: k(x, y) = exp(-‖φ(x) − φ(y)‖² / 2τ²)"""

import torch
import torch.nn as nn
from torch import Tensor

from .base import DriftKernel


class GaussianKernel(DriftKernel):
    """
    k(x, y) = exp( -‖φ(x) − φ(y)‖₂² / (2τ²) )

    Smoother than ExponentialKernel; less sensitive to outliers.
    """

    def __init__(self, encoder: nn.Module | None = None, tau: float = 0.05):
        super().__init__()
        self.encoder = encoder
        self.tau = tau

    def pairwise_weights(self, x: Tensor, y: Tensor) -> Tensor:
        fx = self._encode(x, self.encoder)   # [N, F]
        fy = self._encode(y, self.encoder)   # [M, F]
        dists_sq = torch.cdist(fx.float(), fy.float(), p=2).pow(2)
        return torch.exp(-dists_sq / (2 * self.tau ** 2))

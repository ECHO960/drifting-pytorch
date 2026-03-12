"""Exponential kernel: k(x, y) = exp(-‖φ(x) − φ(y)‖ / τ)"""

import torch
import torch.nn as nn
from torch import Tensor

from .base import DriftKernel


class ExponentialKernel(DriftKernel):
    """
    k(x, y) = exp( -‖φ(x) − φ(y)‖₂ / τ )

    φ is an optional frozen feature encoder (e.g. DINOv2).
    If encoder is None, kernel is computed directly in raw input space.
    """

    def __init__(self, encoder: nn.Module | None = None, tau: float = 0.05):
        super().__init__()
        self.encoder = encoder
        self.tau = tau

    def pairwise_weights(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            x: [N, *] query samples
            y: [M, *] key samples
        Returns:
            weights: [N, M]
        """
        fx = self._encode(x, self.encoder)   # [N, F]
        fy = self._encode(y, self.encoder)   # [M, F]
        # torch.cdist: [N, M]  (p=2 = Euclidean)
        dists = torch.cdist(fx.float(), fy.float(), p=2)
        return torch.exp(-dists / self.tau)

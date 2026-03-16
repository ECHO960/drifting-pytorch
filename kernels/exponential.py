"""Exponential kernel container: k(x, y) = exp(-‖φ(x) − φ(y)‖ / τ)

The actual kernel computation is performed inline in losses/drifting.py with
feature normalization (§A.5). This class stores the encoder and taus used there.
"""

import torch.nn as nn

from .base import DriftKernel


class ExponentialKernel(DriftKernel):
    def __init__(self, encoder: nn.Module | None = None, tau: float = 0.05):
        super().__init__()
        self.encoder = encoder
        self.tau = tau

"""Gaussian (RBF) kernel container: k(x, y) = exp(-‖φ(x) − φ(y)‖² / 2τ²)

The actual kernel computation is performed inline in losses/drifting.py with
feature normalization (§A.5). This class stores the encoder and taus used there.
Note: the loss always uses the exponential form; this variant is available for
future use if the loss is updated to honour kernel.type.
"""

import torch.nn as nn

from .base import DriftKernel


class GaussianKernel(DriftKernel):
    def __init__(self, encoder: nn.Module | None = None, tau: float = 0.05):
        super().__init__()
        self.encoder = encoder
        self.tau = tau

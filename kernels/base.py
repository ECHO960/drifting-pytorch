"""Abstract base class for drift kernels."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch import Tensor


class DriftKernel(nn.Module, ABC):
    """
    A drift kernel computes pairwise weights k(x_i, y_j) for all pairs
    in two batches x [N, *D] and y [M, *D].

    Weights are unnormalized; callers normalize as needed.
    """

    @abstractmethod
    def pairwise_weights(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            x: [N, D] or [N, C, H, W] — query samples
            y: [M, D] or [M, C, H, W] — key samples
        Returns:
            weights: [N, M] unnormalized kernel weights
        """

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.pairwise_weights(x, y)

    # ------------------------------------------------------------------
    # Helpers shared across subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(x: Tensor) -> Tensor:
        """Flatten all dims except batch to [N, D]."""
        return x.flatten(1)

    @staticmethod
    def _encode(x: Tensor, encoder: nn.Module | None) -> Tensor:
        """Apply encoder if provided; otherwise flatten raw input."""
        if encoder is None:
            return DriftKernel._flatten(x)
        with torch.no_grad():
            return encoder(x)

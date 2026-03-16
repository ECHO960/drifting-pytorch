"""Abstract base class for drift kernels."""

import torch.nn as nn
from torch import Tensor


class DriftKernel(nn.Module):
    """
    Container for the encoder and temperature parameters used by the drifting loss.

    The loss (losses/drifting.py) reads kernel.encoder and kernel.taus directly
    and computes the exponential kernel inline. The pairwise_weights method is
    intentionally not defined here — keeping the kernel computation in one place.
    """

    @staticmethod
    def _flatten(x: Tensor) -> Tensor:
        """Flatten all dims except batch to [N, D]."""
        return x.flatten(1)

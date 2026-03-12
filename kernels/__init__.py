from .base import DriftKernel
from .exponential import ExponentialKernel
from .gaussian import GaussianKernel

KERNEL_REGISTRY = {
    "exponential": ExponentialKernel,
    "gaussian": GaussianKernel,
}


def build_kernel(cfg, encoder=None):
    """Build a DriftKernel from OmegaConf config."""
    cls = KERNEL_REGISTRY[cfg.type]
    return cls(encoder=encoder, tau=cfg.tau)


__all__ = ["DriftKernel", "ExponentialKernel", "GaussianKernel", "build_kernel"]

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
    # Support single tau or list of taus (multiple temperatures, §A.6)
    taus = list(cfg.taus) if cfg.get("taus") else [cfg.tau]
    kernel = cls(encoder=encoder, tau=taus[0])
    kernel.taus = taus
    return kernel


__all__ = ["DriftKernel", "ExponentialKernel", "GaussianKernel", "build_kernel"]

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor


class _GatherLayer(torch.autograd.Function):
    """All-gather with gradient support for the local rank's slice."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        if not (dist.is_available() and dist.is_initialized()):
            return (x,)
        out = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(out, x.contiguous())
        return tuple(out)

    @staticmethod
    def backward(ctx, *grads):
        x, = ctx.saved_tensors
        if not (dist.is_available() and dist.is_initialized()):
            return grads[0]
        grad = torch.stack(grads)
        dist.all_reduce(grad)
        return grad[dist.get_rank()]


def all_gather(x: Tensor) -> Tensor:
    """Gather x from all ranks → [world*N, C]. Grad flows to local slice."""
    return torch.cat(_GatherLayer.apply(x), dim=0)


def all_gather_nograd(x: Tensor) -> Tensor:
    """Gather x from all ranks → [world*N, C]. No gradient."""
    if not (dist.is_available() and dist.is_initialized()):
        return x
    out = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(out, x.contiguous())
    return torch.cat(out, dim=0)


def rank() -> int:
    return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0


def get_rank_slice(tensor: Tensor, local_size: int) -> Tensor:
    r = rank()
    return tensor[r * local_size:(r + 1) * local_size]

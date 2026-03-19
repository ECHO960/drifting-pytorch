"""
MNIST data loading for Drifting sanity-check experiments.

Uses ClassStratifiedBatchSampler (same logic as imagenet.py):
each batch has n_pos images from ONE class + n_unc from random other classes.
Images are returned in [-1, 1], shape [1, 28, 28].
"""

from __future__ import annotations

import random
from collections import defaultdict

import torchvision.datasets as dsets
import torchvision.transforms as T
from torch.utils.data import DataLoader


class ClassStratifiedBatchSampler:
    def __init__(self, labels, n_pos: int, n_unc: int):
        idx_by_class: dict[int, list[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            idx_by_class[int(lbl)].append(i)
        self.idx_by_class = dict(idx_by_class)
        self.classes = sorted(self.idx_by_class.keys())
        self.n_pos = n_pos
        self.n_unc = n_unc

    def __iter__(self):
        for _ in range(len(self)):
            c = random.choice(self.classes)
            pos_idx = random.choices(self.idx_by_class[c], k=self.n_pos)
            others = [x for x in self.classes if x != c]
            unc_idx = [
                random.choice(self.idx_by_class[random.choice(others)])
                for _ in range(self.n_unc)
            ]
            yield pos_idx + unc_idx

    def __len__(self) -> int:
        total = sum(len(v) for v in self.idx_by_class.values())
        return max(1, total // (self.n_pos + self.n_unc))


def build_mnist_loader(cfg):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),   # [0,1] → [-1,1]
    ])
    root = cfg.get("root", "data/mnist")
    dataset = dsets.MNIST(root=root, train=True, download=True, transform=transform)

    # Single-GPU: use random mixed-class batches so same_class_mask is sparse
    # (with all images from one class, k_neg=0 → V=0 → no training signal).
    # Multi-GPU with class_stratified=true: use ClassStratifiedBatchSampler instead.
    batch_size = cfg.get("batch_size", 64)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=True,
        drop_last=True,
    )
    return loader


def build_mnist_eval_loader(cfg):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    root = cfg.get("root", "data/mnist")
    dataset = dsets.MNIST(root=root, train=False, download=True, transform=transform)
    return DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", cfg.get("n_samples_per_class", 64) + cfg.get("n_uncond", 0)),
        shuffle=False,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=True,
        drop_last=True,
    )

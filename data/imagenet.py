"""
ImageNet dataset with class-stratified batching.

The drifting loss requires N_pos ≥ 64 positive samples per class per batch
(see Table 2, paper).  ClassStratifiedSampler ensures each mini-batch
contains exactly n_classes_per_batch × n_samples_per_class samples,
all grouped by class label.

Usage:
    dataset = ImageNetLatentDataset(root, vae, split='train')
    sampler = ClassStratifiedSampler(dataset, n_classes=8, n_per_class=64)
    loader  = DataLoader(dataset, batch_sampler=sampler, num_workers=8)
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


# ---------------------------------------------------------------------------
# Dataset: raw ImageNet → on-the-fly VAE encode
# ---------------------------------------------------------------------------

class ImageNetDataset(Dataset):
    """
    Standard ImageNet folder dataset.
    Returns (image_tensor [3,H,W] in [-1,1], class_label).
    Does NOT encode to latent here — VAE encode is done in the training loop
    (easier to batch and keep on GPU).
    """

    MEAN = [0.5, 0.5, 0.5]
    STD  = [0.5, 0.5, 0.5]

    def __init__(self, root: str, split: str = "train", image_size: int = 256):
        self.inner = ImageFolder(
            os.path.join(root, split),
            transform=T.Compose([
                T.Resize(image_size, interpolation=T.InterpolationMode.LANCZOS),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=self.MEAN, std=self.STD),
            ]),
        )
        # Build class → index list for stratified sampler
        self.class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, (_, label) in enumerate(self.inner.samples):
            self.class_to_indices[label].append(idx)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        img, label = self.inner[idx]
        return img, label

    @property
    def num_classes(self):
        return len(self.inner.classes)


# ---------------------------------------------------------------------------
# Sampler: class-stratified mini-batches
# ---------------------------------------------------------------------------

class ClassStratifiedSampler(Sampler):
    """
    Each batch = n_classes_per_batch × n_per_class samples.
    Classes are sampled without replacement per epoch; samples within
    each class are sampled with replacement if needed.

    Args:
        dataset:            ImageNetDataset (or any with .class_to_indices)
        n_classes_per_batch: number of distinct classes per batch
        n_per_class:         samples per class per batch
        steps:               total training steps (determines epoch length)
    """

    def __init__(
        self,
        dataset: ImageNetDataset,
        n_classes_per_batch: int = 8,
        n_per_class: int = 64,
        steps: int = 400_000,
    ):
        self.class_to_indices = dataset.class_to_indices
        self.all_classes      = list(self.class_to_indices.keys())
        self.n_classes        = n_classes_per_batch
        self.n_per_class      = n_per_class
        self.steps            = steps

    def __len__(self):
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.steps):
            classes = random.sample(self.all_classes, self.n_classes)
            batch_indices = []
            for cls in classes:
                pool = self.class_to_indices[cls]
                # sample with replacement if pool is smaller than needed
                if len(pool) >= self.n_per_class:
                    chosen = random.sample(pool, self.n_per_class)
                else:
                    chosen = random.choices(pool, k=self.n_per_class)
                batch_indices.extend(chosen)
            yield batch_indices


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_imagenet_loader(cfg, rank: int = 0, world_size: int = 1):
    """
    Build ImageNet DataLoader with class-stratified sampler.

    For DDP, each rank gets its own sampler slice via DistributedClassStratifiedSampler.
    """
    from torch.utils.data import DataLoader

    dataset = ImageNetDataset(
        root=cfg.root,
        split="train",
        image_size=cfg.image_size,
    )

    if world_size > 1:
        sampler = DistributedClassStratifiedSampler(
            dataset,
            n_classes_per_batch=cfg.n_classes_per_batch,
            n_per_class=cfg.n_samples_per_class,
            steps=cfg.get("steps", 400_000),
            rank=rank,
            world_size=world_size,
        )
    else:
        sampler = ClassStratifiedSampler(
            dataset,
            n_classes_per_batch=cfg.n_classes_per_batch,
            n_per_class=cfg.n_samples_per_class,
            steps=cfg.get("steps", 400_000),
        )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.get("num_workers", 8),
        pin_memory=True,
    )
    return loader, dataset.num_classes


# ---------------------------------------------------------------------------
# DDP-aware sampler: each rank handles a disjoint class subset per batch
# ---------------------------------------------------------------------------

class DistributedClassStratifiedSampler(Sampler):
    """
    Multi-GPU version: splits the n_classes_per_batch × world_size classes
    across ranks, so all ranks together see the full global batch.
    """

    def __init__(
        self,
        dataset: ImageNetDataset,
        n_classes_per_batch: int,
        n_per_class: int,
        steps: int,
        rank: int,
        world_size: int,
    ):
        self.class_to_indices = dataset.class_to_indices
        self.all_classes      = list(self.class_to_indices.keys())
        self.n_classes        = n_classes_per_batch
        self.n_per_class      = n_per_class
        self.steps            = steps
        self.rank             = rank
        self.world_size       = world_size

    def __len__(self):
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(42)   # deterministic across ranks for shared class list
        for step in range(self.steps):
            # All ranks share the same RNG state → same class order per step
            total_classes = self.n_classes * self.world_size
            classes = rng.sample(self.all_classes, total_classes)
            my_classes = classes[self.rank * self.n_classes:(self.rank + 1) * self.n_classes]

            batch_indices = []
            for cls in my_classes:
                pool = self.class_to_indices[cls]
                if len(pool) >= self.n_per_class:
                    chosen = random.sample(pool, self.n_per_class)
                else:
                    chosen = random.choices(pool, k=self.n_per_class)
                batch_indices.extend(chosen)
            yield batch_indices

"""
ImageNet data loading for Drifting training.

Loads local parquet shards (HF download format) as a map-style dataset.
Training uses ClassStratifiedBatchSampler: each batch contains n_pos images
from ONE randomly chosen class (true same-class positives) plus n_unc images
from random other classes (unconditional negatives for CFG).
"""

from __future__ import annotations

import glob
import os
import random
import warnings
from collections import defaultdict

import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


def _get_transform(image_size: int, train: bool = True) -> T.Compose:
    if train:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.08, 1.0), interpolation=T.InterpolationMode.LANCZOS),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.LANCZOS),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


warnings.filterwarnings("ignore", category=UserWarning, module="PIL")


class ParquetImageNet(Dataset):
    def __init__(self, parquet_files: list[str], image_size: int, train: bool = True):
        from datasets import load_dataset
        self.ds = load_dataset("parquet", data_files=parquet_files, split="train")
        self.transform = _get_transform(image_size, train=train)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        try:
            sample = self.ds[idx]
            img = self.transform(sample["image"].convert("RGB"))
            # Reject NaN / degenerate images (e.g. all-black, all-white after norm)
            if not img.isfinite().all() or img.std() < 1e-3:
                return self[idx - 1]
            return img, sample["label"]
        except Exception:
            # Corrupt image — return a neighbour so the batch is never stalled.
            return self[idx - 1]


class ClassStratifiedBatchSampler:
    """
    Each batch: n_samples_per_class images from each of n_classes_per_batch randomly
    chosen classes, plus n_unc images from other classes (unconditional negatives).

    Total batch size = n_classes_per_batch * n_samples_per_class + n_unc.

    For DDP, each rank runs its own independent sampler — separate processes have
    independent random states, so ranks naturally select different classes each step.
    """

    def __init__(self, labels, n_classes_per_batch: int, n_samples_per_class: int, n_unc: int):
        idx_by_class: dict[int, list[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            idx_by_class[int(lbl)].append(i)
        self.idx_by_class = dict(idx_by_class)
        self.classes = sorted(self.idx_by_class.keys())
        self.n_classes_per_batch = n_classes_per_batch
        self.n_samples_per_class = n_samples_per_class
        self.n_unc = n_unc

    def __iter__(self):
        for _ in range(len(self)):
            chosen = random.sample(self.classes, k=self.n_classes_per_batch)
            pos_idx = []
            for c in chosen:
                pos_idx += random.choices(self.idx_by_class[c], k=self.n_samples_per_class)
            others = [x for x in self.classes if x not in set(chosen)]
            unc_idx = [
                random.choice(self.idx_by_class[random.choice(others)])
                for _ in range(self.n_unc)
            ]
            yield pos_idx + unc_idx

    def __len__(self) -> int:
        total = sum(len(v) for v in self.idx_by_class.values())
        batch_size = self.n_classes_per_batch * self.n_samples_per_class + self.n_unc
        return max(1, total // batch_size)


def _make_train_loader(parquet_files, cfg, num_workers: int):
    dataset = ParquetImageNet(parquet_files, cfg.image_size, train=True)
    n_pos = cfg.n_samples_per_class
    n_unc = cfg.get("n_uncond", n_pos)

    if cfg.get("class_stratified", True):
        n_classes_per_batch = cfg.get("n_classes_per_batch", 1)
        batch_size = n_classes_per_batch * n_pos + n_unc
        print(f"[data] class-stratified batching — {n_classes_per_batch} classes × {n_pos} samples + {n_unc} uncond = {batch_size} per GPU", flush=True)
        print(f"[data] building class index ({len(dataset)} images)…", flush=True)
        batch_sampler = ClassStratifiedBatchSampler(dataset.ds["label"], n_classes_per_batch, n_pos, n_unc)
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        print(f"[data] random mixed-class batching", flush=True)
        loader = DataLoader(
            dataset,
            batch_size=n_pos + n_unc,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    return loader


def _make_eval_loader(parquet_files, cfg, rank, world_size, num_workers: int):
    dataset = ParquetImageNet(parquet_files, cfg.image_size, train=False)
    sampler = None
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=cfg.n_samples_per_class + cfg.get("n_uncond", cfg.n_samples_per_class),
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def build_imagenet_loader(cfg, rank: int = 0, world_size: int = 1):
    """
    Returns (loader, None, num_classes).
    The class-stratified batch sampler handles class selection; no DistributedSampler
    is needed — each DDP rank runs its own independent sampler (different class per step).
    """
    files = sorted(glob.glob(os.path.join(cfg.root, "data", "train-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No train parquet files found under {cfg.root}/data/")
    num_workers = cfg.get("num_workers", 8)
    loader = _make_train_loader(files, cfg, num_workers)
    return loader, None, 1000


def build_imagenet_eval_loader(cfg, rank: int = 0, world_size: int = 1):
    """Returns eval loader over val/test split. Returns None if not found."""
    files = sorted(glob.glob(os.path.join(cfg.root, "data", "val-*.parquet")))
    if not files:
        files = sorted(glob.glob(os.path.join(cfg.root, "data", "test-*.parquet")))
    if not files:
        return None
    num_workers = cfg.get("num_workers", 8)
    return _make_eval_loader(files, cfg, rank, world_size, num_workers)

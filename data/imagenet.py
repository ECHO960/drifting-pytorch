"""
ImageNet data loading for Drifting training.

Loads local parquet shards (HF download format) as a map-style dataset,
wrapped in a standard PyTorch DataLoader with shuffle.
"""

from __future__ import annotations

import glob
import os
import warnings

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


def _make_loader(parquet_files, cfg, rank, world_size, shuffle, train: bool = True):
    dataset = ParquetImageNet(parquet_files, cfg.image_size, train=train)
    sampler = None
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    loader = DataLoader(
        dataset,
        batch_size=cfg.n_samples_per_class + cfg.get("n_uncond", cfg.n_samples_per_class),
        sampler=sampler,
        shuffle=(sampler is None and shuffle),
        num_workers=cfg.get("num_workers", 8),
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler


def build_imagenet_loader(cfg, rank: int = 0, world_size: int = 1):
    """Returns (loader, sampler, num_classes). Call sampler.set_epoch(epoch) each epoch for DDP."""
    files = sorted(glob.glob(os.path.join(cfg.root, "data", "train-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No train parquet files found under {cfg.root}/data/")
    loader, sampler = _make_loader(files, cfg, rank, world_size, shuffle=True, train=True)
    return loader, sampler, 1000


def build_imagenet_eval_loader(cfg, rank: int = 0, world_size: int = 1):
    """Returns eval loader over test split (ImageNet val). Returns None if not found."""
    files = sorted(glob.glob(os.path.join(cfg.root, "data", "val-*.parquet")))
    if not files:
        files = sorted(glob.glob(os.path.join(cfg.root, "data", "test-*.parquet")))
    if not files:
        return None
    loader, _ = _make_loader(files, cfg, rank, world_size, shuffle=False, train=False)
    return loader

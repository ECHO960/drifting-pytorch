"""
Robotics dataset — Diffusion Policy Zarr format.

Compatible with data collected by https://github.com/real-stanford/diffusion_policy

Zarr store layout:
  data/
    obs:         [T_total, obs_dim]   float32
    action:      [T_total, action_dim] float32
  meta/
    episode_ends: [N_episodes]         int64  (cumulative end indices)

Each sample is a (obs_seq, action_seq) pair of length `horizon`.
Task ID (integer) is used as the "class label" for the drifting kernel.

To use multiple tasks: set cfg.zarr_path to a glob or list, and each
task will get a unique integer ID assigned at load time.

Download data to:  data/robotics/<task_name>.zarr
"""

from __future__ import annotations

import os
import glob
from typing import Iterator

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler, DataLoader


# ---------------------------------------------------------------------------
# Single-task Zarr dataset
# ---------------------------------------------------------------------------

class ZarrEpisodeDataset(Dataset):
    """
    Loads one zarr store and exposes (obs_chunk, action_chunk, task_id) triples.

    Args:
        zarr_path:   path to <task>.zarr
        obs_key:     zarr key for observations
        action_key:  zarr key for actions
        horizon:     action sequence length to return
        pad_before:  replicate first obs N times at start of episode
        pad_after:   replicate last obs N times at end of episode
        task_id:     integer label used as "class" in the drifting kernel
    """

    def __init__(
        self,
        zarr_path: str,
        obs_key: str = "obs",
        action_key: str = "action",
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        task_id: int = 0,
    ):
        import zarr
        store = zarr.open(zarr_path, mode="r")

        self.obs    = np.array(store["data"][obs_key])        # [T, obs_dim]
        self.action = np.array(store["data"][action_key])     # [T, action_dim]
        ep_ends     = np.array(store["meta"]["episode_ends"]) # [N_ep]

        self.horizon    = horizon
        self.pad_before = pad_before
        self.pad_after  = pad_after
        self.task_id    = task_id

        # Build list of (ep_start, ep_end) indices
        ep_starts = np.concatenate([[0], ep_ends[:-1]])
        self.episodes = list(zip(ep_starts.tolist(), ep_ends.tolist()))

        # Pre-compute valid sample indices: (global_start, ep_idx)
        self.indices: list[tuple[int, int]] = []
        for ep_idx, (ep_start, ep_end) in enumerate(self.episodes):
            ep_len = ep_end - ep_start
            for t in range(ep_len):
                self.indices.append((ep_start + t, ep_idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        t_global, ep_idx = self.indices[idx]
        ep_start, ep_end = self.episodes[ep_idx]

        # Obs: single step (extendable to obs_horizon if needed)
        obs = torch.from_numpy(self.obs[t_global].copy()).float()  # [obs_dim]

        # Action chunk [horizon, action_dim] with boundary padding
        action_seq = []
        for h in range(self.horizon):
            t = t_global + h
            t = max(ep_start, min(ep_end - 1, t))   # clamp to episode bounds
            action_seq.append(self.action[t])
        action = torch.from_numpy(np.stack(action_seq)).float()    # [H, action_dim]

        return {
            "obs":     obs,
            "action":  action,
            "task_id": torch.tensor(self.task_id, dtype=torch.long),
        }

    @property
    def action_dim(self) -> int:
        return self.action.shape[-1]

    @property
    def obs_dim(self) -> int:
        return self.obs.shape[-1]


# ---------------------------------------------------------------------------
# Multi-task dataset (concatenates multiple zarr stores)
# ---------------------------------------------------------------------------

class MultiTaskRoboticsDataset(Dataset):
    """Concatenates multiple ZarrEpisodeDatasets, one per task."""

    def __init__(self, task_datasets: list[ZarrEpisodeDataset]):
        self.datasets   = task_datasets
        self.cumlen     = np.cumsum([len(d) for d in task_datasets])
        self.num_tasks  = len(task_datasets)

        # Build task → sample indices for stratified sampling
        self.task_to_indices: dict[int, list[int]] = {}
        offset = 0
        for d in task_datasets:
            tid = d.task_id
            self.task_to_indices[tid] = list(range(offset, offset + len(d)))
            offset += len(d)

    def __len__(self):
        return int(self.cumlen[-1])

    def __getitem__(self, idx: int) -> dict:
        ds_idx = int(np.searchsorted(self.cumlen, idx, side="right"))
        offset = int(self.cumlen[ds_idx - 1]) if ds_idx > 0 else 0
        return self.datasets[ds_idx][idx - offset]


# ---------------------------------------------------------------------------
# Task-stratified sampler (analog of ClassStratifiedSampler for robotics)
# ---------------------------------------------------------------------------

class TaskStratifiedSampler(Sampler):
    """
    Each batch = n_tasks_per_batch × n_per_task samples.
    Ensures the drifting kernel sees enough positives per task.
    """

    def __init__(
        self,
        dataset: MultiTaskRoboticsDataset,
        n_tasks_per_batch: int = 4,
        n_per_task: int = 32,
        steps: int = 100_000,
    ):
        import random as _random
        self._random = _random
        self.task_to_indices = dataset.task_to_indices
        self.all_tasks       = list(self.task_to_indices.keys())
        self.n_tasks         = min(n_tasks_per_batch, len(self.all_tasks))
        self.n_per_task      = n_per_task
        self.steps           = steps

    def __len__(self):
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.steps):
            tasks = self._random.sample(self.all_tasks, self.n_tasks)
            batch = []
            for t in tasks:
                pool = self.task_to_indices[t]
                chosen = (
                    self._random.sample(pool, self.n_per_task)
                    if len(pool) >= self.n_per_task
                    else self._random.choices(pool, k=self.n_per_task)
                )
                batch.extend(chosen)
            yield batch


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_robotics_loader(cfg, rank: int = 0, world_size: int = 1):
    """
    Build robotics DataLoader from config.

    cfg.zarr_path can be:
      - a single filename:  "pusht.zarr"
      - a glob pattern:     "*.zarr"
      - a list:             ["pusht.zarr", "lift.zarr"]
    """
    root = cfg.root
    zarr_path_cfg = cfg.zarr_path

    # Resolve zarr paths
    if isinstance(zarr_path_cfg, str):
        if "*" in zarr_path_cfg:
            paths = sorted(glob.glob(os.path.join(root, zarr_path_cfg)))
        else:
            paths = [os.path.join(root, zarr_path_cfg)]
    else:
        paths = [os.path.join(root, p) for p in zarr_path_cfg]

    if not paths:
        raise FileNotFoundError(
            f"No zarr files found at '{root}/{zarr_path_cfg}'.\n"
            f"See data/README.md for download instructions."
        )

    task_datasets = [
        ZarrEpisodeDataset(
            zarr_path=p,
            obs_key=cfg.get("obs_key", "obs"),
            action_key=cfg.get("action_key", "action"),
            horizon=cfg.get("horizon", 16),
            pad_before=cfg.get("pad_before", 1),
            pad_after=cfg.get("pad_after", 7),
            task_id=tid,
        )
        for tid, p in enumerate(paths)
    ]
    dataset = MultiTaskRoboticsDataset(task_datasets)

    sampler = TaskStratifiedSampler(
        dataset,
        n_tasks_per_batch=cfg.get("n_tasks_per_batch", 4),
        n_per_task=cfg.get("n_samples_per_task", 32),
        steps=cfg.get("steps", 100_000),
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )
    action_dim = task_datasets[0].action_dim
    num_tasks  = dataset.num_tasks
    return loader, action_dim, num_tasks

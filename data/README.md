# Data Setup

All data paths are configurable via the `data.root` field in your yaml config.

---

## Image Generation — ImageNet

**Download to:** `data/imagenet/`

ImageNet requires manual download from the [official source](https://www.image-net.org/download.php) (registration required).

Expected layout:
```
data/imagenet/
├── train/
│   ├── n01440764/   ← synset folders
│   ├── n01443537/
│   └── ...          (1000 classes)
└── val/
    ├── n01440764/
    └── ...
```

**Config:**
```yaml
data:
  type: imagenet
  root: data/imagenet
  image_size: 256
  n_classes_per_batch: 8
  n_samples_per_class: 64
```

---

## Robotics — Diffusion Policy Zarr Format

**Download to:** `data/robotics/`

Data follows the [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) zarr format.

### PushT (quickstart)
```bash
mkdir -p data/robotics
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zarr.zip
unzip pusht.zarr.zip -d data/robotics/
# Result: data/robotics/pusht.zarr/
```

### Other tasks
| Task | URL |
|------|-----|
| pusht | `https://diffusion-policy.cs.columbia.edu/data/training/pusht.zarr.zip` |
| robomimic_can_ph | `https://diffusion-policy.cs.columbia.edu/data/training/robomimic_can_ph.zarr.zip` |
| robomimic_lift_ph | `https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lift_ph.zarr.zip` |
| kitchen_mixed | `https://diffusion-policy.cs.columbia.edu/data/training/kitchen_mixed.zarr.zip` |

All follow the same zarr layout:
```
<task>.zarr/
├── data/
│   ├── obs      [T_total, obs_dim]    float32
│   └── action   [T_total, action_dim] float32
└── meta/
    └── episode_ends  [N_episodes]     int64
```

**Config (single task):**
```yaml
data:
  type: robotics
  root: data/robotics
  zarr_path: pusht.zarr
  horizon: 16
  n_tasks_per_batch: 1
  n_samples_per_task: 64
```

**Config (multi-task):**
```yaml
data:
  type: robotics
  root: data/robotics
  zarr_path: "*.zarr"        # glob — all zarr stores in root
  horizon: 16
  n_tasks_per_batch: 4
  n_samples_per_task: 32
```

---

## Pretrained Weights

| Model | Path | Download |
|-------|------|---------|
| DiT-B/2 | `checkpoints/pretrained/DiT-B-2-256x256.pt` | `bash scripts/download_pretrained.sh` |
| DiT-L/2 | `checkpoints/pretrained/DiT-L-2-256x256.pt` | same |
| SD VAE  | auto via HuggingFace diffusers | auto on first run |
| DINOv2  | auto via torch.hub | auto on first run |

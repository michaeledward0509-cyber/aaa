# Training on CPU and Free Cloud (Laptop + Colab/Kaggle)

**→ For a single step-by-step build guide, see [BuildInceptionFusionStepByStep.md](BuildInceptionFusionStepByStep.md).**

You can train the Inception fusion model (ViT + YOLOv11 + EfficientNetV2) using **only a laptop CPU** and **free cloud** (Google Colab, Kaggle). The main idea: run the heavy steps (YOLO training, ViT+YOLO precompute) once on free GPU or overnight on CPU; then train the fusion on your laptop from a small cache.

**Subnet dataset augmentation** is **on by default** with **strength `light`** (mild: flip, ±10° rotation, slight brightness/contrast). Use `--augment_strength moderate` or `--augment_strength strong` for more augmentation, or `--no_augment` to disable.

---

## Strategy

| Step | Where | What |
|------|--------|------|
| 1. Data | Laptop or Colab | Download subnet data once (`download_data.py` or imagefolder). |
| 2. YOLO training | **Free Colab/Kaggle GPU** (or laptop overnight) | Train YOLOv11-cls; get `best.pt`. |
| 3. Precompute branches | **Free Colab/Kaggle GPU** (or laptop overnight) | Run ViT + YOLO on all images once; save `branch_cache.npz`. |
| 4. Fusion training | **Laptop CPU** | Train EfficientNetV2 + fusion (Keras) using `--precomputed_branches`; no ViT/YOLO loaded. |

Steps 3 and 4 are the trick: precomputing saves **RAM** and **time** on CPU because ViT and YOLO are never loaded during fusion training.

---

## Free cloud options

| Service | Free tier | Best for |
|--------|-----------|----------|
| **Google Colab** | 1× GPU (T4), ~12 GB RAM, a few hours per session | YOLO training, precompute; download `best.pt` and `branch_cache.npz` to laptop. |
| **Kaggle** | ~30 h/week GPU (P100) | Same; run notebooks or scripts. |
| **Laptop (e.g. 24 GB RAM)** | Your machine | Precompute overnight if no Colab; fusion training with larger batch (e.g. 16–24) and augmentation. |

With **24 GB RAM** you can use larger batch sizes on your laptop (e.g. `--batch_size 16` or `24` for fusion training) and run precompute with `--batch_size 16` without OOM.

---

## Step 1: Prepare data (once)

On your laptop (or in Colab):

```bash
poetry run python base_miner/datasets/download_data.py
```

Or use a local imagefolder with `train/roadwork/` and `train/no_roadwork/`. Note the path (e.g. `imagefolder:/path/to/roadwork_data`).

---

## Step 2: Train YOLO on free GPU (Colab or Kaggle)

**Option A – Google Colab**

1. **Runtime → Change runtime type → GPU (T4).**
2. Upload your repo (or clone from Git) and install deps:
   ```python
   !pip install transformers ultralytics datasets pillow torch torchvision
   ```
3. Upload your dataset (or mount Google Drive) and point to the imagefolder.
4. Train YOLO with **strong augmentation** (built into Ultralytics; use `augment=True` by default). Use **nano** for speed and to stay within Colab’s ~12 GB RAM:

```python
# In Colab (free GPU) - Python script with strong augmentation
from ultralytics import YOLO
model = YOLO('yolo11n-cls')
model.train(
    data='/path/to/roadwork_yolo_data',
    epochs=50,
    imgsz=224,
    batch=32,
    augment=True,
    # Strong augmentation parameters
    hsv_h=0.05, degrees=25, translate=0.2, shear=10,
    perspective=0.0001, mixup=0.1, mosaic=1.0
)
# Download runs/classify/train/weights/best.pt to your laptop (or save to Drive).
```

Or using CLI:
```bash
!yolo classify train data=/path/to/roadwork_yolo_data model=yolo11n-cls epochs=50 imgsz=224 batch=32 augment=True hsv_h=0.05 degrees=25 translate=0.2 shear=10 mixup=0.1
```

**Option B – Laptop CPU (slow)**

```bash
yolo classify train data=/path/to/roadwork_yolo_data model=yolo11n-cls epochs=30 imgsz=224 batch=8 augment=True hsv_h=0.05 degrees=25 translate=0.2 shear=10 mixup=0.1
```

Use small `batch` (e.g. 8) and fewer epochs if needed. You only need a usable `best.pt`.

---

## Step 3: Precompute ViT + YOLO (once)

Run ViT and YOLO on the **whole dataset** once and save a cache. Do this on **Colab/Kaggle GPU** (faster) or **laptop CPU** (overnight).

**On Colab** (after uploading repo, dataset, and `best.pt`; ~12 GB RAM – use batch_size 16 or 24 if you have more):

```bash
!cd /path/to/streetvision-subnet && python base_miner/scripts/precompute_branches.py \
  --dataset_path imagefolder:/path/to/roadwork_data \
  --yolo_weights /path/to/best.pt \
  --output_cache ./branch_cache.npz \
  --split train \
  --batch_size 16
```

**On laptop CPU** (smaller batch to avoid OOM):

```bash
poetry run python base_miner/scripts/precompute_branches.py \
  --dataset_path imagefolder:/path/to/roadwork_data \
  --yolo_weights ./best.pt \
  --output_cache ./branch_cache.npz \
  --split train \
  --batch_size 4
```

Then **download `branch_cache.npz`** to your laptop if you ran on Colab/Kaggle.

---

## Step 4: Train fusion (laptop or Colab; no ViT/YOLO if using cache)

Fusion training only needs the dataset (for images → EfficientNetV2) and the cache (for ViT/YOLO probs). ViT and YOLO are **not** loaded when using `--precomputed_branches`, so RAM use is much lower.

**Subnet dataset augmentation** is **on by default** with **strength `light`** (mild: flip, ±10° rotation, slight brightness/contrast). Use `--augment_strength moderate` or `strong` for more, or `--no_augment` to disable.

**Keras** (recommended for CPU; often faster on CPU):

```bash
poetry run python base_miner/scripts/train_inception_fusion_keras.py \
  --dataset_path imagefolder:/path/to/roadwork_data \
  --precomputed_branches ./branch_cache.npz \
  --output_dir ./inception_fusion_keras \
  --epochs 10 \
  --batch_size 16
```

**PyTorch** (augmentation: random crop, flip, rotation, color jitter):

```bash
poetry run python base_miner/scripts/train_inception_fusion.py \
  --dataset_path imagefolder:/path/to/roadwork_data \
  --precomputed_branches ./branch_cache.npz \
  --output_dir ./inception_fusion_weights \
  --epochs 10 \
  --batch_size 16
```

Use the **same** `--dataset_path` and `--split` as in Step 3 so sample order matches the cache.

- **24 GB RAM**: Use `--batch_size 16` or `24` for faster training; augmentation stays enabled by default.
- **8 GB RAM**: Use `--batch_size 8` (or 4 if OOM).
- **Epochs**: 10–15 is a good range with augmentation; you can reduce to 5 for a quick run.

---

## Subnet dataset augmentation

We apply **data augmentation to the subnet dataset** during fusion training so the model sees mild variants of each image. This is **on by default** with **strength `light`** (a little augmentation).

| Strength | Keras | PyTorch |
|----------|--------|---------|
| **light** (default) | ±10° rotation, brightness/contrast 0.9–1.1 | ±10° rotation, light color jitter |
| **moderate** | ±15° rotation, brightness/contrast 0.85–1.15 | ±15° rotation, moderate color jitter |
| **strong** | ±25° rotation, brightness 0.7–1.35, contrast 0.7–1.35, vertical flip 20%, optional blur | Larger crop, ±25° rotation, vertical flip 20%, strong color jitter |

Example (default is light augmentation; use `--augment_strength strong` for more):

```bash
poetry run python base_miner/scripts/train_inception_fusion_keras.py \
  --dataset_path imagefolder:/path/to/roadwork_data \
  --precomputed_branches ./branch_cache.npz \
  --output_dir ./inception_fusion_keras \
  --epochs 10 --batch_size 16
```

Augmentation is applied only to the **EfficientNetV2 branch** images; ViT and YOLO probs from the cache are for the original images. YOLO training (Step 2) uses **YOLOv11** (or YOLOv8 as fallback) with **strong augmentation** (e.g. `hsv_h=0.05`, `degrees=25`, `translate=0.2`, `shear=10`, `mixup=0.1`) to maximize data diversity when training on subnet-only data.

---

## Tips for low resources and Colab

| Issue | What to do |
|-------|------------|
| **24 GB RAM laptop** | Precompute with `--batch_size 16`; fusion with `--batch_size 16` or `24`. Augmentation on. |
| Colab (~12 GB RAM) | Precompute with `--batch_size 12`–16; if OOM, reduce to 8. |
| Out of memory (precompute) | Use `--batch_size 4` or 2. |
| Fusion training slow on CPU | Use Keras script; keep batch_size 8–24; fewer epochs (5–10). |
| No GPU on Colab | Precompute on laptop overnight; use batch_size 4–8. |
| Dataset too large | Use a subset for a first run, or use the same split everywhere. |

---

## Summary

1. **Data**: Get subnet data once (laptop or Colab).
2. **YOLO**: Train on **Colab** free GPU (with **strong augmentation**) → get `best.pt`.
3. **Precompute**: Run `precompute_branches.py` once on **Colab** (or laptop) → get `branch_cache.npz`.
4. **Fusion**: Train with `--precomputed_branches ./branch_cache.npz` on your laptop (e.g. 24 GB RAM: `--batch_size 16`–24; **augmentation on by default**). Keras or PyTorch.

After that, point the miner at your fusion model as in [TrainingInceptionFusion.md](TrainingInceptionFusion.md).

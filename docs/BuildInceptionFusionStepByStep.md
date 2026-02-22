# Build the Inception Fusion Model – Step by Step

This guide walks you through building the **Inception-style fusion model** (ViT + YOLO + EfficientNetV2) from scratch using **only subnet data**, with **Colab** for the heavy steps and your **laptop** for fusion training.

---

## Overview

| Step | What you do | Where |
|------|-------------|--------|
| **1** | Set up repo and get subnet data | Laptop |
| **2** | Train YOLO classifier (2 classes) | Colab (GPU) or laptop |
| **3** | Precompute ViT + YOLO probabilities | Colab (GPU) or laptop |
| **4** | Train EfficientNetV2 + fusion from cache | Laptop |
| **5** | Configure miner and (optional) evaluate | Laptop |

---

## Step 1: Set up the repo and get subnet data

**On your laptop** (in the repo root):

1. Install dependencies:
   ```bash
   cd /path/to/streetvision-subnet
   poetry install
   ```

2. Download the subnet dataset (or use your own imagefolder):
   ```bash
   poetry run python base_miner/datasets/download_data.py
   ```
   Data is stored under `~/.cache/huggingface`. If you use a **local imagefolder** instead, create this layout:
   ```
   /path/to/roadwork_data/
     train/
       no_roadwork/
         *.jpg
       roadwork/
         *.jpg
   ```
   Remember the path: you will use either the Hugging Face dataset (e.g. after converting to imagefolder) or `imagefolder:/path/to/roadwork_data`.

3. **For YOLO**, you need the same images in **imagefolder** layout. If you downloaded the HF dataset, export or symlink it so you have a folder, e.g.:
   ```
   /path/to/roadwork_yolo_data/
     train/
       no_roadwork/
       roadwork/
   ```
   Set `DATA_PATH=/path/to/roadwork_yolo_data` (or your actual path) for the next steps.

---

## Step 2: Train YOLO (2-class classifier)

YOLO must be trained **first**; you will use its weights in precompute and later in the miner.

**Option A – Google Colab (recommended, free GPU)**

1. Open [Google Colab](https://colab.research.google.com), **Runtime → Change runtime type → GPU (T4)**.
2. Install and prepare:
   ```python
   !pip install ultralytics transformers datasets pillow torch torchvision
   ```

3. **Get your dataset into Colab** (choose one):

   **Option 3a – Mount Google Drive**  
   Put your dataset on Drive in the imagefolder layout (e.g. `MyDrive/roadwork_data/train/roadwork/` and `train/no_roadwork/`). In a Colab cell:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   # Your path is then e.g. /content/drive/MyDrive/roadwork_data
   DATA_PATH = "/content/drive/MyDrive/roadwork_data"
   ```
   Use `DATA_PATH` as `data=` in the YOLO and precompute commands.

   **Option 3b – Upload a zip from your laptop**  
   On your laptop, zip the folder that contains `train/roadwork/` and `train/no_roadwork/` (e.g. `roadwork_data.zip`). In Colab: **Files** (left sidebar) → **Upload to session storage** → choose the zip. Then in a cell:
   ```python
   !unzip -q roadwork_data.zip -d /content
   # If the zip contained a folder "roadwork_data", your path is:
   DATA_PATH = "/content/roadwork_data"
   ```

   **Option 3c – Download from Hugging Face and export to imagefolder (in Colab session)**  
   Load the subnet dataset and write it to an imagefolder under `/content/` (session-only unless you copy to Drive):
   ```python
   from datasets import load_dataset
   import os
   ds = load_dataset("natix-network-org/roadwork", split="train")
   DATA_PATH = "/content/roadwork_data"
   for i, ex in enumerate(ds):
       label_id = ex.get("label", ex.get("labels", 0))
       label_name = "roadwork" if label_id == 1 else "no_roadwork"
       out_dir = os.path.join(DATA_PATH, "train", label_name)
       os.makedirs(out_dir, exist_ok=True)
       path = os.path.join(out_dir, f"{i:06d}.jpg")
       ex["image"].save(path)
   # DATA_PATH is ready to use
   ```

   **Option 3d – Put the dataset on Google Drive (recommended if you use Colab)**  
   So the dataset lives on Drive and you don’t re-download every time:
   1. Mount Drive in Colab (see Option 3a).
   2. Download the subnet dataset from Hugging Face and save it as an imagefolder **on Drive**:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")

   from datasets import load_dataset
   import os

   ds = load_dataset("natix-network-org/roadwork", split="train")
   # Save to a folder on your Drive (e.g. My Drive/roadwork_data)
   DATA_PATH = "/content/drive/MyDrive/roadwork_data"
   for i, ex in enumerate(ds):
       label_id = ex.get("label", ex.get("labels", 0))
       label_name = "roadwork" if label_id == 1 else "no_roadwork"
       out_dir = os.path.join(DATA_PATH, "train", label_name)
       os.makedirs(out_dir, exist_ok=True)
       path = os.path.join(out_dir, f"{i:06d}.jpg")
       ex["image"].save(path)
   print("Dataset saved to Drive at:", DATA_PATH)
   ```
   After this runs once, your dataset is in **Google Drive** under `My Drive/roadwork_data`. In future Colab sessions, mount Drive and set `DATA_PATH = "/content/drive/MyDrive/roadwork_data"` (no need to re-download).

   **Why is the Hugging Face cache so big (e.g. ~32GB) but my export only ~4GB?**  
   Hugging Face uses **two** caches: (1) **Hub cache** (`~/.cache/huggingface/hub`) — raw downloaded files (for this dataset, the Parquet is ~10.5 GB); (2) **Datasets cache** (`~/.cache/huggingface/datasets`) — the same data converted to **Arrow** for fast loading. Arrow often stores image columns in a form that uses more space (decoded or less compressed), and there can be extra files (indices, fingerprints). So total cache ≈ 10.5 + 10–20+ GB → ~32 GB is normal.  
   Your **export** is smaller because: you write a **single copy** of the images as **JPEG** (one split, e.g. train). About 8.5k images × ~500 KB each ≈ **4 GB**. So the 4 GB folder is the “real” size of the dataset as plain JPEGs; the 32 GB is HF’s internal storage (downloaded Parquet + Arrow cache). After exporting, you can use the 4 GB imagefolder for training and clear the HF cache to free the 32 GB if you want.

4. **If you only have `train/`** (no `val/` or `test/`), YOLO will crash because it builds loaders for val and test and gets `None`. Create val and test by splitting train (run once before training):
   ```python
   # In Colab, after exporting to DATA_PATH (e.g. Drive):
   DATA_PATH = "/content/drive/MyDrive/roadwork_data"
   import os, random, shutil
   train_dir = os.path.join(DATA_PATH, "train")
   for c in os.listdir(train_dir):
       class_dir = os.path.join(train_dir, c)
       if not os.path.isdir(class_dir): continue
       names = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
       random.seed(42); random.shuffle(names)
       n_val = max(1, int(len(names)*0.1)); n_test = max(1, int(len(names)*0.1))
       for split, count in [("val", n_val), ("test", n_test)]:
           os.makedirs(os.path.join(DATA_PATH, split, c), exist_ok=True)
           for f in names[:count]:
               shutil.move(os.path.join(class_dir, f), os.path.join(DATA_PATH, split, c, f))
           names = names[count:]
   print("val/ and test/ created.")
   ```
   Or from the repo: `poetry run python workspace/split_for_yolo.py /path/to/roadwork_data`

5. Train YOLO (classification, not detection) with **strong augmentation**:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolo11n-cls')
   model.train(
       data='/content/roadwork_data',  # Replace with your DATA_PATH
       model='yolo11n-cls',
       epochs=50,
       imgsz=224,
       batch=32,
       augment=True,
       # Strong augmentation parameters
       hsv_h=0.05, degrees=25, translate=0.2, shear=10,
       perspective=0.0001, mixup=0.1, mosaic=1.0
   )
   ```
   Or using CLI:
   ```bash
   !yolo classify train data=/content/roadwork_data model=yolo11n-cls epochs=50 imgsz=224 batch=32 augment=True hsv_h=0.05 degrees=25 translate=0.2 shear=10 mixup=0.1
   ```
   Replace `/content/roadwork_data` with your `DATA_PATH` (e.g. `/content/drive/MyDrive/roadwork_data`). After training, the best weights are in:
   `runs/classify/train/weights/best.pt`.

6. **Download `best.pt`** to your laptop (e.g. into `~/streetvision-subnet/best.pt` or a folder you prefer).

**Option B – Laptop (CPU, slower)**

```bash
cd /path/to/streetvision-subnet
yolo classify train data=/path/to/roadwork_yolo_data model=yolo11n-cls epochs=30 imgsz=224 batch=8 augment=True hsv_h=0.05 degrees=25 translate=0.2 shear=10 mixup=0.1
```

Or use the Python script `workspace/train_yolo.py` which includes strong augmentation parameters:
```bash
poetry run python workspace/train_yolo.py
```

Use the path to `runs/classify/train/weights/best.pt` as your YOLO weights (e.g. `BEST_PT=/path/to/best.pt`).

---

## Step 3: Precompute ViT + YOLO (one-time)

This runs the default ViT and your YOLO on **every training image** and saves their outputs to a cache. Fusion training will use this cache so it does **not** load ViT or YOLO (saving RAM and time on the laptop).

**Option A – Colab**

1. Upload the repo and `best.pt` to Colab (or clone repo and upload `best.pt`).
2. Install deps if not already:
   ```python
   !pip install transformers ultralytics datasets pillow torch torchvision
   ```
3. Run precompute (use the **same** dataset path and split you will use in Step 4):
   ```python
   !cd /content/streetvision-subnet && python base_miner/scripts/precompute_branches.py \
     --dataset_path imagefolder:/content/roadwork_data \
     --yolo_weights /content/best.pt \
     --output_cache ./branch_cache.npz \
     --split train \
     --batch_size 16
   ```
   Replace paths with your actual Colab paths.

4. **Download `branch_cache.npz`** to your laptop (e.g. into the repo root or a folder you will use in Step 4).

**Option B – Laptop**

From the repo root:

```bash
poetry run python base_miner/scripts/precompute_branches.py \
  --dataset_path imagefolder:/path/to/roadwork_data \
  --yolo_weights /path/to/best.pt \
  --output_cache ./branch_cache.npz \
  --split train \
  --batch_size 8
```

Use the **exact same** `--dataset_path` and `--split` in Step 4. If you get OOM, use `--batch_size 4`.

---

## Step 4: Train EfficientNetV2 + fusion (on the laptop)

Fusion training loads only the **dataset** (for images) and the **cache** (for ViT/YOLO probs). No ViT or YOLO models are loaded, so this step is light on RAM and runs on CPU. The EfficientNetV2 branch replaces the previous GoogLeNet/InceptionV3 architecture for better performance.

**Important:** Use the **same** `--dataset_path` and `--split` as in Step 3.

**Option A – Keras (recommended on CPU)**

```bash
cd /path/to/streetvision-subnet

poetry run python base_miner/scripts/train_inception_fusion_keras.py \
  --dataset_path imagefolder:/path/to/roadwork_data \
  --precomputed_branches ./branch_cache.npz \
  --output_dir ./inception_fusion_keras \
  --epochs 10 \
  --batch_size 16 \
  --efficientnet_size b0
```

- With **24 GB RAM** you can try `--batch_size 24`. With **8 GB** use `--batch_size 8`.
- Output: directory `./inception_fusion_keras/` with `efficientnetv2_branch.keras` and `fusion_head.keras`.
- **EfficientNetV2 model size**: Choose from `b0` (smallest/fastest, default), `b1`, `b2`, `b3`, `s`, `m`, or `l` (largest/most accurate). Use `--efficientnet_size b0` for CPU training or `--efficientnet_size s` for better accuracy on GPU.
- **Early stopping** is on by default: 15% validation split, stop after 3 epochs without val loss improvement. Use `--validation_split 0.15 --early_stopping_patience 3` (defaults) or `--validation_split 0` to disable.
- Augmentation is **light** by default; use `--augment_strength moderate` or `strong` for more, or `--no_augment` to disable.

**Option B – PyTorch**

```bash
poetry run python base_miner/scripts/train_inception_fusion.py \
  --dataset_path imagefolder:/path/to/roadwork_data \
  --precomputed_branches ./branch_cache.npz \
  --output_dir ./inception_fusion_weights \
  --epochs 10 \
  --batch_size 16
```

- Output: `./inception_fusion_weights/googlenet_fusion.pt`.

---

## Step 5: Configure the miner and run

1. **Edit the detector config**  
   Copy and edit the fusion config:
   ```bash
   cp base_miner/detectors/configs/inception_fusion_roadwork.yaml base_miner/detectors/configs/my_inception_fusion.yaml
   ```
   Edit `my_inception_fusion.yaml` and set **absolute paths**:

   **If you used Keras (Step 4A):**
   ```yaml
   vit_config: ViT_roadwork.yaml
   yolo_weights_path: /absolute/path/to/best.pt
   googlenet_fusion_path: /absolute/path/to/streetvision-subnet/inception_fusion_keras
   use_keras: true
   roadwork_class_index: 1
   ```

   **If you used PyTorch (Step 4B):**
   ```yaml
   vit_config: ViT_roadwork.yaml
   yolo_weights_path: /absolute/path/to/best.pt
   googlenet_fusion_path: /absolute/path/to/streetvision-subnet/inception_fusion_weights/googlenet_fusion.pt
   use_keras: false
   roadwork_class_index: 1
   ```

2. **Point the miner at your model**  
   In your miner env (e.g. `miner.env` or `.env`):
   ```bash
   IMAGE_DETECTOR=InceptionFusion
   IMAGE_DETECTOR_CONFIG=my_inception_fusion.yaml
   IMAGE_DETECTOR_DEVICE=cuda
   ```
   Use `cpu` if you have no GPU.

3. **Run the miner**  
   Start the miner as usual (see [Mining.md](Mining.md)). It will load ViT, YOLO, and your EfficientNetV2 + fusion and use them for inference.

---

## Optional: Evaluate before mining

2. Commands (from repo root)
Your Inception Fusion vs default ViT (recommended):
poetry run python base_miner/scripts/evaluate_detector.py \
  --dataset_path "imagefolder:./roadwork_data" \
  --split test \
  --my_detector InceptionFusion \
  --my_config my_inception_fusion.yaml \
  --device cpu

Default model: ViT (ViT_roadwork.yaml).
Your model: Inception Fusion with base_miner/detectors/configs/my_inception_fusion.yaml (same as miner).

Default only (no “yours”):
poetry run python base_miner/scripts/evaluate_detector.py \  --dataset_path "imagefolder:./roadwork_data" \  --split val \  --device cuda

Your model only (no default):
poetry run python base_miner/scripts/evaluate_detector.py \  --dataset_path "imagefolder:./roadwork_data" \  --split val \  --no_default \  --my_detector InceptionFusion \  --my_config my_inception_fusion.yaml \  --device cuda

Quick run (e.g. 200 samples):
poetry run python base_miner/scripts/evaluate_detector.py \  --dataset_path "imagefolder:./roadwork_data" \  --split val \  --max_samples 200 \  --my_detector InceptionFusion \  --my_config my_inception_fusion.yaml \  --device cuda

If you don’t have a GPU, use --device cpu.
---

## Troubleshooting

| Problem | What to do |
|---------|------------|
| Precompute: “YOLO weights not found” | Use an absolute path to `best.pt` and ensure the file exists. |
| Precompute: OOM | Use `--batch_size 4` (or 2). |
| Fusion: “Precomputed cache has X samples but dataset has Y” | Use the **same** `--dataset_path` and `--split` in Step 3 and Step 4. |
| Fusion: OOM on laptop | Lower `--batch_size` (e.g. 8 or 4). |
| Miner: "googlenet_fusion_path not found" | Use **absolute** paths in the YAML and set `use_keras: true` only if you trained with Keras. The config key `googlenet_fusion_path` is kept for backward compatibility but should point to the directory containing `efficientnetv2_branch.keras` (or `googlenet_branch.keras` for older models). |

For **ideas to improve** the model (deeper fusion head, more epochs, fine-tuning, TTA, etc.), see [ImprovingTheModel.md](ImprovingTheModel.md).

For more detail on data, Colab, and augmentation, see [TrainingOnCPUAndFreeCloud.md](TrainingOnCPUAndFreeCloud.md) and [TrainingInceptionFusion.md](TrainingInceptionFusion.md).

#Latency
Offline latency script (recommended)
Run the new script so it uses the same detector and config as your miner:
cd /root/Workspace/aaapoetry run python scripts/measure_detector_latency.py
It loads your detector from miner.env (InceptionFusion + my_inception_fusion.yaml), runs warmup then many inference steps, and prints:
Mean, std, median (p50), p95, p99 in milliseconds
A note if p99 is above the validator timeout (9 s)
Options:
# More warmup and runs for stable statspoetry run python scripts/measure_detector_latency.py --warmup 5 --runs 50# GPUpoetry run python scripts/measure_detector_latency.py --device cuda# Use a real image instead of a dummy onepoetry run python scripts/measure_detector_latency.py --image /path/to/image.jpg
This measures inference-only latency (decode + detector forward).


# Evaluation Results
yours:
  accuracy   = 0.9409
  mcc        = 0.8135
  reward_raw = 0.8772  (0.5*mcc + 0.5*acc)
  reward     = 0.6750  (reward_raw^3.0)
  n_valid    = 626/626

--- Comparison (yours vs default) ---
  accuracy: 0.9409 vs 0.9313  (better)
  mcc:      0.8135 vs 0.7832  (better)
  reward:   0.6750 vs 0.6300  (better)

# Training the Inception-Style Fusion Model (ViT + YOLOv11 + EfficientNetV2 / GoogLeNet)

**→ For a single end-to-end walkthrough, see [BuildInceptionFusionStepByStep.md](BuildInceptionFusionStepByStep.md).**

You can build a **combined model** that fuses three architectures using an **Inception-style** idea: **parallel branches** (default ViT, YOLOv11, and a CNN branch) whose outputs are concatenated and passed through a **learned fusion head** to produce the final roadwork probability. Training uses **only subnet data** (e.g. `natix-network-org/roadwork` or your imagefolder).

---

## Is it possible?

Yes. The design is:

1. **Default model (ViT)** – Pre-trained subnet model (`natix-network-org/roadwork`), used as-is (frozen).
2. **YOLO** – **YOLOv11** in classification mode (2 classes: roadwork / no roadwork), trained separately on subnet data (see [TrainingYOLO.md](TrainingYOLO.md)); frozen in fusion training. (YOLOv8 is supported as a fallback.)
3. **CNN branch** – **EfficientNetV2** (Keras, recommended) or **GoogLeNet** (PyTorch legacy): 2-class head trained on subnet data.
4. **Inception-style fusion** – A small layer that takes the three branch outputs (P(roadwork) from each) and learns weights to produce the final P(roadwork). Only this fusion head and the CNN head are trained; ViT and YOLO stay frozen.

All training (YOLO, GoogLeNet head, and fusion) uses **only the subnet’s training data** (no external datasets). **Data augmentation** is applied by default during fusion training with **light** strength (a little: flip, ±10° rotation, slight brightness/contrast). Use `--augment_strength moderate` or `strong` for more, or `--no_augment` to disable.

---

## Prerequisites

| Component   | What you need |
|------------|----------------|
| **Data**   | Subnet data only: run `poetry run python base_miner/datasets/download_data.py` so that `natix-network-org/roadwork` is available, or use an imagefolder with `train/roadwork/` and `train/no_roadwork/`. |
| **ViT**    | Default model from config (`ViT_roadwork.yaml`). No extra training. |
| **YOLO**   | A YOLOv11-cls model trained for 2 classes on the same subnet data. See [TrainingYOLO.md](TrainingYOLO.md). You need the path to `best.pt`. |
| **CNN branch + fusion** | EfficientNetV2 (Keras) or GoogLeNet (PyTorch), trained by the script below on subnet data. |

---

## Step 1: Prepare data (subnet only)

Use the same layout as in [TrainingYOLO.md](TrainingYOLO.md) and [Training.md](Training.md):

```bash
poetry run python base_miner/datasets/download_data.py
```

If you use a local imagefolder:

```
/path/to/roadwork_data/
  train/
    no_roadwork/
      *.jpg
    roadwork/
      *.jpg
```

---

## Step 2: Train YOLO (subnet data only)

Train a YOLOv11 **classifier** (not detection) on the same data:

```bash
yolo classify train \
  data=/path/to/roadwork_yolo_data \
  model=yolo11n-cls \
  epochs=80 \
  imgsz=224 \
  batch=32
```

Set `yolo_weights_path` in the fusion config to your `weights/best.pt`.

---

## Step 3: Train CNN head + fusion (subnet data only)

This script trains only the **CNN 2-class head** (EfficientNetV2 in Keras or GoogLeNet in PyTorch) and the **fusion layer**; ViT and YOLO are frozen.

```bash
poetry run python base_miner/scripts/train_inception_fusion.py \
  --dataset_path imagefolder:/path/to/roadwork_data \
  --yolo_weights /path/to/roadwork_yolo/weights/best.pt \
  --output_dir ./inception_fusion_weights \
  --epochs 10 \
  --batch_size 16
```

- Use the **same dataset** as for YOLO (subnet data only).
- `--dataset_path` can be `imagefolder:/path/to/...` or a Hugging Face dataset path (e.g. the roadwork dataset).
- Output: `./inception_fusion_weights/googlenet_fusion.pt` (GoogLeNet head + fusion head state dicts; PyTorch legacy path).

### Recommended: Train with Keras (EfficientNetV2)

Use **Keras** (tf.keras) for the **EfficientNetV2** branch and the fusion head (replaces the older InceptionV3/GoogLeNet setup):

```bash
poetry run python base_miner/scripts/train_inception_fusion_keras.py \
  --dataset_path imagefolder:/path/to/roadwork_data \
  --yolo_weights /path/to/roadwork_yolo/weights/best.pt \
  --output_dir ./inception_fusion_keras \
  --epochs 10 \
  --batch_size 16
```

- ViT and YOLO are still run in PyTorch (frozen); only the EfficientNetV2 top and fusion layer are built and trained in Keras.
- Output: directory `./inception_fusion_keras/` with `efficientnetv2_branch.keras` and `fusion_head.keras`.
- In config, set `googlenet_fusion_path` to this directory and `use_keras: true` (config key kept for backward compatibility).

---

## Step 4: Configure and run the miner

Edit `base_miner/detectors/configs/inception_fusion_roadwork.yaml`:

**PyTorch** (after `train_inception_fusion.py`):

```yaml
vit_config: ViT_roadwork.yaml
yolo_weights_path: /absolute/path/to/your/yolo/weights/best.pt
googlenet_fusion_path: /absolute/path/to/inception_fusion_weights/googlenet_fusion.pt
use_keras: false
roadwork_class_index: 1
```

**Keras** (after `train_inception_fusion_keras.py`):

```yaml
vit_config: ViT_roadwork.yaml
yolo_weights_path: /absolute/path/to/your/yolo/weights/best.pt
googlenet_fusion_path: /absolute/path/to/inception_fusion_keras
use_keras: true
roadwork_class_index: 1
```

In `miner.env`:

```bash
IMAGE_DETECTOR=InceptionFusion
IMAGE_DETECTOR_CONFIG=inception_fusion_roadwork.yaml
IMAGE_DETECTOR_DEVICE=cuda
```

---

## Step 5: Evaluate

Compare the fusion model to the default ViT on a held-out split:

```bash
poetry run python base_miner/scripts/evaluate_detector.py \
  --dataset_path imagefolder:/path/to/roadwork_data \
  --split validation \
  --my_detector InceptionFusion \
  --my_config inception_fusion_roadwork.yaml
```

---

## Summary

| Question | Answer |
|----------|--------|
| Can you combine default model, YOLO, and a CNN with an Inception-style module? | **Yes.** Three parallel branches (ViT, YOLOv11, EfficientNetV2 or GoogLeNet) → concatenate their P(roadwork) → learned fusion head → final score. |
| Do you use only subnet training data? | **Yes.** YOLO is trained on subnet data; CNN head and fusion are trained on subnet data; ViT is the existing subnet model. |
| What is trained? | YOLOv11 (or YOLOv8) via `workspace/train_yolo.py` or YOLO CLI; then EfficientNetV2 + fusion via `train_inception_fusion_keras.py` (Keras, recommended) or GoogLeNet + fusion via `train_inception_fusion.py` (PyTorch). ViT and YOLO are frozen during fusion training. |

For more on data sources and evaluation, see [DataSources.md](DataSources.md) and [Evaluation.md](Evaluation.md).

**Training on a laptop (CPU only) or free cloud (Colab/Kaggle):** See [TrainingOnCPUAndFreeCloud.md](TrainingOnCPUAndFreeCloud.md) for precomputing ViT+YOLO once and training fusion from cache on CPU.

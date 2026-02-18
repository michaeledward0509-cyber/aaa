# Winning Strategy: How to Beat Other Miners

**Related docs:** [ImprovingTheModel.md](../docs/ImprovingTheModel.md) · [BuildInceptionFusionStepByStep.md](../docs/BuildInceptionFusionStepByStep.md) · [Evaluation.md](../docs/Evaluation.md)

**Current model stack:** Fusion uses **YOLOv11** (classification; YOLOv8 supported as fallback) and **EfficientNetV2** (Keras CNN branch; PyTorch path still uses GoogLeNet for backward compatibility). Config key `googlenet_fusion_path` points to the Keras output dir (e.g. `inception_fusion_keras/` with `efficientnetv2_branch.keras`).

---

## Understanding the Reward Formula

**Reward = (0.5 × MCC_100 + 0.5 × accuracy_10)³**

- **MCC_100**: Matthews Correlation Coefficient over the last 100 predictions (rewards calibration and balance).
- **accuracy_10**: Accuracy over the last 10 predictions (rewards recent correctness).
- **Exponent 3.0**: Small improvements are amplified (e.g. +3% raw score → ~+10% reward).

**Example:**
- Baseline: MCC=0.70, Acc=0.85 → Reward = (0.775)³ ≈ **0.465**
- Improved: MCC=0.77, Acc=0.89 → Reward = (0.83)³ ≈ **0.572** (~+23% reward)

---

## Prerequisites

- Repo set up: `poetry install`
- Dataset in imagefolder layout: `train/roadwork/`, `train/no_roadwork/`, and (for YOLO) `val/`, `test/` (use `workspace/split_for_yolo.py` if you only have `train/`)
- YOLO trained and `best.pt` available (Step 1 below)
- Precomputed branch cache: `branch_cache.npz` (Step 2 in [BuildInceptionFusionStepByStep.md](../docs/BuildInceptionFusionStepByStep.md))

**Run all commands from repo root:** `cd /path/to/streetvision-subnet`

---

## Phase 1: Quick Wins (Do These First)

### 1. Use Fusion Model with MLP Head

Fusion (ViT + YOLOv11 + EfficientNetV2) with an MLP fusion head usually beats a single model.

```bash
poetry run python base_miner/scripts/train_inception_fusion_keras.py \
  --dataset_path imagefolder:./roadwork_data \
  --precomputed_branches ./branch_cache.npz \
  --output_dir ./inception_fusion_keras \
  --fusion_mlp \
  --efficientnet_size b0 \
  --epochs 20 \
  --lr 5e-4 \
  --batch_size 16 \
  --augment_strength strong \
  --validation_split 0.15 \
  --early_stopping_patience 5
```

- **EfficientNetV2 size:** `b0` (default, fastest), `b1`, `b2`, `s`, `m`, `l` for more accuracy and compute.
- **CPU / low RAM:** use `--batch_size 8` (or 4).
- **High RAM:** try `--batch_size 24`.

### 2. Train a Stronger YOLO Branch

Use a larger YOLO (e.g. small instead of nano) and strong augmentation. Edit `workspace/train_yolo.py`:

```python
# In workspace/train_yolo.py
from ultralytics import YOLO
from pathlib import Path

repo_root = Path(__file__).parent.parent
data_path = repo_root / 'roadwork_data'

# Use small or medium for better accuracy (nano = fastest, least accurate)
model = YOLO('yolo11s-cls')   # or 'yolo11m-cls' if you have GPU headroom
# model = YOLO('yolo11n-cls')  # nano (default)

model.train(
    data=str(data_path),
    epochs=80,
    imgsz=224,
    batch=32,   # reduce to 16 or 8 if OOM
    augment=True,
    hsv_h=0.05,
    degrees=25,
    translate=0.2,
    shear=10,
    mixup=0.1,
    perspective=0.0001,
)
```

Then **re-run precompute** with the new `best.pt` and **re-train fusion** (same commands as in [BuildInceptionFusionStepByStep.md](../docs/BuildInceptionFusionStepByStep.md)).

**Fallback if YOLOv11 fails:** `model = YOLO('yolov8n-cls')`. The stack uses **YOLOv11** (recommended) and **EfficientNetV2** (Keras CNN branch); the config key `googlenet_fusion_path` is kept for backward compatibility and points to the directory with `efficientnetv2_branch.keras`.

### 3. Combine Multiple Data Sources

More diverse data often improves generalization.

```bash
# Download 1/10 of LouisChen15/ConstructionSite
poetry run python workspace/download_louis_subset.py --fraction 0.1 --output_dir ./louis_subset

# Merge into one imagefolder (e.g. copy louis_subset/train/* into roadwork_data/train/)
# Then run split_for_yolo.py if YOLO needs val/test:
poetry run python workspace/split_for_yolo.py ./roadwork_data
```

Train YOLO and fusion on the combined dataset (same scripts, point `--dataset_path` at the merged folder).

---

## Phase 2: Optimization

### 4. Hyperparameter Tuning

- **Learning rate:** try `--lr 3e-4`, `5e-4`, or `1e-3`.
- **Epochs:** use early stopping; 15–25 epochs is often enough.
- **Batch size:** 16–24 if RAM allows; 8 or 4 on CPU/low RAM.

### 5. Evaluate Before Deploying

Use the **same metrics** the subnet uses (accuracy, MCC, reward):

```bash
poetry run python base_miner/scripts/evaluate_detector.py \
  --dataset_path imagefolder:./roadwork_data \
  --split val \
  --my_detector InceptionFusion \
  --my_config inception_fusion_roadwork.yaml
```

**Targets (on a held-out val set):**
- Accuracy **> 0.90**
- MCC **> 0.80**
- Reward **> 0.55**

### 6. Configure Miner After Training

1. **Fusion config** – e.g. `base_miner/detectors/configs/inception_fusion_roadwork.yaml`:
   - `yolo_weights_path`: path to your `best.pt`
   - `googlenet_fusion_path`: path to `./inception_fusion_keras` (Keras, contains `efficientnetv2_branch.keras`) or `.../googlenet_fusion.pt` (PyTorch legacy)
   - `use_keras: true` if you used the Keras script

2. **Miner env** (e.g. `miner.env`):
   - `IMAGE_DETECTOR=InceptionFusion`
   - `IMAGE_DETECTOR_CONFIG=inception_fusion_roadwork.yaml`
   - `IMAGE_DETECTOR_DEVICE=cuda` (or `cpu`)

See [BuildInceptionFusionStepByStep.md](../docs/BuildInceptionFusionStepByStep.md) and [Mining.md](../docs/Mining.md) for full miner setup.

---

## Phase 3: Advanced Techniques

### 7. Test-Time Augmentation (TTA)

In the detector (e.g. `inception_fusion_detector.py`): run inference on original + flipped (and optionally slightly rotated) images, then **average** the predicted probabilities. Often +1–3% accuracy at 2–4× inference cost.

### 8. Ensemble Multiple Models

Train several fusion models (different seeds, LRs, or data splits) and average their probability outputs. Can add +2–4% accuracy with higher compute and latency.

### 9. Fine-Tune EfficientNetV2 Backbone

Unfreeze the last blocks of the EfficientNetV2 backbone in the Keras script (`build_efficientnetv2_keras`); use a small LR for the backbone (e.g. 1e-5) and normal LR for the head. See [ImprovingTheModel.md](../docs/ImprovingTheModel.md). Monitor validation loss to avoid overfitting.

---

## Key Principles

1. **Maximize both MCC and accuracy** – the reward uses them equally.
2. **Small gains matter** – the cube in the reward formula amplifies improvements.
3. **Consistency** – MCC rewards stable, well-calibrated predictions.
4. **Always evaluate locally** – use `evaluate_detector.py` on a val set before mainnet.
5. **Avoid overfitting** – use validation split and early stopping; tune augmentation strength.

---

## Expected Results (Rough Guide)

| Strategy              | Accuracy gain | MCC gain | Reward gain (approx) |
|-----------------------|---------------|----------|------------------------|
| Fusion + MLP          | +2–3%         | +2–3%    | +15–20%                |
| Larger YOLO (s/m)     | +1–2%         | +1–2%    | +8–12%                 |
| Strong augmentation   | +1–2%         | +1–2%    | +8–12%                 |
| More epochs + LR tune | +1–2%         | +1–2%    | +8–12%                 |
| TTA                   | +1–3%         | +1–3%    | +8–18%                 |
| Ensemble              | +2–4%         | +2–4%    | +18–30%                |

---

## Recommended Order

1. **Fusion + MLP** (easiest, high impact)
2. **Strong augmentation + more epochs + LR tuning** (easy, medium impact)
3. **Larger YOLO + re-precompute + re-train fusion** (medium effort, high impact)
4. **Combine datasets** (medium effort, good for generalization)
5. **TTA** (medium effort, high impact if latency is acceptable)
6. **Ensemble** (high effort, highest impact)

Start with 1–3, then add 4–5 as needed. Use `evaluate_detector.py` after each change to confirm gains on your val set.

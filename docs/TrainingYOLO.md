# Training a YOLO Model with Subnet-Only Data

If you **cannot add external data** and must use only the data provided by the subnet (e.g. **natix-network-org/roadwork**), you can still try to beat the default model by:

1. **Using a different architecture** – e.g. YOLO (classification mode) so the model learns a different decision boundary.
2. **Strong data augmentation** – so the same images effectively become more training signal.
3. **Training longer / better hyperparameters** – more epochs, learning rate schedule, etc.
4. **(Optional) Knowledge distillation** – train YOLO to mimic the default ViT’s probabilities on the same images; sometimes the student generalizes better.

The subnet task is **image-level binary classification** (roadwork vs no roadwork), not object detection. So use **YOLO in classification mode** (YOLOv11-cls), not detection.

---

## 1. Prepare data (subnet dataset only)

Download the subnet’s roadwork dataset and convert to imagefolder so YOLO can use it:

```bash
poetry run python base_miner/datasets/download_data.py
```

The dataset is cached under `~/.cache/huggingface`. You need a folder layout:

```
/path/to/roadwork_yolo_data/
  train/
    no_roadwork/
      *.jpg
    roadwork/
      *.jpg
  val/
    no_roadwork/
    roadwork/
```

If the natix dataset has a different structure (e.g. one split with a `label` column), write a short script to export images into `train/roadwork/` and `train/no_roadwork/` (and optionally `val/...`) so that the class names are exactly `no_roadwork` and `roadwork`. YOLO will assign label index 0 to the first class and 1 to the second (alphabetical); the miner expects **index 1 = roadwork**.

---

## 2. Train YOLO classifier (with strong augmentation)

**Option A – Python script (recommended for strong augmentation)**

Use the provided training script with strong augmentation parameters:

```python
from ultralytics import YOLO

model = YOLO('yolo11n-cls')
model.train(
    data='/path/to/roadwork_yolo_data',
    epochs=80,
    imgsz=224,
    batch=32,
    augment=True,  # Enable augmentation
    # Strong augmentation parameters
    hsv_h=0.05,      # Hue shift (default: 0.015, strong: 0.05)
    hsv_s=0.7,       # Saturation (default: 0.7, strong: 0.7)
    hsv_v=0.4,       # Brightness (default: 0.4, strong: 0.4)
    degrees=25,       # Rotation degrees (default: 0, strong: 25)
    translate=0.2,   # Translation (default: 0.1, strong: 0.2)
    scale=0.5,       # Scaling (default: 0.5, strong: 0.5)
    shear=10,        # Shearing (default: 0, strong: 10)
    perspective=0.0001,  # Perspective transform (default: 0, strong: 0.0001)
    flipud=0.0,      # Vertical flip probability (default: 0, strong: 0.0)
    fliplr=0.5,      # Horizontal flip probability (default: 0.5, strong: 0.5)
    mosaic=1.0,       # Mosaic augmentation probability (default: 1.0, strong: 1.0)
    mixup=0.1,       # Mixup augmentation probability (default: 0.0, strong: 0.1)
)
```

**Option B – Ultralytics CLI**

```bash
# Example: YOLOv11n-cls (nano), 2 classes, strong augmentation
yolo classify train \
  data=/path/to/roadwork_yolo_data \
  model=yolo11n-cls \
  epochs=80 \
  imgsz=224 \
  batch=32 \
  augment=True \
  hsv_h=0.05 \
  degrees=25 \
  translate=0.2 \
  shear=10 \
  mixup=0.1
```

To push accuracy with **subnet-only data**, use **strong augmentation** so the model sees many variants of each image. The strong augmentation parameters above include:
- **Color augmentation**: Increased hue shift (hsv_h), saturation, and brightness variations
- **Spatial augmentation**: Rotation (±25°), translation (20%), shearing (10°), perspective transforms
- **Advanced augmentation**: Mosaic (combines 4 images) and Mixup (blends images) for more diversity

Training longer (e.g. 50–100 epochs) and tuning learning rate often help when data is limited.

**Larger models** (e.g. `yolo11s-cls`, `yolo11m-cls`, `yolo11l-cls`) may generalize better but need more compute; try nano first, then scale up if needed. Available sizes: n (nano), s (small), m (medium), l (large), x (xlarge).

---

## 3. Point the miner at your YOLO weights

After training, you get a `weights/best.pt` (or `runs/classify/train/weights/best.pt`). Copy it to a stable path and set the detector config:

**File:** `base_miner/detectors/configs/yolo_roadwork.yaml`

```yaml
weights_path: /absolute/path/to/your/best.pt
roadwork_class_index: 1
```

In `miner.env`:

```bash
IMAGE_DETECTOR=YOLOClassifier
IMAGE_DETECTOR_CONFIG=yolo_roadwork.yaml
IMAGE_DETECTOR_DEVICE=cuda
```

---

## 4. Evaluate YOLO vs default

Use the evaluation script to compare your YOLO model to the default ViT on a held-out split (or the same subnet data split you didn’t train on):

```bash
poetry run python base_miner/scripts/evaluate_detector.py \
  --dataset_path imagefolder:/path/to/roadwork_yolo_data \
  --split val \
  --my_detector YOLOClassifier \
  --my_config yolo_roadwork.yaml
```

If you don’t have a separate val split, use a fixed part of the dataset for validation and don’t train on it.

---

## 5. Getting “more” from the same data (no extra images)

| Method | Description |
|--------|-------------|
| **Strong augmentation** | Random crop, flip, color jitter, blur, etc. So the model sees many variants per image. |
| **More epochs** | With heavy augmentation, training longer can help; watch for overfitting on val. |
| **Knowledge distillation** | Train YOLO to match the default ViT’s **probability** (not just the label) on each image. Sometimes the student (YOLO) generalizes better. |
| **Different architecture** | YOLO vs ViT can yield different decision boundaries; one may suit validator distribution better. |

---

## Summary

- Use **YOLO in classification mode** (e.g. `yolo11n-cls`), not detection. Note: Use `yolo11n-cls` format (not `yolov11n-cls`).
- Use **only subnet-provided data** (e.g. natix roadwork), but **augment heavily** and train with enough epochs.
- Register the detector as **YOLOClassifier** and set **weights_path** in `yolo_roadwork.yaml`.
- Evaluate with **evaluate_detector.py**; if your YOLO beats the default on a fair val set, it has a chance to perform better on the subnet.

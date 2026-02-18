# Step 1: Train YOLOv11n-cls - Building Your Own Model

This is the **first step** in building your own roadwork detection model. YOLOv11n-cls is a lightweight classification model that will serve as one component of your final fusion model.

---

## Prerequisites

1. **Python environment** with dependencies installed:
   ```bash
   cd /path/to/streetvision-subnet
   poetry install
   ```

2. **Dataset** in imagefolder format (see data preparation below)

3. **GPU (recommended)** or CPU (slower)

---

## Step 1.1: Prepare Your Dataset

YOLO requires data in **imagefolder** format with this structure:

```
/path/to/roadwork_data/
  train/
    no_roadwork/     # Class 0: images without roadwork
      img1.jpg
      img2.jpg
      ...
    roadwork/        # Class 1: images with roadwork
      img1.jpg
      img2.jpg
      ...
  val/               # Optional: validation split
    no_roadwork/
      *.jpg
    roadwork/
      *.jpg
```

### Option A: Use Subnet Dataset

Download the subnet's roadwork dataset:

```bash
poetry run python base_miner/datasets/download_data.py
```

Then convert it to imagefolder format. You can use the provided script:

```bash
poetry run python workspace/download_louis_subset.py \
  --output_dir ./roadwork_data \
  --fraction 1.0 \
  --split train
```

This creates `./roadwork_data/train/roadwork/` and `./roadwork_data/train/no_roadwork/`.

### Option B: Use Your Own Dataset

If you have your own images, organize them into the folder structure above. Make sure:
- Class folders are named exactly `no_roadwork` and `roadwork` (lowercase, underscore)
- Images are in common formats (`.jpg`, `.png`, etc.)
- You have both classes represented

### Step 1.2: Split Data (if needed)

If you only have a `train/` folder, create `val/` and `test/` splits:

```bash
poetry run python workspace/split_for_yolo.py /path/to/roadwork_data
```

This will:
- Create `val/` and `test/` folders
- Split 10% to validation, 10% to test, rest stays in train
- Maintains the class structure (`roadwork/` and `no_roadwork/`)

---

## Step 1.3: Configure Training Script

Edit `workspace/train_yolo.py` and set your data path:

```python
# Line 17: Update this path to your dataset
data_path = repo_root / 'roadwork_data'  # Change to your actual path
```

Or use an absolute path:
```python
data_path = Path('/absolute/path/to/roadwork_data')
```

### Adjust Training Parameters (Optional)

The script includes strong augmentation and overfitting prevention. You can adjust:

- **`epochs=100`**: Maximum training epochs (early stopping will stop earlier if needed)
- **`batch=8`**: Batch size (increase if you have more GPU memory, e.g., `batch=16` or `batch=32`)
- **`patience=10`**: Early stopping patience (epochs without improvement before stopping)
- **`weight_decay=0.0005`**: L2 regularization strength

For **GPU training** (recommended):
- Use `batch=16` or `batch=32` if you have enough GPU memory
- Training will be much faster

For **CPU training**:
- Keep `batch=8` or reduce to `batch=4` if you get out-of-memory errors
- Training will be slower (may take hours or overnight)

---

## Step 1.4: Run Training

### On Your Machine

```bash
cd /path/to/streetvision-subnet
poetry run python workspace/train_yolo.py
```

### On Google Colab (Free GPU)

1. Upload your dataset to Google Drive or Colab session
2. Upload `workspace/train_yolo.py` to Colab
3. Install dependencies:
   ```python
   !pip install ultralytics transformers datasets pillow torch torchvision
   ```
4. Update the data path in the script
5. Run:
   ```python
   !python train_yolo.py
   ```

---

## Step 1.5: Monitor Training

During training, you'll see:

- **Training progress**: Epoch number, loss, accuracy
- **Validation metrics**: Validation loss and accuracy (if `val=True`)
- **Early stopping**: Training stops automatically if validation doesn't improve for `patience` epochs
- **Plots**: Training/validation curves saved to `runs/classify/train/`

### What to Watch For

- **Training loss decreasing**: Good sign, model is learning
- **Validation loss decreasing**: Model is generalizing well
- **Validation loss increasing while training loss decreases**: Overfitting! (Early stopping should catch this)
- **Both losses plateau**: Model has converged

---

## Step 1.6: Find Your Trained Model

After training completes, your model weights are saved at:

```
runs/classify/train/weights/best.pt
```

This is your **trained YOLOv11n-cls model**. Copy it to a stable location:

```bash
cp runs/classify/train/weights/best.pt ~/my_yolo_model.pt
# or
cp runs/classify/train/weights/best.pt ./models/yolo11n_roadwork.pt
```

**Important**: Save this path! You'll need it for:
- Step 2: Precomputing branches for fusion training
- Step 3: Configuring the miner to use your YOLO model

---

## Step 1.7: Verify Your Model

Test your trained model:

```python
from ultralytics import YOLO
from PIL import Image

# Load your trained model
model = YOLO('runs/classify/train/weights/best.pt')

# Test on an image
img = Image.open('path/to/test_image.jpg')
results = model(img)

# Print predictions
for result in results:
    print(f"Class: {result.names[result.probs.top1]}, Confidence: {result.probs.top1conf:.4f}")
```

Or use the evaluation script:

```bash
poetry run python base_miner/scripts/evaluate_detector.py \
  --dataset_path imagefolder:/path/to/roadwork_data \
  --split val \
  --my_detector YOLOClassifier \
  --my_config yolo_roadwork.yaml
```

(First, update `base_miner/detectors/configs/yolo_roadwork.yaml` with your model path)

---

## Troubleshooting

### Out of Memory (OOM)

- Reduce `batch` size: `batch=4` or `batch=2`
- Reduce image size: `imgsz=160` instead of `imgsz=224`

### Training Too Slow

- Use GPU if available
- Increase `batch` size if you have GPU memory
- Reduce `epochs` for a quick test run

### Model Not Improving

- Check your data: ensure both classes have enough samples
- Increase augmentation strength
- Train for more epochs (increase `epochs` or `patience`)
- Try a larger model: `yolo11s-cls` or `yolo11m-cls`

### Early Stopping Too Early

- Increase `patience` (e.g., `patience=20`)
- Check if validation split is too small

---

## Next Steps

After successfully training YOLOv11n-cls:

1. **Step 2**: Precompute ViT + YOLO branches (see `BuildInceptionFusionStepByStep.md`)
2. **Step 3**: Train EfficientNetV2 + fusion model
3. **Step 4**: Configure miner with your models
4. **Step 5**: Evaluate and deploy

---

## Quick Reference

| Item | Value |
|------|-------|
| **Model** | `yolo11n-cls` (nano, smallest) |
| **Input size** | 224×224 pixels |
| **Classes** | 2 (no_roadwork=0, roadwork=1) |
| **Output** | `runs/classify/train/weights/best.pt` |
| **Training time** | ~30 min - 2 hours (GPU) or 4-8 hours (CPU) |
| **Recommended batch** | 8 (CPU) or 16-32 (GPU) |

---

## Summary

✅ **What you did**: Trained YOLOv11n-cls classifier on roadwork data  
✅ **What you got**: `best.pt` - your trained YOLO model weights  
✅ **What's next**: Use this model in Step 2 (precompute branches) or directly in the miner

Your YOLOv11n-cls model is now ready! This is the foundation for building your fusion model.

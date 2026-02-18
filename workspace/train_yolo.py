"""
Train YOLOv11 classifier for roadwork detection with strong augmentation and overfitting prevention.

This is STEP 1 in building your own model. After training, you'll get best.pt which you'll use in:
- Step 2: Precomputing branches for fusion training
- Step 3: Training the fusion model
- Step 4: Configuring the miner

Overfitting prevention techniques:
- Early stopping (patience=10): Stops training if validation loss doesn't improve for 10 epochs
- Weight decay (L2 regularization): Penalizes large weights to prevent overfitting
- Learning rate scheduling: Gradually reduces learning rate for stable convergence
- Strong data augmentation: Increases data diversity to improve generalization
- Validation monitoring: Tracks validation metrics to detect overfitting early

Usage:
    poetry run python workspace/train_yolo.py
    
    OR for unbuffered output (to see metrics in real-time):
    poetry run python -u workspace/train_yolo.py

See docs/Step1_TrainYOLOv11.md for detailed instructions.
"""

import sys
import os
from ultralytics import YOLO
from pathlib import Path

# Ensure output is not buffered so we see metrics in real-time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# Set environment variables to ensure verbose output
os.environ['YOLO_VERBOSE'] = 'True'

# Get the repo root directory (parent of workspace)
repo_root = Path(__file__).parent.parent

# ============================================================================
# CONFIGURATION: Set your dataset path here
# ============================================================================
# Option 1: Relative path (dataset in repo root)
data_path = repo_root / 'roadwork_data'

# Option 2: Absolute path (uncomment and set your path)
# data_path = Path('/absolute/path/to/your/roadwork_data')

# Option 3: Use Hugging Face dataset (convert to imagefolder first)
# Run: poetry run python workspace/download_louis_subset.py --output_dir ./roadwork_data
# Then use: data_path = repo_root / 'roadwork_data'

# Dataset structure should be:
#   roadwork_data/
#     train/
#       no_roadwork/  (class 0)
#       roadwork/     (class 1)
#     val/            (optional but recommended)
#       no_roadwork/
#       roadwork/
# ============================================================================

# ============================================================================
# MODEL SELECTION: Choose your YOLO model
# ============================================================================
# Option 1: YOLOv11-cls (recommended - more stable than YOLOv12)
# Ultralytics will automatically download the model if not found locally
# Available sizes: yolo11n-cls, yolo11s-cls, yolo11m-cls, yolo11l-cls, yolo11x-cls
# Note: Use 'yolo11n-cls' format (not 'yolov11n-cls')
model = YOLO('yolo11n-cls')  # nano version (smallest, fastest)

# Option 2: YOLOv12-cls (newest, but may have training instability)
# Uncomment to try YOLOv12 instead:
# model = YOLO('yolo12n-cls')  # nano version

# Option 3: Larger YOLOv11 for better accuracy (if you have GPU memory)
# model = YOLO('yolo11s-cls')  # small version
# model = YOLO('yolo11m-cls')  # medium version

# Option 4: If YOLOv11/v12 don't work, fall back to YOLOv8 (proven stable)
# model = YOLO('yolov8n-cls')  # original version you were using
# ============================================================================

print(f"Training YOLOv11n-cls on dataset: {data_path}")
print(f"Make sure your dataset has train/no_roadwork/ and train/roadwork/ folders")
print(f"Output will be saved to: runs/classify/train/weights/best.pt")
print()
print("=" * 70)
print("Training Progress:")
print("  - During each epoch: You'll see batch-by-batch loss updates (this is normal)")
print("  - After each epoch: A summary table will appear with:")
print("    Epoch    GPU_mem   train/loss   train/accuracy_top1   val/loss   metrics/accuracy_top1   lr")
print("  --------------------------------------------------------------------------------------------")
print("    1/100      8.0G       0.6234                0.7123    0.5891                 0.7234  0.01")
print("    2/100      8.0G       0.5123                0.7891    0.5012                 0.8012  0.01")
print()
print("NOTE: The epoch summary table appears AFTER each epoch completes.")
print("      During training, you'll see batch progress (loss per batch).")
print("      Wait for the epoch to finish to see the full metrics table.")
print("=" * 70)
print()

# Train the model
print("Starting training...")
print("You'll see batch-level progress during training.")
print("The epoch summary table will appear after each epoch completes.\n")
sys.stdout.flush()

results = model.train(
    data=str(data_path), 
    epochs=100,      # Increased epochs since we have early stopping
    imgsz=224,
    batch=8,         # Adjust based on your GPU/CPU memory (8 for CPU, 16-32 for GPU)
    augment=True,    # Enable augmentation
    verbose=True,    # Show detailed training output including loss and accuracy
    # Strong augmentation parameters
    hsv_h=0.05,      # Hue shift (default: 0.015, strong: 0.05)
    hsv_s=0.7,       # Saturation (default: 0.7, strong: 0.7)
    hsv_v=0.4,       # Brightness (default: 0.4, strong: 0.4)
    degrees=25,      # Rotation degrees (default: 0, strong: 25)
    translate=0.2,   # Translation (default: 0.1, strong: 0.2)
    scale=0.5,       # Scaling (default: 0.5, strong: 0.5)
    shear=10,        # Shearing (default: 0, strong: 10)
    perspective=0.0001,  # Perspective transform (default: 0, strong: 0.0001)
    flipud=0.0,      # Vertical flip probability (default: 0, strong: 0.0)
    fliplr=0.5,      # Horizontal flip probability (default: 0.5, strong: 0.5)
    mosaic=1.0,      # Mosaic augmentation probability (default: 1.0, strong: 1.0)
    mixup=0.1,       # Mixup augmentation probability (default: 0.0, strong: 0.1)
    # Overfitting prevention
    patience=10,     # Early stopping patience: stop after 10 epochs without improvement
    weight_decay=0.0005,  # L2 regularization (weight decay) to prevent overfitting
    val=True,        # Enable validation during training
    plots=True,      # Generate training/validation plots to monitor overfitting
    save=True,       # Save checkpoints
    save_period=10,  # Save checkpoint every 10 epochs
    # Learning rate scheduling (helps prevent overfitting)
    lr0=0.01,        # Initial learning rate
    lrf=0.01,        # Final learning rate (lr0 * lrf)
    warmup_epochs=3, # Warmup epochs for stable training start
)

# Print final training results
print()
print("=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
if results:
    print("\nFinal Training Metrics:")
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"  Best epoch: {metrics.get('metrics/best_epoch', 'N/A')}")
        if 'train/loss' in metrics:
            print(f"  Best train loss: {metrics['train/loss']:.4f}")
        if 'val/loss' in metrics:
            print(f"  Best val loss: {metrics['val/loss']:.4f}")
        if 'train/accuracy_top1' in metrics:
            print(f"  Best train accuracy: {metrics['train/accuracy_top1']:.4f}")
        if 'metrics/accuracy_top1' in metrics:
            print(f"  Best val accuracy: {metrics['metrics/accuracy_top1']:.4f}")
    elif hasattr(results, 'results'):
        print(f"  Results: {results.results}")
    else:
        print(f"  Training completed successfully")

print("\nYour model is saved at:")
print("  runs/classify/train/weights/best.pt")
print("\nNext steps:")
print("  1. Copy best.pt to a stable location")
print("  2. Use it in Step 2: Precompute branches (see BuildInceptionFusionStepByStep.md)")
print("  3. Or configure it directly in yolo_roadwork.yaml for the miner")
print("=" * 70)

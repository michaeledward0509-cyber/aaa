# CPU-Only Optimization Guide (16GB RAM)

## Can You Run This on CPU with 16GB RAM?

**Yes, but with limitations and optimizations.** While the `min_compute.yml` specifies GPU as "required", the codebase fully supports CPU execution. However, you'll need to make strategic choices about models and configurations.

## What's Possible

### ✅ **Mining/Inference** (Recommended for CPU)
- **Single model inference** (ViT or YOLO) - **Fully feasible**
- **Smaller batch sizes** - Use batch_size=1 for inference
- **Lighter models** - Use ViT-base or YOLO-nano instead of larger variants

### ⚠️ **Training** (Challenging but possible)
- **Fine-tuning small models** - Possible with very small batches
- **Training from scratch** - Not recommended (too slow)
- **Precomputed branches approach** - Best strategy (see below)

## Key Limitations

1. **Speed**: CPU inference is 10-50x slower than GPU
2. **Memory**: 16GB RAM limits batch sizes and model sizes
3. **Concurrent requests**: May struggle with multiple simultaneous requests
4. **Training**: Very slow; use precomputed branches or cloud GPU

## Recommended Setup for CPU Mining

### 1. Use Lightweight Models

**Option A: ViT-base (Recommended)**
```yaml
# Use the default ViT model - it's already optimized
IMAGE_DETECTOR=ViT
IMAGE_DETECTOR_CONFIG=ViT_roadwork.yaml
IMAGE_DETECTOR_DEVICE=cpu
```

**Option B: YOLO-nano (Smallest)**
```yaml
IMAGE_DETECTOR=YOLOClassifier
IMAGE_DETECTOR_CONFIG=yolo_roadwork.yaml
IMAGE_DETECTOR_DEVICE=cpu
```

### 2. Optimize Your miner.env

```bash
# Device Settings - CRITICAL for CPU
IMAGE_DETECTOR_DEVICE=cpu
VIDEO_DETECTOR_DEVICE=cpu

# Use default lightweight models
IMAGE_DETECTOR=ViT
IMAGE_DETECTOR_CONFIG=ViT_roadwork.yaml

# Network settings
NETUID=323  # Testnet first to test
SUBTENSOR_NETWORK=test
```

### 3. Memory Optimizations

#### A. Set Environment Variables (Before Running)
```bash
# Limit PyTorch threads to avoid memory spikes
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Use CPU-friendly PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### B. Python Memory Settings
Add to your startup script:
```python
import torch
torch.set_num_threads(4)  # Limit CPU threads
torch.set_num_interop_threads(2)
```

### 4. Model Loading Optimizations

The codebase already supports CPU, but you can optimize further by modifying detector loading:

**For ViT Detector** (`base_miner/detectors/vit_detector.py`):
- Already uses `device="cpu"` parameter
- Pipeline handles preprocessing efficiently
- Consider using `torch.float32` instead of `float16` on CPU (better compatibility)

**For YOLO Detector** (`base_miner/detectors/yolo_classifier_detector.py`):
- Already supports CPU via `device="cpu"`
- Uses Ultralytics which has good CPU support
- Consider using YOLO-nano (smallest) variant

## Training on CPU (Advanced)

### Strategy: Precomputed Branches Approach

This is the **best approach** for CPU training with limited RAM:

1. **Precompute on Cloud GPU** (Google Colab/Kaggle - FREE)
   - Train YOLO once on free GPU
   - Precompute ViT + YOLO features → save cache
   - Download cache to your laptop

2. **Train Fusion on CPU** (Uses only cache, not full models)
   ```bash
   poetry run python base_miner/scripts/train_inception_fusion_keras.py \
     --dataset_path imagefolder:/path/to/data \
     --precomputed_branches ./branch_cache.npz \
     --output_dir ./fusion_model \
     --epochs 10 \
     --batch_size 4  # Small batch for 16GB RAM
   ```

See `docs/TrainingOnCPUAndFreeCloud.md` for detailed instructions.

## Performance Expectations

### Inference Speed (Single Image)
- **GPU (GTX 1660)**: ~10-50ms per image
- **CPU (Modern 4-core)**: ~200-1000ms per image
- **CPU (Older 2-core)**: ~500-2000ms per image

### Memory Usage
- **ViT-base on CPU**: ~2-4GB RAM
- **YOLO-nano on CPU**: ~1-2GB RAM
- **System overhead**: ~2-4GB RAM
- **Total**: ~5-10GB RAM (fits in 16GB with margin)

## Troubleshooting

### Out of Memory Errors

1. **Reduce batch size**: Set `batch_size=1` in inference
2. **Use smaller model**: Switch to YOLO-nano or ViT-small
3. **Close other applications**: Free up RAM
4. **Add swap space**: 
   ```bash
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### Slow Performance

1. **Use fewer threads**: `export OMP_NUM_THREADS=2`
2. **Disable unnecessary features**: Turn off video detector if not needed
3. **Use optimized PyTorch**: Install CPU-optimized build:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

### Model Loading Issues

1. **Check disk space**: Models download from HuggingFace (~500MB-2GB)
2. **Use local cache**: Set `HF_HOME` environment variable
3. **Pre-download models**: Run once before mining:
   ```python
   from transformers import AutoModelForImageClassification
   AutoModelForImageClassification.from_pretrained('natix-network-org/roadwork')
   ```

## Best Practices

1. **Start on Testnet**: Test your setup on testnet (netuid 323) first
2. **Monitor Resources**: Use `htop` or `top` to watch CPU/RAM usage
3. **Use Default Model First**: Start with the default ViT model before training custom ones
4. **Optimize Incrementally**: Get it working first, then optimize
5. **Consider Cloud GPU**: For training, use free Colab/Kaggle GPU

## Configuration Checklist

- [ ] Set `IMAGE_DETECTOR_DEVICE=cpu` in `miner.env`
- [ ] Use lightweight model (ViT or YOLO-nano)
- [ ] Set environment variables for thread limits
- [ ] Test on testnet first
- [ ] Monitor memory usage
- [ ] Have swap space configured
- [ ] Close unnecessary applications

## Expected Rewards

**Important**: CPU miners will have:
- **Slower response times** → May timeout on some validators
- **Lower throughput** → Fewer concurrent requests handled
- **Same accuracy** → Model quality doesn't depend on CPU vs GPU

You may earn **lower rewards** compared to GPU miners due to slower response times, but you can still participate and earn rewards if your model is accurate.

## Next Steps

1. **Test Setup**: Run miner on testnet with CPU
2. **Monitor Performance**: Check response times and memory usage
3. **Optimize**: Adjust batch sizes and model choices
4. **Consider Training**: Use cloud GPU for training, CPU for inference
5. **Upgrade Path**: If profitable, consider investing in a GPU later

## Additional Resources

- `docs/TrainingOnCPUAndFreeCloud.md` - Detailed CPU training guide
- `docs/Mining.md` - General mining setup
- `docs/Training.md` - Training guide (use with cloud GPU)

---

**Bottom Line**: Yes, you can run this on CPU with 16GB RAM, especially for mining/inference. Training is challenging but possible with the precomputed branches approach. Start simple, test thoroughly, and optimize incrementally.

# Ideas to Improve the Inception Fusion Model

After you have built the fusion model (see [BuildInceptionFusionStepByStep.md](BuildInceptionFusionStepByStep.md)), you can try the following to improve accuracy and robustness.

---

## 1. Deeper fusion head (MLP)

The default fusion is a single linear layer (3 → 2). A **small MLP** (3 → 16 → 2 with ReLU and dropout) can learn non-linear combinations of the three branch outputs and sometimes generalizes better.

**PyTorch:**
```bash
poetry run python base_miner/scripts/train_inception_fusion.py \
  --dataset_path imagefolder:/path/to/roadwork_data \
  --precomputed_branches ./branch_cache.npz \
  --output_dir ./inception_fusion_weights \
  --fusion_mlp \
  --epochs 10 --batch_size 16
```

**Keras:**
```bash
poetry run python base_miner/scripts/train_inception_fusion_keras.py \
  --dataset_path imagefolder:/path/to/roadwork_data \
  --precomputed_branches ./branch_cache.npz \
  --output_dir ./inception_fusion_keras \
  --fusion_mlp \
  --epochs 10 --batch_size 16
```

The detector loads the MLP automatically when the checkpoint was trained with `--fusion_mlp` (PyTorch saves a `fusion_head_type` field). For Keras, the saved `.keras` file already encodes the architecture.

---

## 2. Train longer and tune learning rate

- **More epochs:** Try 15–20 instead of 10 (watch for overfitting if you have a validation set).
- **Learning rate:** Lower the LR for a finer fit, e.g. `--lr 5e-4` or `3e-4`. Use a **learning-rate schedule** (e.g. cosine decay) if you add it to the script.

---

## 3. Fine-tune the EfficientNetV2 backbone (Keras)

Currently the EfficientNetV2 backbone is frozen; only the top 2-class head is trained. You can **unfreeze the last few blocks** and train with a small LR so the backbone adapts to roadwork scenes.

In `train_inception_fusion_keras.py`, in `build_efficientnetv2_keras()`, set `base.trainable = True` and optionally use a lower LR for the base (e.g. 1e-5) and a higher LR for the new head (e.g. 1e-3). This requires more compute and care to avoid overfitting.

---

## 4. Use a slightly larger YOLO

If you have enough RAM/GPU, train **YOLOv11s-cls** or **YOLOv11m-cls** (or `yolo11l-cls`) instead of **YOLOv11n-cls**. Larger models often generalize better at the cost of speed and memory. Available sizes: n (nano), s (small), m (medium), l (large), x (xlarge).

---

## 5. Early stopping (built-in)

Training scripts support **early stopping** so training stops when validation loss stops improving.

- **`--validation_split 0.15`** (default): use 15% of the data for validation; the rest for training.
- **`--early_stopping_patience 3`** (default): stop after 3 epochs with no validation loss improvement.
- The **best checkpoint** (by validation loss) is saved; when early stopping triggers, that best model is what you get.
- To disable: `--validation_split 0` (no validation, no early stopping).

Example:
```bash
poetry run python base_miner/scripts/train_inception_fusion_keras.py \
  --dataset_path imagefolder:/path/to/roadwork_data \
  --precomputed_branches ./branch_cache.npz \
  --output_dir ./inception_fusion_keras \
  --epochs 20 \
  --validation_split 0.15 \
  --early_stopping_patience 3
```

---

## 6. Test-time augmentation (TTA)

At inference, run the model on **several augmented views** of the same image (e.g. original + horizontal flip + small rotations) and **average** the predicted probabilities. This can improve robustness at the cost of 2–4× inference time. You would implement this in the detector’s `__call__` by applying a few fixed transforms, running each view through the pipeline, and averaging the fusion output.

---

## 7. Stronger augmentation (if not overfitting)

If you have enough data and the model is not overfitting, try **moderate** or **strong** augmentation:

```bash
--augment_strength moderate   # or strong
```

---

## 8. Align with how the validator scores

If the subnet documentation or code describes how miners are scored (e.g. agreement with a reference model, or accuracy on a hidden set), you can:

- Prefer **models or fusion strategies** that maximize that objective.
- Optionally **calibrate** the fusion output (e.g. temperature scaling) on a held-out set so that confidence matches accuracy.

---

## Summary

| Idea | Effort | When to try |
|------|--------|-------------|
| **Deeper fusion (--fusion_mlp)** | Low | First thing to try after the baseline. |
| More epochs / lower LR | Low | If validation loss is still improving at epoch 10. |
| Fine-tune EfficientNetV2 backbone | Medium | If you have compute and a validation set to monitor overfitting. |
| Larger YOLO (s/m) | Low (retrain YOLO + re-precompute) | If you have RAM/GPU headroom. |
| Validation + early stopping | Medium (script change) | To reduce overfitting. |
| TTA at inference | Medium (detector change) | For extra robustness when latency is acceptable. |
| Stronger augmentation | Low | When the model is not overfitting. |

Start with **--fusion_mlp** and **more epochs**; then add the others as needed.

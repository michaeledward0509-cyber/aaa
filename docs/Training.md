# Building Your Own Model (Easy & Efficient)

The **easiest and most efficient** way to build your own model is to **fine-tune a pre-trained model** on roadwork data. Training from scratch is slower and usually unnecessary.

---

## Pre-trained models you can use

| Pre-trained model | Use case |
|-------------------|----------|
| **`natix-network-org/roadwork`** | **Recommended.** Already fine-tuned for roadwork by the subnet. Use this as your starting point and fine-tune further on your own data (or more data) to improve. |
| **`google/vit-base-patch16-224`** | Generic Vision Transformer (ImageNet). Use if you want to train a roadwork classifier from a general vision model. |

The miner expects a **binary** classifier with two labels: **"Roadwork"** (class 1) and **"None"** (class 0). The fine-tune script below produces a model in that format.

---

## Quick start: fine-tune with the included script

### 1. Prepare your data

Use an **imagefolder** layout (binary labels):

```
/path/to/roadwork_data/
  train/
    no_roadwork/    ← label 0
      img1.jpg
      img2.png
    roadwork/       ← label 1
      img1.jpg
      img2.png
```

You can use the existing [natix roadwork dataset](https://huggingface.co/datasets/natix-network-org/roadwork) (see [DataSources.md](DataSources.md)) or your own. For more data sources, see [DataSources.md](DataSources.md).

### 2. Run the fine-tune script

From the repo root:

**Option A: Fine-tune the default subnet model (recommended)**

```bash
poetry run python base_miner/scripts/finetune_vit_roadwork.py \
  --dataset_path imagefolder:/path/to/roadwork_data \
  --output_dir ./my_roadwork_model \
  --base_model natix-network-org/roadwork \
  --epochs 3 \
  --batch_size 16
```

**Option B: Start from a generic ViT**

```bash
poetry run python base_miner/scripts/finetune_vit_roadwork.py \
  --dataset_path imagefolder:/path/to/roadwork_data \
  --output_dir ./my_roadwork_model \
  --base_model google/vit-base-patch16-224 \
  --epochs 5 \
  --batch_size 16
```

**Use a GPU** for speed; the script uses CUDA if available. Optional: `--fp16` for mixed precision, `--lr 1e-5` to lower the learning rate.

### 3. Use your model in the miner

- **Local:** Create a config YAML (e.g. `base_miner/detectors/configs/my_roadwork.yaml`) with:

  ```yaml
  hf_repo: "/absolute/path/to/my_roadwork_model"   # or your Hugging Face repo
  config_name: "config.json"
  weights: "model.safetensors"
  ```

  Then in `miner.env`: `IMAGE_DETECTOR=ViT`, `IMAGE_DETECTOR_CONFIG=my_roadwork.yaml`.

- **Hugging Face:** Upload the contents of `--output_dir` to a new HF repo (e.g. `your-username/roadwork-v1`). Set `hf_repo: "your-username/roadwork-v1"` in the config. See [Mining.md](Mining.md) for submission (model_card.json, MODEL_URL).

### 4. Evaluate before deploying

Compare your model to the default on a held-out test set:

```bash
poetry run python base_miner/scripts/evaluate_detector.py \
  --dataset_path imagefolder:/path/to/roadwork_data \
  --split test \
  --my_detector ViT \
  --my_config my_roadwork.yaml
```

See [Evaluation.md](Evaluation.md) for details.

---

## Why fine-tune instead of training from scratch?

- **Faster:** You need fewer epochs and less data.
- **Better:** Pre-trained weights already capture general visual features; you only adapt the head (and optionally last layers) to “roadwork vs no roadwork.”
- **Compatible:** The script keeps the same interface (ViT + “Roadwork” / “None”) so the existing miner and evaluation scripts work without code changes.

---

## Summary

| Step | Action |
|------|--------|
| 1 | Get roadwork + no_roadwork images in imagefolder layout (see [DataSources.md](DataSources.md)). |
| 2 | Run `finetune_vit_roadwork.py` with `--base_model natix-network-org/roadwork` (or `google/vit-base-patch16-224`). |
| 3 | Save to `--output_dir`; optionally upload to Hugging Face. |
| 4 | Add a detector config YAML pointing to your model; set it in `miner.env`. |
| 5 | Run [evaluate_detector.py](../base_miner/scripts/evaluate_detector.py) to compare with the default, then deploy. |

Using the **default pre-trained model** (`natix-network-org/roadwork`) as your base is the easiest and most efficient way to build your own model for the subnet.

**Using YOLO (classification) with subnet-only data:** If you cannot add external data and want to try a different architecture (e.g. YOLO), see [TrainingYOLO.md](TrainingYOLO.md) for training a YOLO classifier and plugging it into the miner.

**Combining ViT + YOLO + GoogLeNet (Inception-style fusion):** To fuse the default model, YOLO, and GoogLeNet with a learned fusion layer using only subnet data, see [TrainingInceptionFusion.md](TrainingInceptionFusion.md).

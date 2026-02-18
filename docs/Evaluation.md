# Evaluating Your Model vs the Default

Before deploying your own model on the subnet, you can evaluate it **locally** using the **same metrics** the validators use, and compare directly to the default ViT model.

## Metrics (same as the subnet)

Validators score miners using:

- **Accuracy** – fraction of correct binary predictions (after rounding probability to 0/1).
- **MCC** (Matthews correlation coefficient) – balanced measure that rewards good calibration.
- **Reward** – `(0.5 * MCC + 0.5 * accuracy) ** 3.0` (same formula as in `natix/validator/reward.py`).

So the script below gives you a **subnet-style score** on your own test set.

## 1. Prepare a labeled test set

You need images with **binary labels**: **0** = no roadwork, **1** = roadwork.

**Option A: Imagefolder (recommended)**

Create a folder structure:

```
/path/to/roadwork_eval/
  train/              # or only "test" if you prefer
    no_roadwork/      # label 0
      img1.jpg
      img2.png
    roadwork/         # label 1
      img1.jpg
      img2.png
```

Class names can be anything; the loader will assign 0 and 1. Use a **separate test (or validation) split** if you want to avoid evaluating on training data.

**Option B: Hugging Face dataset**

Use any dataset that has **`image`** and **`label`** (or **`labels`**) columns, with labels 0 and 1.

## 2. Run the evaluation script

From the repo root:

```bash
# Evaluate only the default model (ViT_roadwork)
poetry run python base_miner/scripts/evaluate_detector.py \
  --dataset_path imagefolder:/path/to/roadwork_eval \
  --split test \
  --device cuda
```

**Compare your model vs default** (your config points to your weights, e.g. your Hugging Face repo):

```bash
poetry run python base_miner/scripts/evaluate_detector.py \
  --dataset_path imagefolder:/path/to/roadwork_eval \
  --split test \
  --my_detector ViT \
  --my_config my_roadwork.yaml \
  --device cuda
```

Your config `my_roadwork.yaml` should look like `ViT_roadwork.yaml` but with your `hf_repo` (and optional `weights` / `config_name`). The script loads both detectors, runs them on the same samples, and prints metrics for each plus a short comparison.

**Evaluate only your model** (skip default):

```bash
poetry run python base_miner/scripts/evaluate_detector.py \
  --dataset_path imagefolder:/path/to/roadwork_eval \
  --no_default \
  --my_detector ViT \
  --my_config my_roadwork.yaml
```

**Quick run** (cap number of samples):

```bash
poetry run python base_miner/scripts/evaluate_detector.py \
  --dataset_path imagefolder:/path/to/roadwork_eval \
  --max_samples 200
```

## 3. Interpret the output

Example:

```
Evaluation (subnet-style metrics)
============================================================

default:
  accuracy   = 0.8520
  mcc        = 0.7012
  reward_raw = 0.7766  (0.5*mcc + 0.5*acc)
  reward     = 0.4675  (reward_raw^3)
  n_valid    = 500/500

yours:
  accuracy   = 0.8880
  mcc        = 0.7721
  ...

--- Comparison (yours vs default) ---
  accuracy: 0.8880 vs 0.8520  (better)
  mcc:      0.7721 vs 0.7012  (better)
  reward:   0.5234 vs 0.4675  (better)
```

- **reward** is what drives your share of emissions on the subnet (after validators set weights from many challenges). Higher reward on a representative test set suggests your model will perform better on the network.
- Use a **held-out test set** (not used for training) and, if possible, data that resembles what validators use (e.g. dashcam/street imagery with and without roadwork).

## 4. Add your detector config

To evaluate a **custom model** (e.g. your own ViT fine-tune), add a YAML under `base_miner/detectors/configs/`, e.g. `my_roadwork.yaml`:

```yaml
hf_repo: "your-hf-username/your-roadwork-model"
config_name: "config.yaml"   # if different
weights: "model.safetensors" # if different
```

If your detector is a **new class** (not ViT), implement it and register it in `base_miner/registry.py` (and use that name for `--my_detector`).

## Summary

| Goal | Command |
|------|--------|
| See default model metrics | `evaluate_detector.py --dataset_path imagefolder:...` |
| Compare your model to default | Add `--my_detector ViT --my_config my_roadwork.yaml` |
| Only evaluate your model | Add `--no_default` and your detector/config |

Using this script lets you iterate on data and architecture and see how your model compares to the default **before** registering and mining on mainnet.

# How to Configure the Miner

This guide covers configuring the miner to use your trained **Inception Fusion** model (ViT + YOLOv11 + EfficientNetV2).

---

## 1. Create or edit the detector config

Config files live in **`base_miner/detectors/configs/`**.

**Option A – Use the existing fusion config**

Edit **`base_miner/detectors/configs/inception_fusion_roadwork.yaml`** (or **`my_inception_fusion.yaml`**).

**Option B – Copy and use your own**

```bash
cp base_miner/detectors/configs/inception_fusion_roadwork.yaml base_miner/detectors/configs/my_inception_fusion.yaml
```

Then edit **`my_inception_fusion.yaml`** and set **absolute paths** so they work no matter where you start the miner from:

```yaml
vit_config: ViT_roadwork.yaml
yolo_weights_path: /home/dr/myenv/streetvision-subnet/workspace/best.pt
googlenet_fusion_path: /home/dr/myenv/streetvision-subnet/inception_fusion_keras
use_keras: true
roadwork_class_index: 1
```

| Field | Meaning |
|-------|--------|
| **vit_config** | ViT config (default `ViT_roadwork.yaml`). Leave as is unless you use a custom ViT. |
| **yolo_weights_path** | Path to your YOLO **best.pt** (e.g. `workspace/best.pt` or `runs/classify/train14/weights/best.pt`). Use **absolute path** to avoid "not found" when starting from another dir. |
| **googlenet_fusion_path** | Path to the **Keras output directory** (contains `efficientnetv2_branch.keras` and `fusion_head.keras`), or path to the PyTorch **.pt** file if you used the PyTorch script. Use **absolute path**. |
| **use_keras** | `true` if you trained with `train_inception_fusion_keras.py`; `false` if you used `train_inception_fusion.py` (GoogLeNet .pt). |
| **roadwork_class_index** | YOLO class index for roadwork (usually `1`). |

---

## 2. Set miner environment (miner.env)

Create **`miner.env`** in the repo root if it doesn’t exist:

```bash
./setup_env.sh
```

Then edit **`miner.env`** and set at least:

```bash
# Use your Inception Fusion model
IMAGE_DETECTOR=InceptionFusion
IMAGE_DETECTOR_CONFIG=my_inception_fusion.yaml
IMAGE_DETECTOR_DEVICE=cuda

# Or if no GPU:
# IMAGE_DETECTOR_DEVICE=cpu

# Network (mainnet)
NETUID=72
SUBTENSOR_NETWORK=finney
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443

# Your wallet
WALLET_NAME=your_wallet
WALLET_HOTKEY=your_hotkey

# Miner port (must be open from internet if you register on mainnet)
MINER_AXON_PORT=8091

# Hugging Face repo for “submitted” model (required by subnet)
MODEL_URL=https://huggingface.co/your-username/roadwork-miner
```

| Variable | Meaning |
|----------|--------|
| **IMAGE_DETECTOR** | Detector class: `InceptionFusion` for fusion; `ViT` for default only; `YOLOClassifier` for YOLO only. |
| **IMAGE_DETECTOR_CONFIG** | Config filename in `base_miner/detectors/configs/` (e.g. `my_inception_fusion.yaml` or `inception_fusion_roadwork.yaml`). |
| **IMAGE_DETECTOR_DEVICE** | `cuda` or `cpu`. |
| **NETUID** | `72` mainnet, `323` testnet. |
| **WALLET_NAME** / **WALLET_HOTKEY** | Your Bittensor wallet. |
| **MINER_AXON_PORT** | Port the miner listens on (e.g. 8091). |
| **MODEL_URL** | Hugging Face repo URL for your “submitted” model (see [Mining.md](Mining.md)). |

---

## 3. Paths: absolute vs relative

- **YAML (yolo_weights_path, googlenet_fusion_path):** Resolved from the **current working directory** when the miner starts. To avoid “file not found”, use **absolute paths** in the YAML (e.g. `/home/dr/myenv/streetvision-subnet/workspace/best.pt`).
- **miner.env:** Usually in repo root; `IMAGE_DETECTOR_CONFIG` is just the **filename** (e.g. `my_inception_fusion.yaml`), not a path.

---

## 4. Start the miner

From the **repo root**:

```bash
source miner.env
./start_miner.sh
```

Or:

```bash
source miner.env
poetry run python neurons/miner.py \
  --neuron.image_detector ${IMAGE_DETECTOR} \
  --neuron.image_detector_config ${IMAGE_DETECTOR_CONFIG} \
  --neuron.image_detector_device ${IMAGE_DETECTOR_DEVICE} \
  --netuid ${NETUID} \
  --subtensor.network ${SUBTENSOR_NETWORK} \
  ...
```

The miner will load ViT, YOLO, and your EfficientNetV2 + fusion from the paths in the YAML.

---

## 5. Quick checklist

- [ ] YOLO **best.pt** path in YAML is **absolute** and the file exists.
- [ ] **googlenet_fusion_path** in YAML points to the **directory** with `efficientnetv2_branch.keras` and `fusion_head.keras` (Keras) or to the **.pt** file (PyTorch), and is **absolute**.
- [ ] **use_keras: true** in YAML if you used the Keras fusion script.
- [ ] **miner.env**: `IMAGE_DETECTOR=InceptionFusion`, `IMAGE_DETECTOR_CONFIG=my_inception_fusion.yaml` (or your config name), `IMAGE_DETECTOR_DEVICE=cuda` or `cpu`.
- [ ] **MODEL_URL** in miner.env set to your Hugging Face “submission” repo.
- [ ] Wallet and NETUID/SUBTENSOR_* set for mainnet or testnet.
- [ ] Miner started from repo root so relative paths in code resolve correctly; using absolute paths in YAML avoids CWD issues.

---

## 6. Other detector types

| Detector | IMAGE_DETECTOR | Config example |
|----------|----------------|-----------------|
| Default ViT | `ViT` | `ViT_roadwork.yaml` |
| YOLO only | `YOLOClassifier` | `yolo_roadwork.yaml` (set `weights_path`) |
| Fusion (ViT + YOLO + EfficientNetV2) | `InceptionFusion` | `inception_fusion_roadwork.yaml` or `my_inception_fusion.yaml` |

For more on registration, ports, and MODEL_URL, see [Mining.md](Mining.md).

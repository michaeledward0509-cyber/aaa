# Running the Miner on a VPS

Run the StreetVision miner on a cloud VPS so validators can reach you without port forwarding at home. The VPS has a public IP and you control the firewall.

---

## 1. VPS requirements

| Item | Recommendation |
|------|----------------|
| **OS** | Ubuntu 22.04 LTS (or 20.04) |
| **CPU** | 2+ cores (4+ for comfortable inference) |
| **RAM** | 8 GB minimum, 16 GB preferred for fusion model |
| **Disk** | 20 GB+ (models and dependencies) |
| **GPU** | Optional; miner runs on CPU. Use CUDA VPS if you want faster inference. |
| **Network** | Public IP, outbound internet allowed |

**Providers:** DigitalOcean, Vultr, Hetzner, Linode, AWS EC2, etc. Pick a region close to you or to the subnet for lower latency.

---

## 2. Create the VPS and connect

1. Create an Ubuntu 22.04 instance and note its **public IP** (e.g. `203.0.113.50`).
2. Add your SSH key or set a password. Connect from your PC:
   ```bash
   ssh root@203.0.113.50
   ```
   (Use the user/IP your provider gives you.)

---

## 3. Install dependencies on the VPS

```bash
# Update and install basics
apt update && apt install -y git python3.11 python3.11-venv python3-pip

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"

# Clone the repo (use your fork or the official repo)
git clone https://github.com/natixnetwork/natix-subnet.git
cd natix-subnet

# Or if you use this repo name:
# git clone <your-repo-url> streetvision-subnet && cd streetvision-subnet

poetry env use 3.11
poetry install
```

---

## 4. Copy your model and config to the VPS

You need on the VPS:

- **YOLO weights:** `best.pt`
- **Keras fusion:** folder with `efficientnetv2_branch.keras` and `fusion_head.keras` (e.g. `inception_fusion_keras/`)
- **Detector config:** e.g. `my_inception_fusion.yaml` with paths that exist **on the VPS**

**Option A – Copy from your local machine (from your PC, not the VPS)**

```bash
# From your Windows PC (PowerShell or WSL) or from your VM, run:
scp /path/to/workspace/best.pt root@203.0.113.50:/root/streetvision-subnet/workspace/
scp -r /path/to/inception_fusion_keras root@203.0.113.50:/root/streetvision-subnet/
```

Adjust paths: use your repo path and the VPS IP. If the repo is `natix-subnet` on the VPS, use `/root/natix-subnet/` instead of `streetvision-subnet`.

**Option B – Upload to Hugging Face / S3 and download on VPS**

Upload `best.pt` and the fusion folder somewhere (e.g. Hugging Face Hub, S3), then on the VPS:

```bash
cd /root/natix-subnet  # or your repo path
mkdir -p workspace
# Example: wget or curl to download best.pt and the fusion folder, then extract
```

**On the VPS: create or edit the detector config**

```bash
cd /root/natix-subnet  # or your repo path
nano base_miner/detectors/configs/my_inception_fusion.yaml
```

Use **absolute paths on the VPS**, for example:

```yaml
vit_config: ViT_roadwork.yaml
yolo_weights_path: /root/natix-subnet/workspace/best.pt
googlenet_fusion_path: /root/natix-subnet/inception_fusion_keras
use_keras: true
roadwork_class_index: 1
```

Save and exit.

---

## 5. Configure miner.env on the VPS

```bash
cp miner.env miner.env.bak   # if miner.env exists
nano miner.env
```

Set at least:

```bash
IMAGE_DETECTOR=InceptionFusion
IMAGE_DETECTOR_CONFIG=my_inception_fusion.yaml
IMAGE_DETECTOR_DEVICE=cpu
# Or IMAGE_DETECTOR_DEVICE=cuda if the VPS has a GPU and CUDA installed

NETUID=72
SUBTENSOR_NETWORK=finney
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443

WALLET_NAME=drwallet
WALLET_HOTKEY=5DtcmJKoriTEWWSHMeca2mjUTMxJFyPRyK9Jh6nohUenxSnH

MINER_AXON_PORT=8091
MODEL_URL=https://huggingface.co/MichaelEdward/roadwork-miner
```

Use your own `WALLET_NAME`, `WALLET_HOTKEY`, and `MODEL_URL`. For testnet use `NETUID=323` and the test endpoint.

---

## 6. Open port 8091 on the VPS

Validators must reach the miner on TCP 8091.

**A) UFW (Linux firewall on the VPS)**

```bash
ufw allow 8091/tcp
ufw allow 22/tcp   # keep SSH
ufw enable
ufw status
```

**B) Provider firewall / security group**

In the cloud dashboard (DigitalOcean, AWS, etc.), add an **inbound** rule: **TCP port 8091**, source **0.0.0.0/0** (or “any”). Save.

---

## 7. Run the miner

```bash
cd /root/natix-subnet   # or your repo path
source miner.env
./start_miner.sh
```

Or:

```bash
poetry run python neurons/miner.py \
  --neuron.image_detector InceptionFusion \
  --neuron.image_detector_config my_inception_fusion.yaml \
  --neuron.image_detector_device cpu \
  --netuid 72 \
  --subtensor.network finney \
  --wallet.name drwallet \
  --wallet.hotkey 5DtcmJKoriTEWWSHMeca2mjUTMxJFyPRyK9Jh6nohUenxSnH \
  --axon.port 8091
```

(Adjust wallet name/hotkey and other args to match your `miner.env`.)

**Run in background (so it keeps running after you disconnect)**

```bash
# Using screen
apt install -y screen
screen -S miner
source miner.env && ./start_miner.sh
# Press Ctrl+A then D to detach. Reattach with: screen -r miner

# Or using tmux
apt install -y tmux
tmux new -s miner
source miner.env && ./start_miner.sh
# Press Ctrl+B then D to detach. Reattach with: tmux attach -t miner
```

---

## 8. Check that the port is open from the internet

From your PC or phone, open:

**https://www.yougetsignal.com/tools/open-ports/**

- **Remote Address:** your VPS public IP (e.g. `203.0.113.50`)
- **Port Number:** `8091`
- Click **Check**

If it says **Port 8091 is open**, validators can reach your miner.

---

## 9. Checklist

- [ ] VPS created (Ubuntu 22.04), SSH works
- [ ] Repo cloned, `poetry install` done
- [ ] `best.pt` and `inception_fusion_keras/` (or your fusion folder) on the VPS
- [ ] `base_miner/detectors/configs/my_inception_fusion.yaml` has **absolute paths on the VPS**
- [ ] `miner.env` has correct wallet, MODEL_URL, NETUID, and detector config
- [ ] Port 8091 open (UFW + provider firewall)
- [ ] Miner started (`./start_miner.sh` or equivalent)
- [ ] Hotkey registered on the subnet (see [Mining.md](Mining.md#registration))

For detector and wallet details, see [ConfigureMiner.md](ConfigureMiner.md) and [Mining.md](Mining.md).

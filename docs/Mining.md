# Miner Guide

## Table of Contents

1. [Installation 🔧](#installation)

   * [Data 📊](#data)
   * [Registration ✍️](#registration)
2. [Mining ⛏️](#mining)

## Before You Proceed ⚠️

**IMPORTANT**: If you are new to Bittensor, we recommend familiarizing yourself with the basics on the [Bittensor Website](https://bittensor.com/) before proceeding.

**Ensure you are running Subtensor locally** to minimize outages and improve performance. See [Run a Subtensor Node Locally](https://github.com/opentensor/subtensor/blob/main/docs/running-subtensor-locally.md#compiling-your-own-binary).

**Be aware of the minimum compute requirements** for our subnet, detailed in [Minimum compute YAML configuration](../min_compute.yml). A GPU is recommended for training, but not required for inference while running a miner.

## Installation

Download the repository and navigate to the folder:

```bash
git clone https://github.com/natixnetwork/natix-subnet.git && cd natix-subnet
```

We recommend using a [Poetry](https://python-poetry.org/docs/) environment with Python 3.11 to manage dependencies:

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry env use 3.11
poetry install
```

This will install all necessary dependencies and prepare your environment.

### Data

*Only required for training -- deployed miner instances do not need access to these datasets.*

For where to get **roadwork / no-roadwork** data and how to add your own sources, see [DataSources.md](DataSources.md).

Optionally, pre-download the training datasets by running:

```bash
poetry run python base_miner/datasets/download_data.py
```

The default list of datasets and their download location is defined in `base_miner/config.py`.

## Mining Requirements ⚠️

To mine on our subnet, you must have a registered hotkey and [have submitted at least one model](#submitted-a-model).

### No-code miner (use the default model)

If you **do not want to train or build your own model**, you can run the miner with the **default public model** (ViT trained on roadwork). Do the following:

1. **Create a Hugging Face repo for “submission”**  
   The subnet requires that each miner has “submitted” a model. Create a new repo on [huggingface.co](https://huggingface.co) (e.g. `your-username/roadwork-miner`). In that repo, add a file **`model_card.json`** with your **hotkey address** and metadata, for example:

   ```json
   {
     "model_name": "StreetVision default miner",
     "description": "Miner using default ViT roadwork classifier (natix-network-org/roadwork).",
     "version": "1.0.0",
     "submitted_by": "YOUR_HOTKEY_SS58_ADDRESS",
     "submission_time": 1234567890
   }
   ```
   Replace `YOUR_HOTKEY_SS58_ADDRESS` with your Bittensor hotkey (e.g. `5Fmvr2...`) and `submission_time` with a Unix timestamp. You do **not** need to upload model weights; the miner will load the default weights from the config (see step 3).

2. **Configure `miner.env`**  
   Use the defaults for the detector and point `MODEL_URL` at your submission repo:

   ```bash
   IMAGE_DETECTOR=ViT
   IMAGE_DETECTOR_CONFIG=ViT_roadwork.yaml
   IMAGE_DETECTOR_DEVICE=cuda   # or cpu if no GPU
   MODEL_URL=https://huggingface.co/your-username/roadwork-miner
   NETUID=72
   SUBTENSOR_NETWORK=finney
   SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
   WALLET_NAME=your_wallet
   WALLET_HOTKEY=your_hotkey
   MINER_AXON_PORT=8091
   ```

3. **Register and run**  
   Register your hotkey on mainnet (netuid 72), ensure your axon port is open, then start the miner (see [Registration](#registration) and [Mining](#mining) below). The miner will load the **default ViT weights** from `ViT_roadwork.yaml` (natix-network-org/roadwork); no training or custom code is required.

You will earn rewards based on how well the default model performs. To earn more, you would need to train or use a better model (see [Deploy Your Model](#deploy-your-model) and [Training](#training)).

## Registration

To reduce the risk of deregistration due to technical issues or poor model performance, we recommend the following:

1. Test your miner on testnet before mining on mainnet.
2. Before registering your hotkey on mainnet, verify that your port is open. If you cannot open a port at home (e.g. corporate network), run the miner on a [VPS](MiningOnVPS.md) instead.

```bash
curl your_ip:your_port
```

#### Mainnet

```bash
btcli s register --netuid 72 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network finney
```

#### Testnet

```bash
btcli s register --netuid 323 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network test
```


## Mining
Run `./setup_env.sh` to generate a `miner.env` file with default configuration.

Make sure to update your `miner.env` file with your wallet name, hotkey, miner port, and model configuration.
```
# following are initial values
IMAGE_DETECTOR=ViT
IMAGE_DETECTOR_CONFIG=ViT_roadwork.yaml
VIDEO_DETECTOR=TALL
VIDEO_DETECTOR_CONFIG=tall.yaml

# Device Settings
IMAGE_DETECTOR_DEVICE=cpu # Options: cpu, cuda
VIDEO_DETECTOR_DEVICE=cpu

NETUID=323                           # 323 for testnet, 72 for mainnet
SUBTENSOR_NETWORK=test               # Networks: finney, test, local
SUBTENSOR_CHAIN_ENDPOINT=wss://test.finney.opentensor.ai:443
                                     # Endpoints:
                                     # - wss://entrypoint-finney.opentensor.ai:443
                                     # - wss://test.finney.opentensor.ai:443/
                                     

# Wallet Configuration
WALLET_NAME=
WALLET_HOTKEY=

# Miner Settings
MINER_AXON_PORT=8091
BLACKLIST_FORCE_VALIDATOR_PERMIT=True # Force validator permit for blacklisting

# Miner details
MODEL_URL= # The URL to your Hugging-face repository
```

Then, start your miner with:

```bash
chmod +x ./start_miner.sh
./start_miner.sh
```

This will launch `run_neuron.py` using Poetry.

You can also optionally run a cache updater service to improve image caching performance:

```bash
chmod +x ./start_cache_updater.sh
./start_cache_updater.sh
```

This invokes `natix/validator/scripts/run_cache_updater.py` in the background.

## Submitted a Model

Miners must publish their model to Hugging Face and include a `model_card.json` with the following format:

```json
{
  "model_name": "<ARBITRARY_MODEL_NAME>",
  "description": "<DESCRIPTION>",
  "version": <VERSION NUMBER IN X.Y.Z format>,
  "submitted_by": "<WALLET_HOTKEY_ADDRESS>",
  "submission_time": <TIMESTAMP>
}
```

Update the `MODEL_URL` variable in your `miner.env` to reflect your Hugging Face repository.

## Deploy Your Model

Update your `miner.env` file to use your trained detector class and configuration.

**Step-by-step:** See **[ConfigureMiner.md](ConfigureMiner.md)** for Inception Fusion (paths, YAML fields, env vars, and checklist).

* Detector types are defined in `base_miner/registry.py` (e.g. ViT, YOLOClassifier, InceptionFusion). Config files live in `base_miner/detectors/configs/`.
* **Weights:** ViT loads from Hugging Face via the `hf_repo` field in its config (no local weights path). YOLOClassifier and InceptionFusion use paths set in the config YAML (`weights_path`, `yolo_weights_path`, `googlenet_fusion_path`) — you can place weights anywhere and point the config at them. See [ConfigureMiner.md](ConfigureMiner.md) for Inception Fusion paths.

## Training

To improve beyond the baseline model, experiment with new datasets, architectures, or hyperparameters. The easiest way is to **fine-tune a pre-trained model** (e.g. the default `natix-network-org/roadwork` or `google/vit-base-patch16-224`) using the included script; see [Training.md](Training.md) for a step-by-step guide and pre-trained options.

## Evaluating your model vs the default

Before deploying your own model, you can evaluate it locally using the **same metrics** as the subnet (accuracy, MCC, reward) and compare to the default ViT. See [Evaluation.md](Evaluation.md) for the evaluation script and usage.

## TensorBoard

Start TensorBoard to view training metrics:

```bash
tensorboard --logdir=./base_miner/checkpoints/<experiment_name>
```

For remote machines:

```bash
ssh -L 7007:localhost:6006 your_username@your_ip
```

Then on the remote machine:

```bash
tensorboard --logdir=./base_miner/checkpoints/<experiment_name> --host 0.0.0.0 --port 6006
```

View it locally at `http://localhost:7007`

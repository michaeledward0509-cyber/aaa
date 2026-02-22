# Mainnet Readiness Review (Miner + Detector)

This checklist covers your current setup and what to change/verify before switching to mainnet (netuid 72).

---

## 1. miner.env

| Variable | Current | Mainnet | Notes |
|----------|---------|---------|--------|
| **NETUID** | 323 | **72** | Change for mainnet. |
| **SUBTENSOR_NETWORK** | test | **finney** | |
| **SUBTENSOR_CHAIN_ENDPOINT** | wss://test.finney.opentensor.ai:443 | **wss://entrypoint-finney.opentensor.ai:443** | |
| **BLACKLIST_FORCE_VALIDATOR_PERMIT** | False | **True** | Docs recommend True on mainnet (only accept validator traffic). |
| **IMAGE_DETECTOR_DEVICE** | cpu | **cuda** (optional) | You have a GPU (GTX 1660 SUPER); cuda will be faster. |
| **MINER_AXON_PORT** | 32707 | 32707 | Keep; ensure this port is open in firewall and cloud security group. |
| **WALLET_NAME** / **WALLET_HOTKEY** | default | your choice | Use the wallet you register on mainnet. |
| **MODEL_URL** | https://huggingface.co/MichaelEdward/roadwork-miner | same | Verify below. |

**MODEL_URL and Hugging Face**

- Repo must exist and be public.
- It must contain **model_card.json** with:
  - `"submitted_by": "<YOUR_HOTKEY_SS58>"` (same as the hotkey in miner.env).
  - `"submission_time": <unix_timestamp>`.
- If you use a different HF repo for your InceptionFusion submission, set `MODEL_URL` to that repo.

---

## 2. Detector config: my_inception_fusion.yaml

| Field | Current | Status |
|-------|---------|--------|
| **vit_config** | ViT_roadwork.yaml | OK – uses default subnet ViT. |
| **yolo_weights_path** | /root/Workspace/aaa/workspace/best.pt | OK – absolute path; ensure this file exists on the VPS. |
| **googlenet_fusion_path** | /root/Workspace/aaa/inception_fusion_keras | OK – absolute path. |
| **use_keras** | true | OK – matches Keras-trained fusion. |
| **roadwork_class_index** | 1 | OK – YOLO class 1 = roadwork. |

**Files that must exist on the miner machine**

- `best.pt` (YOLO weights).
- `inception_fusion_keras/` directory with:
  - `fusion_head.keras` (or `fusion_head/`)
  - `efficientnetv2_branch.keras` or `googlenet_branch.keras` (or corresponding dirs).

If any of these are missing, the miner will fail at startup with `FileNotFoundError`. Your logs showed "Loaded image detection model: InceptionFusion", so on the machine you ran from they exist; **on mainnet, use the same VPS or copy these files to the mainnet miner host.**

---

## 3. Detector code: InceptionFusionDetector

**Output format**

- The detector returns a **float probability in [0, 1]** (P(roadwork)).
- The validator expects:
  - **Float in [0, 1]** → used for reward (and rounded to 0/1 for metrics).
  - **-1** → invalid response, zero reward.
- Your `__call__` returns `float(probs[0, 1])` (roadwork probability), which is correct.

**Branches**

- ViT: from `natix-network-org/roadwork` (loaded at runtime).
- YOLO: from `best.pt` (config path).
- EfficientNetV2 + fusion: from `inception_fusion_keras` (Keras).  
All three are used; no change needed for mainnet.

**Errors**

- If inference throws (e.g. bad image, OOM), the miner’s `forward_image` should set `synapse.prediction = -1` so the validator assigns zero reward instead of an undefined value. (See fix below.)

---

## 4. Miner forward_image – exception handling

**Current:** On exception, `synapse.prediction` is not set, so the response may be invalid or ambiguous.

**Recommendation:** In the `except` block, set `synapse.prediction = -1` so invalid responses are explicit and get zero reward.

---

## 5. Pre-mainnet checklist

- [ ] **Registration:** Register hotkey on mainnet:  
  `btcli s register --netuid 72 --wallet.name <name> --wallet.hotkey <hotkey> --subtensor.network finney`  
  (Requires TAO.)
- [ ] **miner.env:** Set NETUID=72, SUBTENSOR_NETWORK=finney, SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443, BLACKLIST_FORCE_VALIDATOR_PERMIT=True.
- [ ] **MODEL_URL:** Point to your HF repo; confirm model_card.json has your hotkey.
- [ ] **Port 32707:** Open on VPS firewall and cloud security group.
- [ ] **Paths:** Confirm best.pt and inception_fusion_keras/ (with fusion_head.keras and efficientnetv2_branch.keras or googlenet_branch.keras) exist on the mainnet miner host.
- [ ] **Optional:** Set IMAGE_DETECTOR_DEVICE=cuda if the miner runs on a GPU machine.
- [ ] After starting the miner, watch for **"Validator request received (forward_image)"** in logs; if you see it, validators are querying you. Then incentive should update over time.

---

## 6. Summary

| Area | Status | Action |
|------|--------|--------|
| miner.env | Testnet values | Switch to mainnet (72, finney, endpoint); set BLACKLIST_FORCE_VALIDATOR_PERMIT=True. |
| Detector config | Paths absolute, use_keras true | Ensure same paths and Keras files on mainnet host. |
| Detector output | Float [0,1] | Matches protocol; no change. |
| Exception handling | prediction not set on error | Set prediction = -1 on exception (recommended). |
| MODEL_URL / model_card | Set | Verify HF repo and hotkey in model_card.json. |

Once the above is done and the miner runs on mainnet, you should be eligible for rewards when validators query you and set weights.

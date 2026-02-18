"""
Inception-style fusion detector: combines ViT (default), YOLO, and EfficientNetV2.

Uses the DL "Inception" idea: parallel branches (three different architectures)
whose outputs are concatenated and passed through a small fusion head to produce
the final roadwork probability. Only subnet training data is used to train the
fusion layer (and optionally the EfficientNetV2 head).

Requires:
  - ViT: natix-network-org/roadwork (or config)
  - YOLO: a YOLOv11-cls .pt trained for 2 classes (see TrainingYOLO.md)
  - EfficientNetV2: Keras pretrained + 2-class head (trained by train_inception_fusion_keras.py)
  - Fusion weights: learned layer (3 -> 2) trained by train_inception_fusion_keras.py
"""

import gc
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from base_miner.detectors import FeatureDetector
from base_miner.registry import DETECTOR_REGISTRY

warnings.filterwarnings("ignore", category=UserWarning, module="ultralytics")
warnings.filterwarnings("ignore", category=FutureWarning)


class FusionHead(nn.Module):
    """Inception-style fusion: concat [p_vit, p_yolo, p_googlenet] -> logits (2 classes)."""

    def __init__(self, num_branches: int = 3, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(num_branches, num_classes)

    def forward(self, branch_probs: torch.Tensor) -> torch.Tensor:
        # branch_probs: (B, 3) in [0,1]
        return self.fc(branch_probs)


class FusionHeadMLP(nn.Module):
    """Deeper fusion: 3 -> hidden -> 2 with ReLU and dropout. Can learn non-linear combinations."""

    def __init__(self, num_branches: int = 3, num_classes: int = 2, hidden: int = 16, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(num_branches, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, branch_probs: torch.Tensor) -> torch.Tensor:
        x = self.fc1(branch_probs)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


@DETECTOR_REGISTRY.register_module(module_name="InceptionFusion")
class InceptionFusionDetector(FeatureDetector):
    """
    Combines ViT (default subnet model), YOLO classifier, and GoogLeNet in an
    Inception-style parallel branch design. Each branch produces a roadwork
    probability; a learned fusion layer combines them. Returns P(roadwork).
    """

    def __init__(
        self,
        model_name: str = "InceptionFusion",
        config_name: str = "inception_fusion_roadwork.yaml",
        device: str = "cpu",
    ):
        self.vit_config = None
        self.yolo_weights_path = None
        self.googlenet_fusion_path = None
        self.roadwork_class_index = 1
        super().__init__(model_name, config_name, device)

    def load_model_config(self):
        if getattr(self, "hf_repo", None) is not None:
            super().load_model_config()
        else:
            self.config = getattr(self, "config", {}) or {}

    def load_model(self):
        import yaml
        from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

        from base_miner.detectors.configs.constants import CONFIGS_DIR

        device_str = "cuda:0" if self.device.type == "cuda" else "cpu"

        # 1) ViT branch (default subnet model)
        vit_config = getattr(self, "vit_config", "ViT_roadwork.yaml")
        config_path = Path(vit_config) if Path(vit_config).exists() else Path(CONFIGS_DIR) / vit_config
        with open(config_path, "r") as f:
            vit_cfg = yaml.safe_load(f)
        hf_repo = vit_cfg.get("hf_repo", "natix-network-org/roadwork")
        self._vit_pipeline = pipeline(
            "image-classification",
            model=AutoModelForImageClassification.from_pretrained(hf_repo),
            feature_extractor=AutoImageProcessor.from_pretrained(hf_repo, use_fast=True),
            device=-1 if device_str == "cpu" else 0,
        )

        # 2) YOLO branch
        from ultralytics import YOLO

        yolo_path = getattr(self, "yolo_weights_path", None)
        if not yolo_path or not Path(yolo_path).exists():
            raise FileNotFoundError(
                f"YOLO weights_path not set or missing: {yolo_path}. "
                "Train a YOLOv11-cls model (see TrainingYOLO.md) and set yolo_weights_path in config."
            )
        self._yolo = YOLO(yolo_path)
        self._roadwork_idx = int(getattr(self, "roadwork_class_index", 1))

        # 3) EfficientNetV2 branch + fusion head (Keras)
        fusion_path = getattr(self, "googlenet_fusion_path", None)  # Keep config key name for backward compatibility
        use_keras = getattr(self, "use_keras", False)
        if not fusion_path or not Path(fusion_path).exists():
            raise FileNotFoundError(
                f"googlenet_fusion_path not set or missing: {fusion_path}. "
                "Run train_inception_fusion_keras.py and set path in config."
            )
        path = Path(fusion_path)
        if use_keras or (path.is_dir() and (path / "fusion_head.keras").exists()):
            self._use_keras = True
            self._efficientnet_keras, self._fusion_keras = self._load_keras_models(path)
            self._googlenet = None
            self._fusion_head = None
            self._googlenet_transform = None
        else:
            raise ValueError(
                "EfficientNetV2 requires Keras models. Set use_keras: true in config and train with train_inception_fusion_keras.py"
            )

    def _load_keras_models(self, path: Path):
        """Load EfficientNetV2 branch and fusion head from Keras .keras files."""
        from tensorflow import keras

        path = Path(path)
        # Try efficientnetv2_branch.keras first (new), fall back to googlenet_branch.keras (old) for compatibility
        efficientnet_path = path / "efficientnetv2_branch.keras" if path.is_dir() else path.parent / "efficientnetv2_branch.keras"
        if not efficientnet_path.exists():
            efficientnet_path = path / "efficientnetv2_branch"
        if not efficientnet_path.exists():
            # Fallback to old name for backward compatibility
            efficientnet_path = path / "googlenet_branch.keras" if path.is_dir() else path.parent / "googlenet_branch.keras"
        if not efficientnet_path.exists():
            efficientnet_path = path / "googlenet_branch"
        
        fusion_path_k = path / "fusion_head.keras" if path.is_dir() else path.parent / "fusion_head.keras"
        if not fusion_path_k.exists():
            fusion_path_k = path / "fusion_head"
        if not Path(efficientnet_path).exists():
            raise FileNotFoundError(f"Keras EfficientNetV2 branch not found at {efficientnet_path} (also checked googlenet_branch.keras for compatibility)")
        if not Path(fusion_path_k).exists():
            raise FileNotFoundError(f"Keras fusion head not found at {fusion_path_k}")
        efficientnet_model = keras.models.load_model(efficientnet_path)
        fusion_model = keras.models.load_model(fusion_path_k)
        return efficientnet_model, fusion_model

    def _load_googlenet_and_fusion(self, path: str):
        from torchvision.models import googlenet, GoogLeNet_Weights

        googlenet_model = googlenet(weights=GoogLeNet_Weights.DEFAULT)
        # Replace final FC for 2 classes (roadwork / no_roadwork)
        in_features = googlenet_model.fc.in_features
        googlenet_model.fc = nn.Linear(in_features, 2)
        googlenet_model.aux_logits = False

        state = torch.load(path, map_location="cpu", weights_only=True)
        fusion_type = state.get("fusion_head_type", "linear")
        if fusion_type == "mlp":
            fusion_head = FusionHeadMLP(num_branches=3, num_classes=2, hidden=16, dropout=0.1)
        else:
            fusion_head = FusionHead(num_branches=3, num_classes=2)
        if "googlenet_fc" in state:
            googlenet_model.fc.load_state_dict(state["googlenet_fc"])
        if "fusion_head" in state:
            fusion_head.load_state_dict(state["fusion_head"])
        return googlenet_model, fusion_head

    def _get_vit_prob(self, image: Image.Image) -> float:
        out = self._vit_pipeline(image)
        for item in out:
            if item.get("label") == "Roadwork":
                return float(item["score"])
        return 0.0

    def _get_yolo_prob(self, image: Image.Image) -> float:
        if image.mode != "RGB":
            image = image.convert("RGB")
        device = "cuda:0" if self.device.type == "cuda" else "cpu"
        results = self._yolo.predict(source=image, verbose=False, device=device)
        if not results:
            return 0.0
        r = results[0]
        if not hasattr(r, "probs") or r.probs is None:
            return 0.0
        probs = r.probs.data
        if hasattr(probs, "cpu"):
            probs = probs.cpu().numpy()
        else:
            probs = np.asarray(probs)
        if probs.ndim > 1:
            probs = probs.ravel()
        idx = min(self._roadwork_idx, len(probs) - 1)
        return float(probs[idx])

    def _get_googlenet_prob(self, image: Image.Image) -> float:
        """Get EfficientNetV2 probability (kept method name for backward compatibility)."""
        if self._use_keras:
            return self._get_efficientnet_prob_keras(image)
        raise ValueError("EfficientNetV2 requires Keras models. Set use_keras: true in config.")

    def _get_efficientnet_prob_keras(self, image: Image.Image) -> float:
        """Get EfficientNetV2 probability from Keras model."""
        import numpy as np

        if image.mode != "RGB":
            image = image.convert("RGB")
        img = image.resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        # EfficientNetV2 uses same ImageNet normalization as InceptionV3
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        X = np.expand_dims(arr, axis=0)
        probs = self._efficientnet_keras(X, training=False)
        return float(probs[0, 1])

    def __call__(self, image: Image.Image) -> float:
        p_vit = self._get_vit_prob(image)
        p_yolo = self._get_yolo_prob(image)
        p_googlenet = self._get_googlenet_prob(image)
        if self._use_keras:
            import numpy as np
            branch_probs = np.array([[p_vit, p_yolo, p_googlenet]], dtype=np.float32)
            probs = self._fusion_keras(branch_probs, training=False)
            return float(probs[0, 1])
        branch_probs = torch.tensor(
            [[p_vit, p_yolo, p_googlenet]],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            logits = self._fusion_head(branch_probs)
        probs = torch.softmax(logits, dim=1)
        return float(probs[0, 1].cpu().numpy())  # roadwork = index 1

    def free_memory(self):
        if hasattr(self, "_vit_pipeline") and self._vit_pipeline is not None:
            self._vit_pipeline = None
        if hasattr(self, "_yolo"):
            self._yolo = None
        if hasattr(self, "_googlenet") and self._googlenet is not None:
            self._googlenet.cpu()
            del self._googlenet
            self._googlenet = None
        if hasattr(self, "_fusion_head") and self._fusion_head is not None:
            self._fusion_head.cpu()
            del self._fusion_head
            self._fusion_head = None
        if getattr(self, "_use_keras", False):
            self._efficientnet_keras = None
            self._fusion_keras = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

"""
YOLO-based image classifier for roadwork (binary). Uses Ultralytics YOLOv11-cls.

The subnet task is image-level classification (roadwork vs no roadwork), not object
detection. Use YOLO in *classification* mode: train with `yolo classify`, then load
the .pt weights here. This detector returns the probability of class "roadwork" (0–1).
"""

import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from base_miner.detectors import FeatureDetector
from base_miner.registry import DETECTOR_REGISTRY

warnings.filterwarnings("ignore", category=UserWarning, module="ultralytics")


@DETECTOR_REGISTRY.register_module(module_name="YOLOClassifier")
class YOLOClassifierDetector(FeatureDetector):
    """
    Binary roadwork classifier using Ultralytics YOLOv11-cls (or similar).
    Expects a model trained for 2 classes: index 0 = no_roadwork, index 1 = roadwork.
    Returns probability of roadwork (class 1).
    """

    def __init__(
        self,
        model_name: str = "YOLOClassifier",
        config_name: str = "yolo_roadwork.yaml",
        device: str = "cpu",
    ):
        # YAML may only have weights_path; base class expects config_name to exist
        self.weights_path = None
        self.roadwork_class_index = 1
        super().__init__(model_name, config_name, device)

    def load_model_config(self):
        """Skip HF config load if no hf_repo (YOLO uses local weights only)."""
        if getattr(self, "hf_repo", None) is not None:
            super().load_model_config()
        else:
            self.config = getattr(self, "config", {}) or {}

    def load_model(self):
        from ultralytics import YOLO

        path = getattr(self, "weights_path", None)
        if not path or not Path(path).exists():
            raise FileNotFoundError(
                f"YOLO weights_path not set or missing: {path}. "
                "Set weights_path in your detector YAML to a .pt file from 'yolo classify train'."
            )
        self.model = YOLO(path)
        self._roadwork_idx = int(getattr(self, "roadwork_class_index", 1))

    def __call__(self, image: Image.Image) -> float:
        if image.mode != "RGB":
            image = image.convert("RGB")
        device = "cuda:0" if "cuda" in str(self.device) else "cpu"
        results = self.model.predict(source=image, verbose=False, device=device)
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

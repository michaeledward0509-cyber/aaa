#!/usr/bin/env python3
"""
Measure inference latency of your detector (same config as miner).

Reports mean, std, median, p95, p99 in milliseconds. Validator timeout is 9s;
inference should stay well below that.

Usage:
  # Use your miner config (InceptionFusion + my_inception_fusion.yaml):
  poetry run python scripts/measure_detector_latency.py

  # Optional: warmup runs, number of timed runs, device
  poetry run python scripts/measure_detector_latency.py --warmup 5 --runs 50 --device cuda

  # With a real image (otherwise uses a dummy 224x224 RGB):
  poetry run python scripts/measure_detector_latency.py --image path/to/image.jpg
"""
import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load env so we get IMAGE_DETECTOR, IMAGE_DETECTOR_CONFIG, IMAGE_DETECTOR_DEVICE
def load_miner_env():
    env_path = REPO_ROOT / "miner.env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

load_miner_env()

import base_miner.detectors  # noqa: E402
from base_miner.registry import DETECTOR_REGISTRY
from natix.utils.config import get_device
from PIL import Image
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Measure detector inference latency")
    parser.add_argument("--detector", type=str, default=None, help="Detector name (default: from miner.env IMAGE_DETECTOR)")
    parser.add_argument("--config", type=str, default=None, help="Config filename (default: from miner.env IMAGE_DETECTOR_CONFIG)")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda (default: from miner.env IMAGE_DETECTOR_DEVICE)")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup inference runs (excluded from stats)")
    parser.add_argument("--runs", type=int, default=50, help="Number of timed runs")
    parser.add_argument("--image", type=str, default=None, help="Path to one image (default: dummy 224x224 RGB)")
    args = parser.parse_args()

    detector_name = args.detector or os.environ.get("IMAGE_DETECTOR", "InceptionFusion")
    config_name = args.config or os.environ.get("IMAGE_DETECTOR_CONFIG", "my_inception_fusion.yaml")
    device = (args.device or os.environ.get("IMAGE_DETECTOR_DEVICE", "cpu")).lower()
    if device == "auto":
        device = "cuda" if get_device() != "cpu" else "cpu"

    print(f"Loading detector: {detector_name} / {config_name} on {device}")
    detector = DETECTOR_REGISTRY[detector_name](config_name=config_name, device=device)
    print("Detector loaded.")

    if args.image and Path(args.image).exists():
        img = Image.open(args.image).convert("RGB")
        print(f"Using image: {args.image} ({img.size})")
    else:
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8), mode="RGB")
        print("Using dummy 224x224 RGB image")

    # Warmup
    for _ in range(args.warmup):
        detector(img)

    # Timed runs (in seconds)
    times_sec = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        detector(img)
        times_sec.append(time.perf_counter() - t0)

    times_ms = [t * 1000 for t in times_sec]
    times_ms.sort()
    n = len(times_ms)
    mean_ms = sum(times_ms) / n
    variance = sum((x - mean_ms) ** 2 for x in times_ms) / n
    std_ms = variance ** 0.5
    p50 = times_ms[int(n * 0.50)]
    p95 = times_ms[int(n * 0.95)] if n >= 20 else times_ms[-1]
    p99 = times_ms[int(n * 0.99)] if n >= 50 else times_ms[-1]

    print()
    print("Latency (inference only, single image)")
    print("  Mean:   {:.2f} ms".format(mean_ms))
    print("  Std:    {:.2f} ms".format(std_ms))
    print("  Median (p50): {:.2f} ms".format(p50))
    print("  p95:    {:.2f} ms".format(p95))
    print("  p99:    {:.2f} ms".format(p99))
    print()
    if p99 > 9000:
        print("  Warning: p99 > 9000 ms (validator timeout 9s). Risk of timeouts.")
    else:
        print("  Validator timeout: 9s — you are within limit.")


if __name__ == "__main__":
    main()

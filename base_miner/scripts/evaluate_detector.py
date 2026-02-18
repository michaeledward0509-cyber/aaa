#!/usr/bin/env python3
"""
Evaluate a roadwork detector (and optionally compare with the default) on a labeled dataset.

Uses the same metrics as the subnet: accuracy, MCC, and the reward formula
  reward = (0.5 * MCC + 0.5 * accuracy) ** REWARD_CURVE_EXPONENT

Usage:
  # Evaluate default model (ViT_roadwork) on an imagefolder dataset:
  poetry run python base_miner/scripts/evaluate_detector.py --dataset_path imagefolder:/path/to/data

  # Compare your model vs default (your model uses a custom config pointing to your weights):
  poetry run python base_miner/scripts/evaluate_detector.py \\
    --dataset_path imagefolder:/path/to/data \\
    --my_detector ViT --my_config my_roadwork.yaml

  # Evaluate only your model:
  poetry run python base_miner/scripts/evaluate_detector.py \\
    --dataset_path imagefolder:/path/to/data \\
    --no_default --my_detector ViT --my_config my_roadwork.yaml

Dataset: use a path that yields (image, label). Labels must be 0 (no roadwork) or 1 (roadwork).
  - imagefolder:path → folder with train/roadwork/ and train/no_roadwork/ (or test/...)
  - Or a Hugging Face dataset with "image" and "label" columns.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import accuracy_score, matthews_corrcoef

# Add repo root so base_miner and natix are importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import base_miner.detectors  # noqa: F401 - register all detectors (ViT, YOLOClassifier, InceptionFusion, etc.)
from base_miner.registry import DETECTOR_REGISTRY

REWARD_CURVE_EXPONENT = 3.0


def load_labeled_dataset(dataset_path: str, split: str = "train", max_samples: int = None):
    """
    Load a dataset that has (image, label). Labels 0 = no roadwork, 1 = roadwork.

    Args:
        dataset_path: "imagefolder:/path/to/dir" or "org/dataset_name"
        split: "train", "test", or "validation"
        max_samples: cap number of samples (for quick runs)

    Returns:
        list of (PIL.Image, int label)
    """
    if dataset_path.startswith("imagefolder:"):
        _, data_dir = dataset_path.split(":", 1)
        full = load_dataset("imagefolder", data_dir=data_dir)
        if isinstance(full, dict):
            if split not in full:
                split = "train" if "train" in full else list(full.keys())[0]
            dataset = full[split]
        else:
            dataset = full
    else:
        dataset = load_dataset(dataset_path, split=split)

    # Accept "label" or "labels"
    if "label" not in dataset.column_names and "labels" in dataset.column_names:
        label_col = "labels"
    else:
        label_col = "label"

    if label_col not in dataset.column_names:
        raise ValueError(
            f"Dataset must have a 'label' (or 'labels') column. Columns: {dataset.column_names}. "
            "Use an imagefolder with train/roadwork/ and train/no_roadwork/."
        )

    pairs = []
    for i in range(len(dataset)):
        row = dataset[i]
        img = row["image"]
        if not isinstance(img, Image.Image):
            from io import BytesIO
            img = Image.open(BytesIO(img)).convert("RGB")
        else:
            img = img.convert("RGB")
        label = int(row[label_col])
        if label not in (0, 1):
            label = 1 if label > 0 else 0
        pairs.append((img, label))
        if max_samples and len(pairs) >= max_samples:
            break
    return pairs


def compute_metrics(labels: np.ndarray, pred_probs: np.ndarray):
    """Same logic as MinerPerformanceTracker + reward formula."""
    pred_probs = np.array(pred_probs, dtype=float)
    valid = (pred_probs >= 0.0) & (pred_probs <= 1.0)
    pred_probs_clean = np.where(valid, pred_probs, 0.0)
    predictions = np.round(pred_probs_clean).astype(int)

    accuracy = accuracy_score(labels, predictions)
    if len(np.unique(labels)) > 1 and len(np.unique(predictions)) > 1:
        mcc = float(matthews_corrcoef(labels, predictions))
        mcc = max(0.0, mcc)
    else:
        mcc = 0.0

    # Subnet reward formula (validator/reward.py)
    reward_raw = 0.5 * mcc + 0.5 * accuracy
    reward = reward_raw ** REWARD_CURVE_EXPONENT
    return {
        "accuracy": accuracy,
        "mcc": mcc,
        "reward_raw": reward_raw,
        "reward": reward,
        "n_valid": int(valid.sum()),
        "n_total": len(labels),
    }


def run_evaluation(pairs, detector, device: str = "cpu", name: str = "model"):
    """Run detector on all (image, label) pairs and return metrics."""
    labels_list = []
    preds_list = []
    for img, label in pairs:
        try:
            prob = detector(img)
            if prob == -1 or not (0 <= prob <= 1):
                prob = 0.0
        except Exception as e:
            print(f"  Warning: inference failed for {name}: {e}")
            prob = 0.0
        labels_list.append(label)
        preds_list.append(prob)
    labels = np.array(labels_list)
    preds = np.array(preds_list)
    return compute_metrics(labels, preds)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate roadwork detector(s) on a labeled dataset (subnet metrics)."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path: imagefolder:/path/to/data or HuggingFace org/dataset. Must have (image, label).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "validation"],
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap number of samples (for quick runs).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for inference.",
    )
    parser.add_argument(
        "--no_default",
        action="store_true",
        help="Do not evaluate the default model (only --my_detector).",
    )
    parser.add_argument(
        "--default_detector",
        type=str,
        default="ViT",
        help="Default detector class name (default: ViT).",
    )
    parser.add_argument(
        "--default_config",
        type=str,
        default="ViT_roadwork.yaml",
        help="Default detector config (default: ViT_roadwork.yaml).",
    )
    parser.add_argument(
        "--my_detector",
        type=str,
        default=None,
        help="Your detector class name (e.g. ViT) for comparison.",
    )
    parser.add_argument(
        "--my_config",
        type=str,
        default=None,
        help="Your detector config YAML (e.g. my_roadwork.yaml with your hf_repo/weights).",
    )
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_path} (split={args.split})")
    try:
        pairs = load_labeled_dataset(args.dataset_path, split=args.split, max_samples=args.max_samples)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)
    print(f"Loaded {len(pairs)} samples.")

    results = {}

    if not args.no_default:
        print("Loading default detector...")
        default_detector = DETECTOR_REGISTRY[args.default_detector](
            config_name=args.default_config,
            device=args.device,
        )
        results["default"] = run_evaluation(pairs, default_detector, args.device, "default")
        print("Default model done.")

    if args.my_detector and args.my_config:
        print("Loading your detector...")
        my_detector = DETECTOR_REGISTRY[args.my_detector](
            config_name=args.my_config,
            device=args.device,
        )
        results["yours"] = run_evaluation(pairs, my_detector, args.device, "yours")
        print("Your model done.")

    if not results:
        print("No model to evaluate. Use default (don't pass --no_default) or --my_detector + --my_config.")
        sys.exit(1)

    # Report
    print("\n" + "=" * 60)
    print("Evaluation (subnet-style metrics)")
    print("=" * 60)
    for name, m in results.items():
        print(f"\n{name}:")
        print(f"  accuracy   = {m['accuracy']:.4f}")
        print(f"  mcc        = {m['mcc']:.4f}")
        print(f"  reward_raw = {m['reward_raw']:.4f}  (0.5*mcc + 0.5*acc)")
        print(f"  reward     = {m['reward']:.4f}  (reward_raw^{REWARD_CURVE_EXPONENT})")
        print(f"  n_valid    = {m['n_valid']}/{m['n_total']}")

    if "default" in results and "yours" in results:
        d, y = results["default"], results["yours"]
        print("\n--- Comparison (yours vs default) ---")
        print(f"  accuracy: {y['accuracy']:.4f} vs {d['accuracy']:.4f}  ({'better' if y['accuracy'] > d['accuracy'] else 'worse'})")
        print(f"  mcc:      {y['mcc']:.4f} vs {d['mcc']:.4f}  ({'better' if y['mcc'] > d['mcc'] else 'worse'})")
        print(f"  reward:   {y['reward']:.4f} vs {d['reward']:.4f}  ({'better' if y['reward'] > d['reward'] else 'worse'})")
    print()


if __name__ == "__main__":
    main()

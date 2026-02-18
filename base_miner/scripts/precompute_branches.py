#!/usr/bin/env python3
"""
Precompute ViT and YOLO branch probabilities for the whole dataset and save to a cache file.

Uses the **original** subnet dataset images (no augmentation here). Data augmentation
for the subnet dataset is applied during **fusion training** (train_inception_fusion*.py)
with --augment_strength strong by default.

Run this once (on free Colab/Kaggle GPU or overnight on CPU). Then train the fusion
with --precomputed_branches so ViT and YOLO are never loaded during fusion training—
saving memory and time on a laptop CPU.

Usage:
  # On Colab/Kaggle (free GPU) or laptop (slow but works):
  poetry run python base_miner/scripts/precompute_branches.py \\
    --dataset_path imagefolder:/path/to/roadwork_data \\
    --yolo_weights /path/to/yolo/weights/best.pt \\
    --output_cache ./branch_cache.npz \\
    --split train \\
    --batch_size 8

Then train fusion on laptop CPU with small batch (no ViT/YOLO loaded):
  poetry run python base_miner/scripts/train_inception_fusion_keras.py \\
    --dataset_path imagefolder:/path/to/roadwork_data \\
    --precomputed_branches ./branch_cache.npz \\
    --output_dir ./inception_fusion_keras \\
    --epochs 10 --batch_size 8
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

LABEL2ID = {"no_roadwork": 0, "roadwork": 1}


def load_labeled_dataset(dataset_path: str, split: str = "train"):
    if dataset_path.startswith("imagefolder:"):
        _, data_dir = dataset_path.split(":", 1)
        dataset = load_dataset("imagefolder", data_dir=data_dir)
        if isinstance(dataset, dict):
            split = split if split in dataset else "train"
            ds = dataset[split]
        else:
            ds = dataset
    else:
        ds = load_dataset(dataset_path, split=split)

    if "label" not in ds.column_names and "labels" in ds.column_names:
        ds = ds.rename_column("labels", "label")
    if "label" not in ds.column_names:
        raise ValueError(f"Dataset must have 'label'. Got: {ds.column_names}")

    def map_label(ex):
        l = ex["label"]
        if isinstance(l, str):
            l = l.lower().replace(" ", "_")
            ex["label"] = LABEL2ID.get(l, 1 if "road" in l or l == "1" else 0)
        elif l not in (0, 1):
            ex["label"] = 1 if int(l) > 0 else 0
        return ex

    return ds.map(map_label, num_proc=1)


def get_vit_probs(pipe, images, device_str):
    probs = []
    for img in images:
        out = pipe(img)
        if not isinstance(out, list):
            out = [out]
        p = 0.0
        for item in out:
            if item.get("label") == "Roadwork":
                p = item["score"]
                break
        probs.append(p)
    return np.array(probs, dtype=np.float32)


def get_yolo_probs(yolo_model, images, device_str, roadwork_idx=1):
    probs = []
    for img in images:
        r = yolo_model.predict(source=img, verbose=False, device=device_str)
        if not r or not hasattr(r[0], "probs") or r[0].probs is None:
            probs.append(0.0)
            continue
        p = r[0].probs.data
        if hasattr(p, "cpu"):
            p = p.cpu().numpy()
        else:
            p = np.asarray(p)
        if p.ndim > 1:
            p = p.ravel()
        idx = min(roadwork_idx, len(p) - 1)
        probs.append(float(p[idx]))
    return np.array(probs, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute ViT and YOLO probabilities for fusion training (run once, then use --precomputed_branches)."
    )
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="imagefolder:/path or HF dataset (same as you will use for fusion training)")
    parser.add_argument("--yolo_weights", type=str, required=True,
                        help="Path to YOLOv11-cls best.pt")
    parser.add_argument("--output_cache", type=str, default="./branch_cache.npz",
                        help="Output .npz file (p_vit, p_yolo, labels)")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference (smaller = less RAM on CPU)")
    parser.add_argument("--vit_hf_repo", type=str, default="natix-network-org/roadwork")
    args = parser.parse_args()

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_str}")

    print("Loading dataset...")
    ds = load_labeled_dataset(args.dataset_path, split=args.split)
    n = len(ds)
    print(f"Samples: {n}")

    print("Loading ViT...")
    vit_pipeline = pipeline(
        "image-classification",
        model=AutoModelForImageClassification.from_pretrained(args.vit_hf_repo),
        feature_extractor=AutoImageProcessor.from_pretrained(args.vit_hf_repo, use_fast=True),
        device=-1 if device_str == "cpu" else 0,
    )

    print("Loading YOLO...")
    from ultralytics import YOLO
    yolo_model = YOLO(args.yolo_weights)

    p_vit_list = []
    p_yolo_list = []
    labels_list = []

    for start in tqdm(range(0, n, args.batch_size), desc="Precomputing branches"):
        end = min(start + args.batch_size, n)
        images = [ds[int(i)]["image"] for i in range(start, end)]
        if images[0].mode != "RGB":
            images = [im.convert("RGB") for im in images]
        labels_list.extend([ds[int(i)]["label"] for i in range(start, end)])
        p_vit_list.append(get_vit_probs(vit_pipeline, images, device_str))
        p_yolo_list.append(get_yolo_probs(yolo_model, images, device_str))

    p_vit = np.concatenate(p_vit_list, axis=0)
    p_yolo = np.concatenate(p_yolo_list, axis=0)
    labels = np.array(labels_list, dtype=np.int32)

    assert len(p_vit) == n and len(p_yolo) == n and len(labels) == n
    out = Path(args.output_cache)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, p_vit=p_vit, p_yolo=p_yolo, labels=labels)
    print(f"Saved {n} samples to {out}")
    print("Next: train fusion with --precomputed_branches", str(out), "(no ViT/YOLO loaded; CPU-friendly).")


if __name__ == "__main__":
    main()

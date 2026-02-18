#!/usr/bin/env python3
"""
Train the Inception-style fusion model (ViT + YOLO + GoogLeNet) using subnet data only.

Uses the same data as the subnet: natix-network-org/roadwork or an imagefolder
with train/roadwork/ and train/no_roadwork/. ViT and YOLO are frozen; only the
GoogLeNet 2-class head and the fusion layer are trained.

Prerequisites:
  1. ViT: uses default natix-network-org/roadwork (no extra training).
  2. YOLO: you must train a YOLOv11-cls model first (see docs/TrainingYOLO.md)
     and set --yolo_weights to the path of your best.pt.
  3. GoogLeNet + fusion: trained by this script.

Example (subnet data only):
  poetry run python base_miner/scripts/train_inception_fusion.py \\
    --dataset_path imagefolder:/path/to/roadwork_data \\
    --yolo_weights /path/to/yolo/weights/best.pt \\
    --output_dir ./inception_fusion_weights \\
    --epochs 10 --batch_size 16
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from base_miner.detectors.inception_fusion_detector import FusionHead, FusionHeadMLP

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
    """Run ViT on each image; returns (N,) tensor of P(roadwork)."""
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
    return torch.tensor(probs, dtype=torch.float32, device=device_str)


def get_yolo_probs(yolo_model, images, device_str, roadwork_idx=1):
    """Run YOLO on each image; returns (N,) tensor of P(roadwork)."""
    import numpy as np

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
    return torch.tensor(probs, dtype=torch.float32, device=device_str)


def main():
    parser = argparse.ArgumentParser(description="Train Inception fusion (ViT+YOLO+GoogLeNet) on subnet data.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="imagefolder:/path/to/data or HuggingFace org/dataset (subnet data only)")
    parser.add_argument("--yolo_weights", type=str, default=None,
                        help="Path to YOLOv11-cls best.pt (required unless --precomputed_branches is set)")
    parser.add_argument("--precomputed_branches", type=str, default=None,
                        help="Path to .npz from precompute_branches.py. Saves RAM and time on CPU; no ViT/YOLO loaded.")
    parser.add_argument("--output_dir", type=str, default="./inception_fusion_weights",
                        help="Where to save googlenet_fusion.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--vit_hf_repo", type=str, default="natix-network-org/roadwork",
                        help="HuggingFace repo for default ViT")
    parser.add_argument("--fusion_aux_weight", type=float, default=0.3,
                        help="Weight for GoogLeNet auxiliary CE loss")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Apply data augmentation to subnet dataset (default: True).")
    parser.add_argument("--no_augment", action="store_false", dest="augment",
                        help="Disable data augmentation.")
    parser.add_argument("--augment_strength", type=str, default="light",
                        choices=["strong", "moderate", "light"],
                        help="Augmentation strength for subnet dataset: light (default, mild), moderate, or strong.")
    parser.add_argument("--fusion_mlp", action="store_true",
                        help="Use a deeper fusion head (3->16->2 with ReLU/dropout) instead of single linear layer.")
    parser.add_argument("--validation_split", type=float, default=0.15,
                        help="Fraction of data for validation (0 = no validation, no early stopping). Default 0.15.")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Stop training after this many epochs without validation loss improvement. Default 3.")
    args = parser.parse_args()

    if not args.precomputed_branches and not args.yolo_weights:
        raise SystemExit("Provide either --yolo_weights or --precomputed_branches (from precompute_branches.py).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda:0" if device.type == "cuda" else "cpu"
    precomputed = args.precomputed_branches is not None
    if precomputed:
        import numpy as np
        cache = np.load(args.precomputed_branches)
        p_vit_cache = torch.from_numpy(cache["p_vit"].astype(np.float32))
        p_yolo_cache = torch.from_numpy(cache["p_yolo"].astype(np.float32))
        n_pre = len(p_vit_cache)
        print(f"Loaded precomputed branches: {n_pre} samples (no ViT/YOLO will be loaded).")
    else:
        p_vit_cache = p_yolo_cache = None
        n_pre = None

    print("Loading dataset (subnet data only)...")
    train_ds = load_labeled_dataset(args.dataset_path, split=args.split)
    print(f"Train samples: {len(train_ds)}")

    if precomputed and n_pre != len(train_ds):
        raise SystemExit(
            f"Precomputed cache has {n_pre} samples but dataset has {len(train_ds)}. "
            "Use the same --dataset_path and --split as when you ran precompute_branches.py."
        )

    if not precomputed:
        print("Loading ViT (frozen)...")
        vit_pipeline = pipeline(
            "image-classification",
            model=AutoModelForImageClassification.from_pretrained(args.vit_hf_repo),
            feature_extractor=AutoImageProcessor.from_pretrained(args.vit_hf_repo, use_fast=True),
            device=-1 if device_str == "cpu" else 0,
        )
        print("Loading YOLO (frozen)...")
        from ultralytics import YOLO
        yolo_model = YOLO(args.yolo_weights)
    else:
        vit_pipeline = None
        yolo_model = None

    print("Loading GoogLeNet + fusion head...")
    from torchvision.models import googlenet, GoogLeNet_Weights
    import torchvision.transforms as T

    googlenet_model = googlenet(weights=GoogLeNet_Weights.DEFAULT)
    in_features = googlenet_model.fc.in_features
    googlenet_model.fc = nn.Linear(in_features, 2)
    googlenet_model.aux_logits = False
    googlenet_model = googlenet_model.to(device)
    if args.fusion_mlp:
        fusion_head = FusionHeadMLP(num_branches=3, num_classes=2, hidden=16, dropout=0.1).to(device)
        print("Using MLP fusion head (3->16->2 with ReLU/dropout).")
    else:
        fusion_head = FusionHead(num_branches=3, num_classes=2).to(device)

    # Subnet dataset augmentation: strength light (default) / moderate / strong
    augment_strength = getattr(args, "augment_strength", "light")
    if args.augment:
        if augment_strength == "strong":
            googlenet_transform = T.Compose([
                T.Resize((280, 280)),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.2),
                T.RandomRotation(25),
                T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.2, hue=0.08),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif augment_strength == "moderate":
            googlenet_transform = T.Compose([
                T.Resize((256, 256)),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:  # light
            googlenet_transform = T.Compose([
                T.Resize((240, 240)),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.03),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        print(f"Subnet dataset augmentation enabled (strength={augment_strength}): crop, flip, rotation, color jitter.")
    else:
        googlenet_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # Validation transform (no augmentation)
    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    optim = torch.optim.Adam(
        list(googlenet_model.fc.parameters()) + list(fusion_head.parameters()),
        lr=args.lr,
    )
    ce = nn.CrossEntropyLoss()

    def collate(batch):
        images = [b["image"] for b in batch]
        if images[0].mode != "RGB":
            images = [im.convert("RGB") for im in images]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long, device=device)
        googlenet_tensors = torch.stack([googlenet_transform(im) for im in images]).to(device)
        return images, googlenet_tensors, labels

    n_samples = len(train_ds)
    # Train/val split for early stopping (reproducible)
    use_early_stopping = args.validation_split > 0 and args.early_stopping_patience > 0
    if use_early_stopping:
        n_val = max(0, int(n_samples * args.validation_split))
        n_train = n_samples - n_val
        if n_val == 0:
            use_early_stopping = False
            print("Validation split is 0 or too small; early stopping disabled.")
        else:
            torch.manual_seed(42)
            perm = torch.randperm(n_samples, device=device)
            val_indices = perm[:n_val].cpu().tolist()
            train_indices = perm[n_val:].cpu().tolist()
            print(f"Early stopping: validation_split={args.validation_split}, patience={args.early_stopping_patience}, n_train={n_train}, n_val={n_val}")
    else:
        train_indices = list(range(n_samples))
        val_indices = []

    if not precomputed:
        train_subset = Subset(train_ds, train_indices)
        loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=0,
        )
    else:
        p_vit_cache = p_vit_cache.to(device)
        p_yolo_cache = p_yolo_cache.to(device)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_pt = Path(args.output_dir) / "googlenet_fusion.pt"

    best_val_loss = float("inf")
    patience_counter = 0

    def run_validation():
        googlenet_model.eval()
        fusion_head.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for start in range(0, len(val_indices), args.batch_size):
                end = min(start + args.batch_size, len(val_indices))
                batch_indices = val_indices[start:end]
                images = [train_ds[int(i)]["image"] for i in batch_indices]
                if images[0].mode != "RGB":
                    images = [im.convert("RGB") for im in images]
                googlenet_tensors = torch.stack([val_transform(im) for im in images]).to(device)
                if precomputed:
                    labels = torch.tensor(labels_cache[np.array(batch_indices)], dtype=torch.long, device=device)
                    b_idx = torch.tensor(batch_indices, device=device)
                    p_vit = p_vit_cache[b_idx]
                    p_yolo = p_yolo_cache[b_idx]
                else:
                    labels = torch.tensor([train_ds[int(i)]["label"] for i in batch_indices], dtype=torch.long, device=device)
                    p_vit = get_vit_probs(vit_pipeline, images, device_str).to(device)
                    p_yolo = get_yolo_probs(yolo_model, images, device_str).to(device)
                googlenet_logits = googlenet_model(googlenet_tensors)
                p_googlenet = torch.softmax(googlenet_logits, dim=1)[:, 1]
                branch_probs = torch.stack([p_vit, p_yolo, p_googlenet], dim=1)
                fusion_logits = fusion_head(branch_probs)
                loss = ce(fusion_logits, labels) + args.fusion_aux_weight * ce(googlenet_logits, labels)
                total_loss += loss.item() * len(batch_indices)
                pred = fusion_logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += len(batch_indices)
        googlenet_model.train()
        fusion_head.train()
        return total_loss / total if total > 0 else float("inf"), correct / total if total > 0 else 0.0

    for epoch in range(args.epochs):
        googlenet_model.train()
        fusion_head.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        n_batches = 0
        if precomputed:
            train_perm = torch.randperm(len(train_indices), device=device)
            indices = torch.tensor(train_indices, device=device)[train_perm]
            n_train_batches = len(train_indices)
            pbar = tqdm(range(0, n_train_batches, args.batch_size), desc=f"Epoch {epoch+1}/{args.epochs}")
            for start in pbar:
                end = min(start + args.batch_size, n_train_batches)
                batch_idx = indices[start:end]
                images = [train_ds[int(i)]["image"] for i in batch_idx.cpu().tolist()]
                if images[0].mode != "RGB":
                    images = [im.convert("RGB") for im in images]
                googlenet_tensors = torch.stack([googlenet_transform(im) for im in images]).to(device)
                idx_np = batch_idx.cpu().numpy()
                labels = torch.tensor(labels_cache[idx_np], dtype=torch.long, device=device)
                p_vit = p_vit_cache[batch_idx]
                p_yolo = p_yolo_cache[batch_idx]
                googlenet_logits = googlenet_model(googlenet_tensors)
                p_googlenet = torch.softmax(googlenet_logits, dim=1)[:, 1]
                branch_probs = torch.stack([p_vit, p_yolo, p_googlenet], dim=1)
                fusion_logits = fusion_head(branch_probs)
                loss_fusion = ce(fusion_logits, labels)
                loss_googlenet = ce(googlenet_logits, labels)
                loss = loss_fusion + args.fusion_aux_weight * loss_googlenet
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += loss.item()
                # Calculate accuracy
                preds = torch.argmax(fusion_logits, dim=1)
                correct = (preds == labels).sum().item()
                total_correct += correct
                total_samples += len(labels)
                n_batches += 1
                acc = correct / len(labels)
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")
        else:
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", total=(len(train_indices) + args.batch_size - 1) // args.batch_size)
            for images, googlenet_tensors, labels in pbar:
                with torch.no_grad():
                    p_vit = get_vit_probs(vit_pipeline, images, device_str)
                    p_yolo = get_yolo_probs(yolo_model, images, device_str)
                if p_vit.device != device:
                    p_vit, p_yolo = p_vit.to(device), p_yolo.to(device)
                googlenet_logits = googlenet_model(googlenet_tensors)
                p_googlenet = torch.softmax(googlenet_logits, dim=1)[:, 1]
                branch_probs = torch.stack([p_vit, p_yolo, p_googlenet], dim=1)
                fusion_logits = fusion_head(branch_probs)
                loss_fusion = ce(fusion_logits, labels)
                loss_googlenet = ce(googlenet_logits, labels)
                loss = loss_fusion + args.fusion_aux_weight * loss_googlenet
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += loss.item()
                # Calculate accuracy
                preds = torch.argmax(fusion_logits, dim=1)
                correct = (preds == labels).sum().item()
                total_correct += correct
                total_samples += len(labels)
                n_batches += 1
                acc = correct / len(labels)
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")
        avg_loss = total_loss / n_batches
        avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
        print(f"Epoch {epoch+1} avg train loss: {avg_loss:.4f}, avg train acc: {avg_acc:.4f}")

        # Validation and early stopping
        if use_early_stopping and val_indices:
            val_loss, val_acc = run_validation()
            print(f"  val loss: {val_loss:.4f}, val acc: {val_acc:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    {
                        "googlenet_fc": googlenet_model.fc.state_dict(),
                        "fusion_head": fusion_head.state_dict(),
                        "fusion_head_type": "mlp" if args.fusion_mlp else "linear",
                    },
                    out_pt,
                )
                print(f"  -> best val loss, checkpoint saved.")
            else:
                patience_counter += 1
                print(f"  -> no improvement ({patience_counter}/{args.early_stopping_patience})")
                if patience_counter >= args.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1} (no val loss improvement for {args.early_stopping_patience} epochs).")
                    break

    # Save final model only when we didn't use early stopping (when using early stopping, best was saved in the loop)
    if not use_early_stopping or not val_indices:
        torch.save(
            {
                "googlenet_fc": googlenet_model.fc.state_dict(),
                "fusion_head": fusion_head.state_dict(),
                "fusion_head_type": "mlp" if args.fusion_mlp else "linear",
            },
            out_pt,
        )
    print(f"Saved GoogLeNet head + fusion to {out_pt}")
    print("Next: set googlenet_fusion_path in base_miner/detectors/configs/inception_fusion_roadwork.yaml and use IMAGE_DETECTOR=InceptionFusion.")


if __name__ == "__main__":
    main()

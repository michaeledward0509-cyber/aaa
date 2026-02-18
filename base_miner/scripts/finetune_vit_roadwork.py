#!/usr/bin/env python3
"""
Fine-tune a ViT for roadwork binary classification (subnet-compatible).

Uses a pre-trained model as base and your labeled data. Output model can be
loaded by the miner with ViT + a config pointing to the saved repo/path.

Pre-trained options (--base_model):
  - natix-network-org/roadwork   Default subnet model; good starting point (recommended).
  - google/vit-base-patch16-224  Generic ViT; train from scratch on roadwork.

Dataset: imagefolder with train/roadwork/ and train/no_roadwork/ (or use --dataset_path HF).

Example:
  poetry run python base_miner/scripts/finetune_vit_roadwork.py \\
    --dataset_path imagefolder:/path/to/data \\
    --output_dir ./out_roadwork \\
    --base_model natix-network-org/roadwork \\
    --epochs 3 --batch_size 16
"""

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

# Repo root for imports if needed
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Labels must match miner's expectation: "Roadwork" (1) and "None" (0)
LABEL2ID = {"no_roadwork": 0, "roadwork": 1}
ID2LABEL = {0: "None", 1: "Roadwork"}


def load_labeled_dataset_for_training(dataset_path: str, split: str = "train"):
    """Load dataset with (image, label). Prefer imagefolder."""
    if dataset_path.startswith("imagefolder:"):
        _, data_dir = dataset_path.split(":", 1)
        dataset = load_dataset("imagefolder", data_dir=data_dir)
        if isinstance(dataset, dict):
            if split not in dataset:
                split = "train" if "train" in dataset else list(dataset.keys())[0]
            ds = dataset[split]
        else:
            ds = dataset
    else:
        ds = load_dataset(dataset_path, split=split)

    # Ensure label column exists; imagefolder uses "label" with class names or indices
    if "label" not in ds.column_names and "labels" in ds.column_names:
        ds = ds.rename_column("labels", "label")
    if "label" not in ds.column_names:
        raise ValueError(f"Dataset must have 'label' (or 'labels'). Got: {ds.column_names}")

    # If labels are strings (e.g. "roadwork", "no_roadwork"), map to 0/1
    def map_label(ex):
        l = ex["label"]
        if isinstance(l, str):
            l = l.lower().replace(" ", "_")
            ex["label"] = LABEL2ID.get(l, 1 if "road" in l or l == "1" else 0)
        elif l not in (0, 1):
            ex["label"] = 1 if int(l) > 0 else 0
        return ex

    ds = ds.map(map_label, num_proc=1)
    return ds


def collate_fn(examples):
    """Collate batch: pixel_values + labels (ignore other keys like 'image')."""
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    labels = torch.tensor([ex["label"] for ex in examples], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ViT for roadwork (binary).")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="imagefolder:/path/to/data or HuggingFace org/dataset")
    parser.add_argument("--output_dir", type=str, default="./roadwork_finetuned",
                        help="Where to save the model and checkpoints")
    parser.add_argument("--base_model", type=str, default="natix-network-org/roadwork",
                        help="Pre-trained model: natix-network-org/roadwork or google/vit-base-patch16-224")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use for training")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Max training steps (-1 = use epochs)")
    parser.add_argument("--eval_strategy", type=str, default="no",
                        choices=["no", "steps", "epoch"],
                        help="When to run evaluation")
    parser.add_argument("--save_strategy", type=str, default="epoch",
                        choices=["no", "steps", "epoch"])
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    args = parser.parse_args()

    print("Loading dataset...")
    train_ds = load_labeled_dataset_for_training(args.dataset_path, split=args.split)
    print(f"Train samples: {len(train_ds)}")

    print("Loading processor and model...")
    processor = AutoImageProcessor.from_pretrained(args.base_model)
    model = AutoModelForImageClassification.from_pretrained(
        args.base_model,
        num_labels=2,
    )
    # Ensure label names match what the miner expects
    model.config.id2label = ID2LABEL
    model.config.label2id = {v: k for k, v in ID2LABEL.items()}

    def preprocess(examples):
        images = examples["image"]
        if not isinstance(images, list):
            images = [images]
        inputs = processor(images=images, return_tensors="pt")
        pv = inputs["pixel_values"]
        if pv.dim() == 4 and pv.shape[0] == 1:
            pv = pv.squeeze(0)
        examples["pixel_values"] = pv
        return examples

    train_ds = train_ds.with_transform(preprocess)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        fp16=args.fp16,
        logging_steps=10,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collate_fn,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Model and processor saved to {args.output_dir}")
    print("Next: create a YAML under base_miner/detectors/configs/ with hf_repo pointing to this dir or your HF repo, then run the miner with that config.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download 1/10 of the LouisChen15/ConstructionSite training dataset.

Usage:
    poetry run python workspace/download_louis_subset.py
    poetry run python workspace/download_louis_subset.py --output_dir ./louis_subset --fraction 0.1
"""

import argparse
import os
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download a subset of LouisChen15/ConstructionSite dataset")
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Fraction of data to download (default: 0.1 = 1/10)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory to save images (if None, just loads dataset without saving)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download (default: train)",
    )
    args = parser.parse_args()

    print(f"Loading dataset: LouisChen15/ConstructionSite (split={args.split})...")
    dataset = load_dataset("LouisChen15/ConstructionSite", split=args.split)

    total_samples = len(dataset)
    subset_size = int(total_samples * args.fraction)
    print(f"Total samples: {total_samples}")
    print(f"Downloading {subset_size} samples ({args.fraction*100:.1f}%)...")

    # Select subset (first N samples, or use select() for random sampling)
    subset = dataset.select(range(subset_size))
    print(f"Selected {len(subset)} samples")

    if args.output_dir:
        # Save images to disk in imagefolder format
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create class directories
        train_dir = output_path / "train"
        train_dir.mkdir(exist_ok=True)

        # Group by label
        roadwork_dir = train_dir / "roadwork"
        no_roadwork_dir = train_dir / "no_roadwork"
        roadwork_dir.mkdir(exist_ok=True)
        no_roadwork_dir.mkdir(exist_ok=True)

        print(f"Saving images to {output_path}...")
        for i, example in enumerate(subset):
            # Get label (assuming 0 = no_roadwork, 1 = roadwork)
            label = example.get("label", example.get("labels", 0))
            if isinstance(label, list):
                label = label[0] if label else 0

            # Determine class directory
            if label == 1:
                class_dir = roadwork_dir
            else:
                class_dir = no_roadwork_dir

            # Save image
            image = example["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_path = class_dir / f"{i:06d}.jpg"
            image.save(image_path, "JPEG")

            if (i + 1) % 100 == 0:
                print(f"  Saved {i + 1}/{len(subset)} images...")

        print(f"Done! Saved {len(subset)} images to {output_path}")
        print(f"  - {len([f for f in roadwork_dir.glob('*.jpg')])} roadwork images")
        print(f"  - {len([f for f in no_roadwork_dir.glob('*.jpg')])} no_roadwork images")
        print(f"\nUse with: imagefolder:{output_path}")
    else:
        print(f"Dataset loaded in memory ({len(subset)} samples)")
        print("To save to disk, use --output_dir <path>")

    return subset


if __name__ == "__main__":
    main()

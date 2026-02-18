"""
Split an imagefolder that has only train/ into train/val/test for YOLO classify.
YOLO expects all three; if val/ or test/ are missing it gets None and crashes.

Usage (Colab or laptop):
  DATA_PATH = "/content/drive/MyDrive/roadwork_data"  # or your path
  python split_for_yolo.py /content/drive/MyDrive/roadwork_data

Or from repo root:
  poetry run python workspace/split_for_yolo.py /path/to/roadwork_data
"""
import argparse
import os
import random
import shutil


def split_dataset(data_path: str, val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 42) -> None:
    """Create val/ and test/ by moving a fraction of images from train/."""
    random.seed(seed)
    train_dir = os.path.join(data_path, "train")
    if not os.path.isdir(train_dir):
        raise SystemExit(f"Expected {train_dir} to exist. data_path must contain train/<classes>/.")

    class_names = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    if not class_names:
        raise SystemExit(f"No class subdirs found under {train_dir}.")

    for split in ("val", "test"):
        for c in class_names:
            os.makedirs(os.path.join(data_path, split, c), exist_ok=True)

    total_moved = 0
    for c in class_names:
        class_dir = os.path.join(train_dir, c)
        names = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        random.shuffle(names)
        n = len(names)
        n_val = max(1, int(n * val_frac))
        n_test = max(1, int(n * test_frac))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_val, n_test = 1, 1
            n_train = n - 2
        val_names = names[:n_val]
        test_names = names[n_val : n_val + n_test]
        for f in val_names:
            src = os.path.join(class_dir, f)
            dst = os.path.join(data_path, "val", c, f)
            shutil.move(src, dst)
            total_moved += 1
        for f in test_names:
            src = os.path.join(class_dir, f)
            dst = os.path.join(data_path, "test", c, f)
            shutil.move(src, dst)
            total_moved += 1

    print(f"Split complete. Moved {total_moved} images into val/ and test/. Use data={data_path} for YOLO.")


def main() -> None:
    p = argparse.ArgumentParser(description="Split train/ into train/val/test for YOLO classify.")
    p.add_argument("data_path", help="Path to dataset root (containing train/<classes>/).")
    p.add_argument("--val", type=float, default=0.1, help="Fraction for validation (default 0.1).")
    p.add_argument("--test", type=float, default=0.1, help="Fraction for test (default 0.1).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = p.parse_args()
    split_dataset(args.data_path, val_frac=args.val, test_frac=args.test, seed=args.seed)


if __name__ == "__main__":
    main()

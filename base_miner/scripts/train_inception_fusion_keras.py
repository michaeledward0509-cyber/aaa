#!/usr/bin/env python3
"""
Train the Inception-style fusion model (ViT + YOLO + EfficientNetV2) using Keras.

Uses tf.keras for the EfficientNetV2 branch (replaces InceptionV3/GoogLeNet) and the fusion head. ViT and
YOLO are still run in PyTorch to obtain branch probabilities; only the
EfficientNetV2 top and fusion layer are built and trained in Keras. Subnet data only.

Prerequisites: same as train_inception_fusion.py (YOLO weights, subnet dataset).

Example:
  poetry run python base_miner/scripts/train_inception_fusion_keras.py \\
    --dataset_path imagefolder:/path/to/roadwork_data \\
    --yolo_weights /path/to/yolo/weights/best.pt \\
    --output_dir ./inception_fusion_keras \\
    --epochs 10 --batch_size 16
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

# PyTorch for ViT and YOLO (frozen)
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

LABEL2ID = {"no_roadwork": 0, "roadwork": 1}
IMG_SIZE = 224


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


def build_efficientnetv2_keras(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=2, model_size="b0"):
    """EfficientNetV2 as CNN branch (Keras). Replaces InceptionV3/GoogLeNet."""
    from tensorflow import keras
    from tensorflow.keras import layers

    # EfficientNetV2 models: b0, b1, b2, b3, s, m, l, xl
    # b0 is smallest/fastest, good starting point
    model_map = {
        "b0": keras.applications.EfficientNetV2B0,
        "b1": keras.applications.EfficientNetV2B1,
        "b2": keras.applications.EfficientNetV2B2,
        "b3": keras.applications.EfficientNetV2B3,
        "s": keras.applications.EfficientNetV2S,
        "m": keras.applications.EfficientNetV2M,
        "l": keras.applications.EfficientNetV2L,
    }
    
    efficientnet_class = model_map.get(model_size.lower(), keras.applications.EfficientNetV2B0)
    
    base = efficientnet_class(
        include_top=False,
        weights="imagenet",  # Use pretrained ImageNet weights
        input_shape=input_shape,
        pooling="avg",
    )
    # Freeze the pretrained EfficientNetV2 backbone completely
    base.trainable = False
    # Ensure all layers are frozen (explicit freeze)
    for layer in base.layers:
        layer.trainable = False
    
    x = base.output
    # Add trainable classification head (only this will be trained)
    head = layers.Dense(num_classes, activation="softmax", name="roadwork_head")
    x = head(x)
    model = keras.Model(inputs=base.input, outputs=x, name="efficientnetv2_branch")
    
    # Verify freezing: print summary of trainable vs non-trainable params
    total_params = model.count_params()
    trainable_params = sum([keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    print(f"EfficientNetV2-{model_size}: Total params: {total_params:,}, "
          f"Trainable: {trainable_params:,} (head only), "
          f"Non-trainable: {non_trainable_params:,} (frozen pretrained backbone)")
    
    return model


def build_fusion_keras(num_branches=3, num_classes=2, use_mlp=False, hidden=16, dropout=0.1):
    """Fusion head: [p_vit, p_yolo, p_efficientnet] -> logits (2 classes). Optionally a small MLP (3->hidden->2)."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = layers.Input(shape=(num_branches,), name="branch_probs")
    if use_mlp:
        x = layers.Dense(hidden, activation="relu", name="fusion_hidden")(inp)
        x = layers.Dropout(dropout, name="fusion_dropout")(x)
        out = layers.Dense(num_classes, activation="softmax", name="fusion_out")(x)
    else:
        out = layers.Dense(num_classes, activation="softmax", name="fusion_out")(inp)
    return keras.Model(inputs=inp, outputs=out, name="fusion_head")


# Subnet dataset augmentation: strength controls how strong the transforms are.
AUGMENT_STRENGTH = {
    "strong": {"rotation": 25, "brightness": (0.7, 1.35), "contrast": (0.7, 1.35), "vertical_flip_p": 0.2},
    "moderate": {"rotation": 15, "brightness": (0.85, 1.15), "contrast": (0.85, 1.15), "vertical_flip_p": 0.15},
    "light": {"rotation": 10, "brightness": (0.9, 1.1), "contrast": (0.9, 1.1), "vertical_flip_p": 0.1},
}


def augment_image_pil(image_pil, strength="strong", rng=None):
    """Data augmentation for subnet dataset: flip, rotation, brightness/contrast (strength: strong/moderate/light)."""
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np

    if rng is None:
        rng = np.random.default_rng()
    cfg = AUGMENT_STRENGTH.get(strength, AUGMENT_STRENGTH["light"])
    img = image_pil.convert("RGB")
    # Random horizontal flip (natural for road scenes)
    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Random vertical flip (less common)
    if rng.random() < cfg["vertical_flip_p"]:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    # Random rotation
    angle = float(rng.uniform(-cfg["rotation"], cfg["rotation"]))
    # With expand=False, no fill is needed (image stays same size)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
    # Brightness and contrast
    lo_b, hi_b = cfg["brightness"]
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(float(rng.uniform(lo_b, hi_b)))
    lo_c, hi_c = cfg["contrast"]
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(float(rng.uniform(lo_c, hi_c)))
    # Strong only: slight blur (simulates motion/defocus)
    if strength == "strong" and rng.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img


def preprocess_keras(image_pil, augment=False, augment_strength="strong", rng=None):
    """Resize and normalize for EfficientNetV2 (ImageNet normalization). Optionally apply subnet dataset augmentation."""
    from PIL import Image
    import numpy as np

    if augment:
        image_pil = augment_image_pil(image_pil, strength=augment_strength, rng=rng)
    img = image_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return arr


def main():
    parser = argparse.ArgumentParser(
        description="Train Inception fusion (ViT+YOLO+GoogLeNet) with Keras on subnet data."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="imagefolder:/path/to/data or HuggingFace org/dataset (subnet data only)",
    )
    parser.add_argument(
        "--yolo_weights",
        type=str,
        default=None,
        help="Path to YOLOv11-cls best.pt (required unless --precomputed_branches is set)",
    )
    parser.add_argument(
        "--precomputed_branches",
        type=str,
        default=None,
        help="Path to .npz from precompute_branches.py (p_vit, p_yolo, labels). Saves RAM and time on CPU; no ViT/YOLO loaded.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inception_fusion_keras",
        help="Where to save Keras models (efficientnetv2_branch.keras, fusion_head.keras)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--vit_hf_repo",
        type=str,
        default="natix-network-org/roadwork",
        help="HuggingFace repo for default ViT",
    )
    parser.add_argument(
        "--fusion_aux_weight",
        type=float,
        default=0.3,
        help="Weight for EfficientNetV2 branch CE loss",
    )
    parser.add_argument(
        "--efficientnet_size",
        type=str,
        default="b0",
        choices=["b0", "b1", "b2", "b3", "s", "m", "l"],
        help="EfficientNetV2 model size: b0 (smallest/fastest) to l (largest/most accurate). Default: b0",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Apply data augmentation to subnet dataset images (EfficientNetV2 branch). Default: True.",
    )
    parser.add_argument("--no_augment", action="store_false", dest="augment", help="Disable data augmentation.")
    parser.add_argument(
        "--augment_strength",
        type=str,
        default="light",
        choices=["strong", "moderate", "light"],
        help="Augmentation strength for subnet dataset: light (default, mild), moderate, or strong.",
    )
    parser.add_argument(
        "--fusion_mlp",
        action="store_true",
        help="Use a deeper fusion head (3->16->2 with ReLU/dropout) instead of single dense layer.",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.15,
        help="Fraction of train data for validation when --validation_from fraction. Ignored when --validation_from folder. Default 0.15.",
    )
    parser.add_argument(
        "--validation_from",
        type=str,
        default="fraction",
        choices=["fraction", "folder"],
        help="Validation set: 'fraction' = use --validation_split of train; 'folder' = use dataset's val split (e.g. roadwork_data/val). Default fraction.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Stop training after this many epochs without validation loss improvement. Default 3.",
    )
    args = parser.parse_args()

    if not args.precomputed_branches and not args.yolo_weights:
        raise SystemExit("Provide either --yolo_weights or --precomputed_branches (from precompute_branches.py).")
    if args.validation_from == "folder" and args.precomputed_branches and not args.yolo_weights:
        raise SystemExit("When using --validation_from folder with --precomputed_branches, you must also pass --yolo_weights so ViT and YOLO can be loaded for validation on the val set.")

    import tensorflow as tf

    # Avoid TF claiming all GPU memory if we also use PyTorch
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    precomputed = args.precomputed_branches is not None
    if precomputed:
        cache = np.load(args.precomputed_branches)
        p_vit_cache = cache["p_vit"]
        p_yolo_cache = cache["p_yolo"]
        labels_cache = cache["labels"]
        n_pre = len(p_vit_cache)
        print(f"Loaded precomputed branches: {n_pre} samples (no ViT/YOLO will be loaded).")
    else:
        p_vit_cache = p_yolo_cache = labels_cache = None
        n_pre = None

    print("Loading dataset (subnet data only)...")
    train_ds = load_labeled_dataset(args.dataset_path, split=args.split)
    print(f"Train samples: {len(train_ds)}")

    if precomputed and n_pre != len(train_ds):
        raise SystemExit(
            f"Precomputed cache has {n_pre} samples but dataset has {len(train_ds)}. "
            "Use the same --dataset_path and --split as when you ran precompute_branches.py."
        )

    # Load val split from folder when requested (e.g. roadwork_data/val or roadwork_data/validation)
    val_ds = None
    if args.validation_from == "folder":
        for val_split_name in ("validation", "val"):
            try:
                val_ds = load_labeled_dataset(args.dataset_path, split=val_split_name)
                print(f"Validation from folder: {len(val_ds)} samples (split '{val_split_name}')")
                break
            except Exception:
                continue
        if val_ds is None:
            raise SystemExit(
                "Failed to load val split for --validation_from folder. "
                "Ensure your dataset has a 'val' or 'validation' folder (e.g. roadwork_data/val/ or roadwork_data/validation/ with roadwork/ and no_roadwork/)."
            )

    # Load ViT and YOLO: needed when not precomputed, or when precomputed but using val folder (to run validation)
    need_vit_yolo_for_val = (val_ds is not None and precomputed)
    if not precomputed or need_vit_yolo_for_val:
        print("Loading ViT (frozen, PyTorch)...")
        vit_pipeline = pipeline(
            "image-classification",
            model=AutoModelForImageClassification.from_pretrained(args.vit_hf_repo),
            feature_extractor=AutoImageProcessor.from_pretrained(args.vit_hf_repo, use_fast=True),
            device=-1 if device_str == "cpu" else 0,
        )
        print("Loading YOLO (frozen, PyTorch)...")
        from ultralytics import YOLO
        yolo_model = YOLO(args.yolo_weights)
    else:
        vit_pipeline = None
        yolo_model = None

    print(f"Building Keras models (EfficientNetV2-{args.efficientnet_size} + fusion)...")
    efficientnet_model = build_efficientnetv2_keras(num_classes=2, model_size=args.efficientnet_size)
    fusion_model = build_fusion_keras(
        num_branches=3, num_classes=2, use_mlp=args.fusion_mlp, hidden=16, dropout=0.1
    )
    if args.fusion_mlp:
        print("Using MLP fusion head (3->16->2 with ReLU/dropout).")

    efficientnet_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    fusion_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(train_ds)
    # Train/val: either use a fraction of train, or a separate val folder
    use_early_stopping = args.early_stopping_patience > 0
    val_indices = []  # indices into train_ds (used when validation_from fraction)
    if val_ds is not None:
        use_early_stopping = use_early_stopping and len(val_ds) > 0
        train_indices = np.arange(n).tolist()
        if use_early_stopping:
            print(f"Early stopping: validation from folder (val), patience={args.early_stopping_patience}, n_train={n}, n_val={len(val_ds)}")
    elif use_early_stopping and args.validation_split > 0:
        n_val = max(0, int(n * args.validation_split))
        n_train = n - n_val
        if n_val == 0:
            use_early_stopping = False
            print("Validation split is 0 or too small; early stopping disabled.")
        else:
            rng_split = np.random.default_rng(42)
            perm = rng_split.permutation(n)
            val_indices = perm[:n_val].tolist()
            train_indices = perm[n_val:].tolist()
            print(f"Early stopping: validation_split={args.validation_split}, patience={args.early_stopping_patience}, n_train={n_train}, n_val={n_val}")
    else:
        train_indices = np.arange(n).tolist()
        if not use_early_stopping:
            val_indices = []

    indices = np.array(train_indices)
    rng = np.random.default_rng()
    if args.augment:
        print(f"Subnet dataset augmentation enabled (strength={args.augment_strength}): flip, rotation, brightness/contrast.")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        np.random.shuffle(indices)
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        n_train_batches = len(train_indices)
        pbar = tqdm(range(0, n_train_batches, args.batch_size), desc=f"Epoch {epoch+1}/{args.epochs}")
        for start in pbar:
            end = min(start + args.batch_size, n_train_batches)
            batch_idx = indices[start:end]
            images = [train_ds[int(i)]["image"] for i in batch_idx]
            if images[0].mode != "RGB":
                images = [im.convert("RGB") for im in images]
            labels = np.array([train_ds[int(i)]["label"] for i in batch_idx], dtype=np.int32)
            if precomputed:
                p_vit = p_vit_cache[batch_idx]
                p_yolo = p_yolo_cache[batch_idx]
            else:
                p_vit = get_vit_probs(vit_pipeline, images, device_str)
                p_yolo = get_yolo_probs(yolo_model, images, device_str)
            # EfficientNetV2 branch (Keras); apply subnet dataset augmentation when enabled
            X_eff = np.stack([
                preprocess_keras(im, augment=args.augment, augment_strength=args.augment_strength, rng=rng)
                for im in images
            ])
            p_efficientnet_logits = efficientnet_model(X_eff, training=True)
            p_efficientnet = p_efficientnet_logits[:, 1].numpy()
            branch_probs = np.stack([p_vit, p_yolo, p_efficientnet], axis=1).astype(np.float32)

            # Train EfficientNetV2 branch
            result_g = efficientnet_model.train_on_batch(X_eff, labels)
            # Train fusion head
            result_f = fusion_model.train_on_batch(branch_probs, labels)
            
            # Extract loss and accuracy from results
            lg = result_g[0] if isinstance(result_g, (list, tuple)) else result_g
            acc_g = result_g[1] if isinstance(result_g, (list, tuple)) and len(result_g) > 1 else 0.0
            lf = result_f[0] if isinstance(result_f, (list, tuple)) else result_f
            acc_f = result_f[1] if isinstance(result_f, (list, tuple)) and len(result_f) > 1 else 0.0
            
            loss = lf + args.fusion_aux_weight * lg
            # Weighted accuracy (fusion is primary, EfficientNetV2 is auxiliary)
            acc = acc_f  # Use fusion accuracy as primary metric
            total_loss += loss
            total_acc += acc
            n_batches += 1
            pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.4f}")
        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches
        print(f"Epoch {epoch+1} avg train loss: {avg_loss:.4f}, avg train acc: {avg_acc:.4f}")

        # Validation and early stopping (from val_indices into train_ds, or from val_ds folder)
        if use_early_stopping and (val_indices or val_ds is not None):
            val_loss_sum = 0.0
            val_acc_sum = 0.0
            val_n = 0
            if val_ds is not None:
                # Validation from folder (roadwork_data/val): run ViT and YOLO on val images
                for start in range(0, len(val_ds), args.batch_size):
                    end = min(start + args.batch_size, len(val_ds))
                    batch_indices = list(range(start, end))
                    images = [val_ds[int(i)]["image"] for i in batch_indices]
                    if images[0].mode != "RGB":
                        images = [im.convert("RGB") for im in images]
                    labels_val = np.array([val_ds[int(i)]["label"] for i in batch_indices], dtype=np.int32)
                    p_vit_v = get_vit_probs(vit_pipeline, images, device_str)
                    p_yolo_v = get_yolo_probs(yolo_model, images, device_str)
                    X_eff_val = np.stack([preprocess_keras(im, augment=False) for im in images])
                    p_efficientnet_val = efficientnet_model(X_eff_val, training=False).numpy()[:, 1]
                    branch_val = np.stack([p_vit_v, p_yolo_v, p_efficientnet_val], axis=1).astype(np.float32)
                    result_g = efficientnet_model.evaluate(X_eff_val, labels_val, verbose=0)
                    result_f = fusion_model.evaluate(branch_val, labels_val, verbose=0)
                    l_g = result_g[0] if isinstance(result_g, (list, tuple)) else result_g
                    l_f = result_f[0] if isinstance(result_f, (list, tuple)) else result_f
                    acc_f = result_f[1] if isinstance(result_f, (list, tuple)) and len(result_f) > 1 else 0.0
                    val_loss_sum += (l_f + args.fusion_aux_weight * l_g) * len(batch_indices)
                    val_acc_sum += acc_f * len(batch_indices)
                    val_n += len(batch_indices)
            else:
                # Validation from fraction of train (can use precomputed cache)
                for start in range(0, len(val_indices), args.batch_size):
                    end = min(start + args.batch_size, len(val_indices))
                    batch_indices = val_indices[start:end]
                    images = [train_ds[int(i)]["image"] for i in batch_indices]
                    if images[0].mode != "RGB":
                        images = [im.convert("RGB") for im in images]
                    labels_val = np.array([train_ds[int(i)]["label"] for i in batch_indices], dtype=np.int32)
                    if precomputed:
                        p_vit_v = p_vit_cache[batch_indices]
                        p_yolo_v = p_yolo_cache[batch_indices]
                    else:
                        p_vit_v = get_vit_probs(vit_pipeline, images, device_str)
                        p_yolo_v = get_yolo_probs(yolo_model, images, device_str)
                    X_eff_val = np.stack([preprocess_keras(im, augment=False) for im in images])
                    p_efficientnet_val = efficientnet_model(X_eff_val, training=False).numpy()[:, 1]
                    branch_val = np.stack([p_vit_v, p_yolo_v, p_efficientnet_val], axis=1).astype(np.float32)
                    result_g = efficientnet_model.evaluate(X_eff_val, labels_val, verbose=0)
                    result_f = fusion_model.evaluate(branch_val, labels_val, verbose=0)
                    l_g = result_g[0] if isinstance(result_g, (list, tuple)) else result_g
                    l_f = result_f[0] if isinstance(result_f, (list, tuple)) else result_f
                    acc_f = result_f[1] if isinstance(result_f, (list, tuple)) and len(result_f) > 1 else 0.0
                    val_loss_sum += (l_f + args.fusion_aux_weight * l_g) * len(batch_indices)
                    val_acc_sum += acc_f * len(batch_indices)
                    val_n += len(batch_indices)
            val_loss = val_loss_sum / val_n if val_n > 0 else float("inf")
            val_acc = val_acc_sum / val_n if val_n > 0 else 0.0
            print(f"  val loss: {val_loss:.4f}, val acc: {val_acc:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                efficientnet_model.save(out_dir / "efficientnetv2_branch.keras")
                fusion_model.save(out_dir / "fusion_head.keras")
                print(f"  -> best val loss, checkpoint saved.")
            else:
                patience_counter += 1
                print(f"  -> no improvement ({patience_counter}/{args.early_stopping_patience})")
                if patience_counter >= args.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1} (no val loss improvement for {args.early_stopping_patience} epochs).")
                    break

    if not use_early_stopping or (not val_indices and val_ds is None):
        efficientnet_model.save(out_dir / "efficientnetv2_branch.keras")
        fusion_model.save(out_dir / "fusion_head.keras")
    print(f"Saved Keras models to {out_dir}")
    print("  - efficientnetv2_branch.keras")
    print("  - fusion_head.keras")
    print("Next: set googlenet_fusion_path to this directory and use_keras: true in inception_fusion_roadwork.yaml, IMAGE_DETECTOR=InceptionFusion.")


if __name__ == "__main__":
    main()

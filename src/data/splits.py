"""
Stratified train / validation / test splits with near-duplicate perceptual hashing.

Discovers images per class folder (.jpg, .jpeg, .png, .bmp, .tif, .tiff), computes
canonical rotation/flip-invariant perceptual hashes to group identical or near-duplicate
images, and creates a reproducible 70/15/15 split using StratifiedGroupKFold (fixed seed 42).
Subsequent runs reload the saved CSVs to guarantee zero data leakage.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.config import AppConfig

logger = logging.getLogger(__name__)

VALID_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_rgb_image(img_path: Path | str) -> Image.Image:
    """Safely open any supported image format and convert it to 8-bit RGB.

    Handles 16-bit TIFFs (mode 'I;16', 'I', 'F'), palette images ('P'),
    RGBA/LA with alpha, and grayscale ('L').
    """
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(img_path) as img:
        if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            arr = np.array(img, dtype=np.float32)
            max_val = float(arr.max()) if arr.size > 0 else 1.0
            if max_val > 255.0:
                arr = (arr / max(max_val, 65535.0)) * 255.0
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            alpha_composite = Image.alpha_composite(bg, rgba)
            return alpha_composite.convert("RGB")
        return img.convert("RGB")


def compute_canonical_hash(img: Image.Image) -> str:
    """Compute canonical 64-bit perceptual hash invariant across 8 D4 rotations/flips.

    Steps:
    1. 64x64 RGB thumbnail
    2. 8x8 grayscale
    3. Bit string (pixel > mean)
    4. Lexicographical minimum over all 8 rotations and flips
    """
    thumb = img.resize((64, 64), Image.Resampling.BILINEAR)
    gray = thumb.convert("L").resize((8, 8), Image.Resampling.BILINEAR)
    arr = np.array(gray, dtype=np.float32)
    mean_val = float(arr.mean())

    bit_arrays: List[str] = []
    base_variants = [arr, np.fliplr(arr)]
    for base in base_variants:
        for k in range(4):
            rotated = np.rot90(base, k)
            bits = (rotated > mean_val).astype(int).flatten()
            bit_str = "".join(map(str, bits))
            bit_arrays.append(bit_str)
    return min(bit_arrays)


def discover_dataset_images(cfg: AppConfig) -> pd.DataFrame:
    """Scan raw directory for images across all configured classes and compute canonical hashes."""
    raw_dir = cfg.data.raw_dir
    classes = cfg.data.classes
    records = []

    logger.info("Discovering images in %s across %d classes...", raw_dir, len(classes))
    for class_idx, class_name in enumerate(classes):
        class_folder = raw_dir / class_name
        if not class_folder.exists():
            logger.warning("Class directory not found: %s", class_folder)
            continue

        file_paths = [
            p for p in class_folder.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]
        file_paths.sort(key=lambda p: str(p))

        logger.info("Found %d images for class '%s'", len(file_paths), class_name)
        for p in file_paths:
            try:
                rgb_img = load_rgb_image(p)
                c_hash = compute_canonical_hash(rgb_img)
            except Exception as e:
                logger.error("Error computing hash for %s: %s", p, e)
                c_hash = f"error_{p.stem}"

            records.append({
                "original_path": str(p),
                "class_folder": class_name,
                "label_index": class_idx,
                "filename": p.name,
                "canonical_hash": c_hash,
            })

    if not records:
        return pd.DataFrame(columns=["original_path", "class_folder", "label_index", "filename", "canonical_hash", "group_id"])

    df = pd.DataFrame(records)
    # Assign unique integer group ID per unique canonical hash
    unique_hashes = {h: idx for idx, h in enumerate(df["canonical_hash"].unique())}
    df["group_id"] = df["canonical_hash"].map(unique_hashes)
    return df


def compute_class_weights(
    train_df: pd.DataFrame,
    num_classes: int = 5,
) -> Dict[int, float]:
    """Compute balanced class weights for imbalanced training distributions."""
    labels = train_df["label_index"].values
    unique_classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=labels,
    )
    weight_dict = {int(cls): float(w) for cls, w in zip(unique_classes, weights)}
    logger.info("Computed class weights: %s", weight_dict)
    return weight_dict


def print_duplicate_summary(df: pd.DataFrame) -> None:
    """Print summary of duplicate groups and total images in duplicate groups."""
    if len(df) == 0:
        return
    group_counts = df["canonical_hash"].value_counts()
    dup_groups = group_counts[group_counts > 1]
    num_dup_groups = len(dup_groups)
    total_dup_images = int(dup_groups.sum()) if num_dup_groups > 0 else 0

    print("\n" + "=" * 70)
    print("NEAR-DUPLICATE PERCEPTUAL HASH SUMMARY")
    print("=" * 70)
    print(f"Total images analyzed   : {len(df)}")
    print(f"Unique canonical groups : {len(group_counts)}")
    print(f"Duplicate groups (>1 img): {num_dup_groups}")
    print(f"Images in duplicate grps: {total_dup_images}")
    print("=" * 70)


def print_class_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    classes: List[str],
) -> pd.DataFrame:
    """Print a formatted summary table of class counts and percentages across splits."""
    total_samples = len(train_df) + len(val_df) + len(test_df)
    rows = []

    print("\n" + "=" * 70)
    print("DATASET SPLIT CLASS DISTRIBUTION")
    print("=" * 70)
    print(f"{'Class Name':<15} | {'Train':^14} | {'Val':^14} | {'Test':^14} | {'Total':^10}")
    print("-" * 70)

    for idx, cls_name in enumerate(classes):
        n_train = int((train_df["label_index"] == idx).sum()) if len(train_df) > 0 else 0
        n_val = int((val_df["label_index"] == idx).sum()) if len(val_df) > 0 else 0
        n_test = int((test_df["label_index"] == idx).sum()) if len(test_df) > 0 else 0
        n_cls = n_train + n_val + n_test

        pct_train = (n_train / len(train_df) * 100) if len(train_df) > 0 else 0.0
        pct_val = (n_val / len(val_df) * 100) if len(val_df) > 0 else 0.0
        pct_test = (n_test / len(test_df) * 100) if len(test_df) > 0 else 0.0

        print(
            f"{cls_name:<15} | {n_train:5d} ({pct_train:5.1f}%) | "
            f"{n_val:5d} ({pct_val:5.1f}%) | {n_test:5d} ({pct_test:5.1f}%) | {n_cls:6d}"
        )

        rows.append(
            {
                "class_name": cls_name,
                "label_index": idx,
                "train_count": n_train,
                "val_count": n_val,
                "test_count": n_test,
                "total_count": n_cls,
            }
        )

    print("-" * 70)
    train_pct = 100.0 if len(train_df) > 0 else 0.0
    val_pct = 100.0 if len(val_df) > 0 else 0.0
    test_pct = 100.0 if len(test_df) > 0 else 0.0
    print(
        f"{'TOTAL':<15} | {len(train_df):5d} ({train_pct:5.1f}%) | "
        f"{len(val_df):5d} ({val_pct:5.1f}%) | {len(test_df):5d} ({test_pct:5.1f}%) | {total_samples:6d}"
    )
    print("=" * 70 + "\n")

    return pd.DataFrame(rows)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d rows -> %s", len(df), path)


def make_splits(
    cfg: AppConfig,
    force: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create or load reproducible stratified train / val / test splits.

    Uses canonical perceptual hash groups and StratifiedGroupKFold (7 folds, ~70/15/15)
    to guarantee identical/duplicate images never leak across splits.
    """
    splits_dir = cfg.data.splits_dir
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    if not force and train_csv.exists() and val_csv.exists() and test_csv.exists():
        logger.info("Loading existing splits from %s", splits_dir)
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
        print_class_distribution(train_df, val_df, test_df, cfg.data.classes)
        return train_df, val_df, test_df

    logger.info("Discovering images and computing perceptual hashes (seed=%d)...", cfg.seed)
    df = discover_dataset_images(cfg)

    empty_cols = ["original_path", "class_folder", "label_index", "filename", "canonical_hash", "group_id"]
    if len(df) == 0:
        logger.warning("No images found in raw_dir: %s. Empty splits created.", cfg.data.raw_dir)
        train_df = pd.DataFrame(columns=empty_cols)
        val_df = pd.DataFrame(columns=empty_cols)
        test_df = pd.DataFrame(columns=empty_cols)
        _write_csv(train_df, train_csv)
        _write_csv(val_df, val_csv)
        _write_csv(test_df, test_csv)
        return train_df, val_df, test_df

    print_duplicate_summary(df)

    # Use StratifiedGroupKFold
    # Check minimum number of groups across classes
    min_groups_per_class = min(
        df.groupby("label_index")["group_id"].nunique()
    )
    n_splits = min(7, max(3, min_groups_per_class))

    if min_groups_per_class >= 3:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
        splits = list(sgkf.split(df, df["label_index"], df["group_id"]))
        # Fold 0 -> test, Fold 1 -> val, Folds 2..n -> train
        test_idx = splits[0][1]
        val_idx = splits[1][1]
        train_idx = [i for i in range(len(df)) if i not in set(test_idx) and i not in set(val_idx)]

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
    else:
        # Fallback if too few groups per class to fold
        logger.warning("Fewer than 3 groups in at least one class. Falling back to simple stratified split.")
        train_val, test_df = train_test_split(
            df,
            test_size=cfg.data.test_fraction,
            stratify=df["label_index"],
            random_state=cfg.seed,
        )
        adj_val = cfg.data.val_fraction / (1.0 - cfg.data.test_fraction)
        train_df, val_df = train_test_split(
            train_val,
            test_size=adj_val,
            stratify=train_val["label_index"],
            random_state=cfg.seed,
        )

    _write_csv(train_df, train_csv)
    _write_csv(val_df, val_csv)
    _write_csv(test_df, test_csv)

    print_class_distribution(train_df, val_df, test_df, cfg.data.classes)
    return train_df, val_df, test_df

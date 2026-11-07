"""
Stratified train / validation / test splits with Pearson correlation near-duplicate grouping.

Audit findings addressed:
- ~3,100 / 9,097 images are near-duplicates (same photo at 224x224 and 1024x768).
- 16 .bmp files inside PreB folder are single dark cells on black backgrounds from a different source.
- AML = 2000 close-up TIFFs (400x400); ~1000 Normal images are black-background single-cell BMPs (450x450);
  all other images are wide-field smear JPGs.
- Random Forest on mean RGB + black-pixel share reaches 97% accuracy, showing stain color shortcut.

Pipeline steps:
1. Lists files per class, skipping files matching exclude_class_extensions (case-insensitive) and exclude_files.
2. Extracts 64x64 RGB thumbnails (PIL draft mode for JPEG, 8 worker threads) and records original image sizes (w, h).
   Caches thumbnails and sizes in data/processed/thumbs64.npz.
3. Converts thumbnails to 32x32 grayscale, subtracts mean, and L2-normalizes across 8 dihedral transforms.
   Pairs with max Pearson correlation >= dedup_threshold (default 0.995) are grouped with connected_components.
4. Safety checks:
   - Groups > 25 images are kept as singletons (logged warning).
   - Groups containing multiple classes drop all of their members (logged).
5. If drop_duplicates is True, keeps the image with largest width*height (ties: first path alphabetically).
   Saves dropped files to data/splits/dropped_duplicates.csv.
6. Splits remaining images using StratifiedGroupKFold (groups = duplicate groups), 70/15/15, seed 42.
   Saves train.csv, val.csv, test.csv and prints per-split class counts.
"""
from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.config import AppConfig

logger = logging.getLogger(__name__)

VALID_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_rgb_image(img_path: Path | str) -> Image.Image:
    """Safely open any supported image format and convert it to 8-bit RGB."""
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


def load_thumbnail_and_size(path: Path | str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Make a 64x64 RGB thumbnail using PIL draft mode and record original (w, h)."""
    p = Path(path)
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(p) as img:
        orig_w, orig_h = img.size
        if p.suffix.lower() in (".jpg", ".jpeg"):
            try:
                img.draft("RGB", (64, 64))
            except Exception:
                pass
        if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            arr = np.array(img, dtype=np.float32)
            max_val = float(arr.max()) if arr.size > 0 else 1.0
            if max_val > 255.0:
                arr = (arr / max(max_val, 65535.0)) * 255.0
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
            rgb = Image.fromarray(arr).convert("RGB")
        elif img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(bg, rgba).convert("RGB")
        else:
            rgb = img.convert("RGB")
        thumb = rgb.resize((64, 64), Image.Resampling.BILINEAR)
        return np.array(thumb, dtype=np.uint8), (orig_w, orig_h)


def load_thumbnail_64(path: Path | str) -> np.ndarray:
    """Backward-compatible helper returning only 64x64 thumbnail array."""
    thumb, _ = load_thumbnail_and_size(path)
    return thumb


def load_or_compute_thumbnails(
    file_paths: List[Path],
    cache_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load cached 64x64 thumbnails and (w, h) sizes or compute in parallel with 8 workers.

    Returns:
        thumbs64: (N, 64, 64, 3) uint8 array
        sizes: (N, 2) int32 array of [orig_width, orig_height]
    """
    if cache_path.exists():
        try:
            cached = np.load(cache_path)
            if "thumbs" in cached and "sizes" in cached and len(cached["thumbs"]) == len(file_paths):
                logger.info("Loaded %d cached thumbnails and sizes from %s", len(cached["thumbs"]), cache_path)
                return cached["thumbs"], cached["sizes"]
            logger.warning(
                "Cached thumbnails count (%d) mismatches file list (%d). Recomputing...",
                len(cached["thumbs"]) if "thumbs" in cached else 0,
                len(file_paths),
            )
        except Exception as e:
            logger.warning("Failed to load cached thumbnails (%s). Recomputing...", e)

    logger.info("Reading %d image thumbnails and sizes (ThreadPoolExecutor with 8 workers)...", len(file_paths))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(load_thumbnail_and_size, file_paths))

    thumbs_list = [r[0] for r in results]
    sizes_list = [r[1] for r in results]

    thumbs64 = np.stack(thumbs_list, axis=0)
    sizes = np.array(sizes_list, dtype=np.int32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, thumbs=thumbs64, sizes=sizes)
    logger.info("Cached %d thumbnails and sizes to %s", len(thumbs64), cache_path)
    return thumbs64, sizes


def compute_near_duplicate_groups(
    thumbs64: np.ndarray,
    dedup_threshold: float = 0.995,
    chunk_size: int = 500,
) -> Tuple[np.ndarray, int, int, int, int]:
    """Find near-duplicate groups using Pearson correlation across 8 dihedral transforms.

    Returns:
        (group_ids, num_groups, images_inside_groups, largest_group, num_dup_groups)
    """
    N = len(thumbs64)
    if N == 0:
        return np.array([], dtype=np.int32), 0, 0, 0, 0

    # 1. Grayscale & average 2x2 blocks to 32x32
    gray64 = (
        0.2989 * thumbs64[..., 0].astype(np.float32)
        + 0.5870 * thumbs64[..., 1].astype(np.float32)
        + 0.1140 * thumbs64[..., 2].astype(np.float32)
    )  # (N, 64, 64)
    gray32 = gray64.reshape(N, 32, 2, 32, 2).mean(axis=(2, 4))  # (N, 32, 32)

    # 2. Subtract mean and L2-normalize for each of the 8 D4 dihedral transforms
    transforms = []
    base_variants = [gray32, np.flip(gray32, axis=2)]
    for base in base_variants:
        for k in range(4):
            rot = np.rot90(base, k, axes=(1, 2))
            v = rot.reshape(N, -1)
            v = v - v.mean(axis=1, keepdims=True)
            n = np.linalg.norm(v, axis=1, keepdims=True)
            v = np.where(n > 1e-8, v / np.maximum(n, 1e-8), 0.0)
            transforms.append(v)

    # 3. Compute Pearson correlation in chunks of 500 rows with matrix multiplication
    logger.info(
        "Computing pair-wise Pearson correlations across 8 dihedral transforms in chunks of %d (threshold=%.4f)...",
        chunk_size,
        dedup_threshold,
    )
    pairs = []
    V_ref = transforms[0]
    for i in range(0, N, chunk_size):
        i_end = min(i + chunk_size, N)
        A = V_ref[i:i_end]
        for j in range(i, N, chunk_size):
            j_end = min(j + chunk_size, N)
            max_corr = np.zeros((i_end - i, j_end - j), dtype=np.float32)
            for k in range(8):
                B_k = transforms[k][j:j_end]
                corr_k = np.dot(A, B_k.T)
                np.maximum(max_corr, corr_k, out=max_corr)
            if i == j:
                mask = np.triu(max_corr >= dedup_threshold, k=1)
            else:
                mask = max_corr >= dedup_threshold
            rows, cols = np.where(mask)
            for r, c in zip(rows, cols):
                pairs.append((i + r, j + c))

    logger.info("Found %d near-duplicate pair connections.", len(pairs))

    # 4. Build groups with scipy.sparse.csgraph.connected_components
    if pairs:
        r, c = zip(*pairs)
        data = np.ones(len(r) * 2, dtype=bool)
        rows = np.concatenate([r, c])
        cols = np.concatenate([c, r])
        adj = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    else:
        adj = sp.csr_matrix((N, N), dtype=bool)

    n_components, labels = connected_components(adj, directed=False)

    # 5. Safety: if a group has > 25 images, treat members as singletons and log warning
    final_group_ids = np.zeros(N, dtype=np.int32)
    next_group_id = 0
    component_members: Dict[int, List[int]] = {}
    for idx, comp in enumerate(labels):
        component_members.setdefault(comp, []).append(idx)

    for comp, members in component_members.items():
        if len(members) > 25:
            logger.warning(
                "Safety threshold triggered: Group %d has %d images (> 25 limit). "
                "Treating members as independent singletons to prevent large false-positive clustering.",
                comp,
                len(members),
            )
            for m in members:
                final_group_ids[m] = next_group_id
                next_group_id += 1
        else:
            for m in members:
                final_group_ids[m] = next_group_id
            next_group_id += 1

    group_sizes = pd.Series(final_group_ids).value_counts()
    dup_groups = group_sizes[group_sizes > 1]
    num_groups = len(group_sizes)
    images_inside_groups = int(dup_groups.sum()) if len(dup_groups) > 0 else 0
    largest_group = int(group_sizes.max()) if len(group_sizes) > 0 else 0
    num_dup_groups = len(dup_groups)

    return final_group_ids, num_groups, images_inside_groups, largest_group, num_dup_groups


def discover_dataset_images(cfg: AppConfig) -> pd.DataFrame:
    """Scan raw directory for images across classes, applying exclude_class_extensions & exclude_files."""
    raw_dir = cfg.data.raw_dir
    classes = cfg.data.classes
    exclude_set = {Path(f).as_posix() for f in cfg.data.exclude_files}
    records = []

    logger.info("Discovering images in %s across %d classes...", raw_dir, len(classes))
    total_excluded_by_ext = 0

    for class_idx, class_name in enumerate(classes):
        class_folder = raw_dir / class_name
        if not class_folder.exists():
            logger.warning("Class directory not found: %s", class_folder)
            continue

        raw_files = [
            p for p in class_folder.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]
        raw_files.sort(key=lambda p: str(p))

        # Check exclude_class_extensions for this class (case-insensitive)
        class_exclude_exts = {
            ext.lower() for ext in cfg.data.exclude_class_extensions.get(class_name, [])
        }

        file_paths = []
        excluded_ext_count = 0
        for p in raw_files:
            # Check extension exclusion
            if p.suffix.lower() in class_exclude_exts:
                excluded_ext_count += 1
                continue

            # Check exclude_files
            rel_raw = p.relative_to(raw_dir).as_posix()
            rel_cls = p.relative_to(class_folder).as_posix()
            if rel_raw in exclude_set or rel_cls in exclude_set or p.name in exclude_set:
                logger.info("Excluding file per config exclude_files: %s", p)
                continue

            file_paths.append(p)

        if excluded_ext_count > 0:
            logger.info(
                "Excluded %d file(s) for class '%s' matching extension(s) %s (Audit note: e.g. PreB non-smear .bmp cells from different source).",
                excluded_ext_count,
                class_name,
                sorted(list(class_exclude_exts)),
            )
            total_excluded_by_ext += excluded_ext_count

        logger.info("Retained %d images for class '%s'", len(file_paths), class_name)
        for p in file_paths:
            records.append({
                "original_path": str(p),
                "class_folder": class_name,
                "label_index": class_idx,
                "filename": p.name,
            })

    if total_excluded_by_ext > 0:
        logger.info("Total files excluded across all classes by exclude_class_extensions: %d", total_excluded_by_ext)

    if not records:
        return pd.DataFrame(columns=["original_path", "class_folder", "label_index", "filename"])

    return pd.DataFrame(records)


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
    print(f"{'Class Name':<15} | {'Train':^16} | {'Val':^16} | {'Test':^16} | {'Total':^7}")
    print("-" * 70)

    for idx, name in enumerate(classes):
        n_train = int((train_df["label_index"] == idx).sum()) if len(train_df) > 0 else 0
        n_val = int((val_df["label_index"] == idx).sum()) if len(val_df) > 0 else 0
        n_test = int((test_df["label_index"] == idx).sum()) if len(test_df) > 0 else 0
        n_total = n_train + n_val + n_test

        p_train = (n_train / len(train_df) * 100) if len(train_df) > 0 else 0.0
        p_val = (n_val / len(val_df) * 100) if len(val_df) > 0 else 0.0
        p_test = (n_test / len(test_df) * 100) if len(test_df) > 0 else 0.0

        print(
            f"{name:<15} | {n_train:5d} ({p_train:5.1f}%) | "
            f"{n_val:5d} ({p_val:5.1f}%) | "
            f"{n_test:5d} ({p_test:5.1f}%) | {n_total:7d}"
        )
        rows.append({
            "class_name": name,
            "label_index": idx,
            "train_count": n_train,
            "val_count": n_val,
            "test_count": n_test,
            "total_count": n_total,
        })

    print("-" * 70)
    print(
        f"{'TOTAL':<15} | {len(train_df):5d} (100.0%) | "
        f"{len(val_df):5d} (100.0%) | "
        f"{len(test_df):5d} (100.0%) | {total_samples:7d}"
    )
    print("=" * 70 + "\n")
    return pd.DataFrame(rows)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved split CSV (%d rows) -> %s", len(df), path)


def process_and_deduplicate(cfg: AppConfig) -> pd.DataFrame:
    """Discover dataset, extract thumbnails & sizes, run Pearson dedup, and apply safety/dropping."""
    df = discover_dataset_images(cfg)
    if len(df) == 0:
        return df

    # Load / compute 64x64 thumbnails and (w, h) sizes
    file_paths = [Path(p) for p in df["original_path"]]
    cache_path = cfg.data.processed_dir / "thumbs64.npz"

    thumbs64, sizes = load_or_compute_thumbnails(file_paths, cache_path)
    df["orig_width"] = sizes[:, 0]
    df["orig_height"] = sizes[:, 1]
    df["orig_area"] = df["orig_width"] * df["orig_height"]

    # Compute near-duplicate connected components
    N = len(thumbs64)
    gray64 = (
        0.2989 * thumbs64[..., 0].astype(np.float32)
        + 0.5870 * thumbs64[..., 1].astype(np.float32)
        + 0.1140 * thumbs64[..., 2].astype(np.float32)
    )
    gray32 = gray64.reshape(N, 32, 2, 32, 2).mean(axis=(2, 4))

    transforms = []
    base_variants = [gray32, np.flip(gray32, axis=2)]
    for base in base_variants:
        for k in range(4):
            rot = np.rot90(base, k, axes=(1, 2))
            v = rot.reshape(N, -1)
            v = v - v.mean(axis=1, keepdims=True)
            n = np.linalg.norm(v, axis=1, keepdims=True)
            v = np.where(n > 1e-8, v / np.maximum(n, 1e-8), 0.0)
            transforms.append(v)

    chunk_size = 500
    pairs = []
    V_ref = transforms[0]
    for i in range(0, N, chunk_size):
        i_end = min(i + chunk_size, N)
        A = V_ref[i:i_end]
        for j in range(i, N, chunk_size):
            j_end = min(j + chunk_size, N)
            max_corr = np.zeros((i_end - i, j_end - j), dtype=np.float32)
            for k in range(8):
                B_k = transforms[k][j:j_end]
                corr_k = np.dot(A, B_k.T)
                np.maximum(max_corr, corr_k, out=max_corr)
            if i == j:
                mask = np.triu(max_corr >= cfg.data.dedup_threshold, k=1)
            else:
                mask = max_corr >= cfg.data.dedup_threshold
            rows, cols = np.where(mask)
            for r, c in zip(rows, cols):
                pairs.append((i + r, j + c))

    if pairs:
        r, c = zip(*pairs)
        data = np.ones(len(r) * 2, dtype=bool)
        rows = np.concatenate([r, c])
        cols = np.concatenate([c, r])
        adj = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    else:
        adj = sp.csr_matrix((N, N), dtype=bool)

    n_components, labels = connected_components(adj, directed=False)

    # Organize members by connected component
    component_members: Dict[int, List[int]] = {}
    for idx, comp in enumerate(labels):
        component_members.setdefault(comp, []).append(idx)

    # Safety checks:
    # 1. If a group contains more than one class, drop ALL of its members and log them
    # 2. If a group has > 25 images, do not merge it (treat members as singletons) and log warning
    drop_indices_cross_class: Set[int] = set()
    final_group_ids = np.zeros(N, dtype=np.int32)
    next_group_id = 0

    for comp, members in component_members.items():
        member_classes = set(df["class_folder"].iloc[members].tolist())
        if len(member_classes) > 1:
            logger.warning(
                "Cross-class duplicate detected in component %d across classes %s (%d images). Dropping all members.",
                comp,
                member_classes,
                len(members),
            )
            for m in members:
                logger.warning(
                    "  Dropping cross-class duplicate member: %s (class: %s)",
                    df["original_path"].iloc[m],
                    df["class_folder"].iloc[m],
                )
                drop_indices_cross_class.add(m)
        elif len(members) > 25:
            logger.warning(
                "Safety threshold triggered: Group %d has %d images (> 25 limit). "
                "Treating members as independent singletons to prevent large false-positive clustering.",
                comp,
                len(members),
            )
            for m in members:
                final_group_ids[m] = next_group_id
                next_group_id += 1
        else:
            for m in members:
                final_group_ids[m] = next_group_id
            next_group_id += 1

    df["group_id"] = final_group_ids

    # Drop cross-class duplicate images
    if drop_indices_cross_class:
        logger.warning(
            "Dropping %d cross-class duplicate images across dataset.",
            len(drop_indices_cross_class),
        )
        df = df.drop(index=list(drop_indices_cross_class)).reset_index(drop=True)

    n_before_dedup = len(df)
    group_sizes = df["group_id"].value_counts()
    dup_groups = group_sizes[group_sizes > 1]

    print("\n" + "=" * 70)
    print("NEAR-DUPLICATE GROUPING SUMMARY (PEARSON CORRELATION)")
    print("=" * 70)
    print(f"Total images analyzed   : {n_before_dedup}")
    print(f"Total unique groups     : {len(group_sizes)}")
    print(f"Duplicate groups (>1)   : {len(dup_groups)}")
    print(f"Images inside groups    : {int(dup_groups.sum()) if len(dup_groups) > 0 else 0}")
    print(f"Largest group size      : {int(group_sizes.max()) if len(group_sizes) > 0 else 0}")
    print("=" * 70)

    # Drop duplicates if enabled: keep exactly one per group with largest orig_width*orig_height
    # Ties broken by first path alphabetically
    if cfg.data.drop_duplicates and len(df) > 0:
        kept_rows = []
        dropped_rows = []

        for gid, grp in df.groupby("group_id"):
            # Sort by area descending, then original_path ascending
            sorted_grp = grp.sort_values(
                by=["orig_area", "original_path"],
                ascending=[False, True],
            )
            kept_row = sorted_grp.iloc[0].to_dict()
            kept_rows.append(kept_row)

            if len(sorted_grp) > 1:
                for _, d_row in sorted_grp.iloc[1:].iterrows():
                    d_dict = d_row.to_dict()
                    d_dict["kept_path"] = kept_row["original_path"]
                    dropped_rows.append(d_dict)

        df_dropped = pd.DataFrame(dropped_rows)
        dropped_csv = cfg.data.splits_dir / "dropped_duplicates.csv"
        dropped_csv.parent.mkdir(parents=True, exist_ok=True)
        df_dropped.to_csv(dropped_csv, index=False)
        logger.info("Saved %d dropped duplicate records -> %s", len(df_dropped), dropped_csv)

        df = pd.DataFrame(kept_rows).reset_index(drop=True)
        n_after_dedup = len(df)

        print("\n" + "=" * 70)
        print("DUPLICATE REMOVAL SUMMARY (drop_duplicates=True)")
        print("=" * 70)
        print(f"Images before deduplication : {n_before_dedup}")
        print(f"Images after deduplication  : {n_after_dedup} (dropped {len(dropped_rows)} duplicates)")
        print(f"Dropped duplicate records   : {dropped_csv}")
        print("\nPer-Class Counts After Deduplication:")
        for cls_name in cfg.data.classes:
            cnt = int((df["class_folder"] == cls_name).sum())
            print(f"  {cls_name:<15}: {cnt}")
        print("=" * 70 + "\n")

    return df


def make_splits(
    cfg: AppConfig,
    force: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create or load reproducible stratified train / val / test splits.

    Uses Pearson correlation near-duplicate groups and StratifiedGroupKFold
    (7 folds, ~70/15/15) to guarantee near-duplicate images never leak across splits.
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

    logger.info("Processing dataset and computing near-duplicate groups (seed=%d)...", cfg.seed)
    df = process_and_deduplicate(cfg)

    empty_cols = ["original_path", "class_folder", "label_index", "filename", "orig_width", "orig_height", "orig_area", "group_id"]
    if len(df) == 0:
        logger.warning("No images found in raw_dir: %s. Empty splits created.", cfg.data.raw_dir)
        train_df = pd.DataFrame(columns=empty_cols)
        val_df = pd.DataFrame(columns=empty_cols)
        test_df = pd.DataFrame(columns=empty_cols)
        _write_csv(train_df, train_csv)
        _write_csv(val_df, val_csv)
        _write_csv(test_df, test_csv)
        return train_df, val_df, test_df

    # Use StratifiedGroupKFold with 7 folds (~70/15/15)
    min_groups_per_class = min(
        df.groupby("label_index")["group_id"].nunique()
    )
    n_splits = min(7, max(3, min_groups_per_class))

    if min_groups_per_class >= 3:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
        splits = list(sgkf.split(df, df["label_index"], df["group_id"]))
        # Fold 0 -> test (~15%), Fold 1 -> val (~15%), Folds 2..n -> train (~70%)
        test_idx = splits[0][1]
        val_idx = splits[1][1]
        train_idx = [i for i in range(len(df)) if i not in set(test_idx) and i not in set(val_idx)]

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
    else:
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

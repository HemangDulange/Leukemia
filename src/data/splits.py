"""
Stratified train / validation / test splits and class weight computation.

Generates ONE reproducible split (fixed seed 42) at file level and saves CSVs
to ``data/splits/{train,val,test}.csv``.  Subsequent runs reload the saved CSVs
to prevent any data leakage or reshuffling between validation and test sets.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.config import AppConfig
from src.data.pairing import build_pairs

logger = logging.getLogger(__name__)


def compute_class_weights(
    train_df: pd.DataFrame,
    num_classes: int = 4,
) -> Dict[int, float]:
    """Compute balanced class weights for imbalanced training distributions.

    Uses the standard inverse-frequency formula:
    w_j = N / (n_classes * N_j)

    Parameters
    ----------
    train_df:
        DataFrame containing ``label_index`` column for the training split.
    num_classes:
        Total number of classes.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping ``class_index -> class_weight``.
    """
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
    """Print a formatted summary table of class counts and percentages across splits.

    Parameters
    ----------
    train_df:
        Training split DataFrame.
    val_df:
        Validation split DataFrame.
    test_df:
        Test split DataFrame.
    classes:
        List of class folder names in canonical order.

    Returns
    -------
    pd.DataFrame
        Distribution summary DataFrame.
    """
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
    print(
        f"{'TOTAL':<15} | {len(train_df):5d} (100.0%) | "
        f"{len(val_df):5d} (100.0%) | {len(test_df):5d} (100.0%) | {total_samples:6d}"
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

    Ensures ONE fixed split across runs by persisting splits to CSVs.

    Parameters
    ----------
    cfg:
        Loaded application configuration.
    force:
        If ``True``, regenerates splits from scratch even if CSVs exist.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_df, val_df, test_df)``
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

    logger.info("Generating new stratified splits (seed=%d)...", cfg.seed)
    pairs = build_pairs(cfg)

    if len(pairs) == 0:
        logger.warning("No image pairs found. Empty split DataFrames returned.")
        empty_cols = ["original_path", "segmented_path", "class_folder", "label_index", "filename"]
        train_df = pd.DataFrame(columns=empty_cols)
        val_df = pd.DataFrame(columns=empty_cols)
        test_df = pd.DataFrame(columns=empty_cols)
        _write_csv(train_df, train_csv)
        _write_csv(val_df, val_csv)
        _write_csv(test_df, test_csv)
        return train_df, val_df, test_df

    val_frac = cfg.data.val_fraction
    test_frac = cfg.data.test_fraction

    # 1. Carve off stratified test split
    train_val, test_df = train_test_split(
        pairs,
        test_size=test_frac,
        stratify=pairs["label_index"],
        random_state=cfg.seed,
    )

    # 2. Split train_val into train and validation splits
    adjusted_val = val_frac / (1.0 - test_frac)
    train_df, val_df = train_test_split(
        train_val,
        test_size=adjusted_val,
        stratify=train_val["label_index"],
        random_state=cfg.seed,
    )

    # Save CSVs to disk for absolute reproducibility across runs
    _write_csv(train_df, train_csv)
    _write_csv(val_df, val_csv)
    _write_csv(test_df, test_csv)

    print_class_distribution(train_df, val_df, test_df, cfg.data.classes)
    return train_df, val_df, test_df

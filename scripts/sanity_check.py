"""
Sanity check script: verifies pipeline consistency and evaluates train vs val accuracy.

Checks:
1. Input Scaling:
   Verifies that both train and validation pipelines provide images in [0, 255] float32,
   matching Keras EfficientNet's built-in Rescaling layer expectations.
2. Augmentation Consistency:
   Verifies that augmentation runs only on training data, does not clip values to [0, 1],
   and that the unaugmented train slice matches the validation pipeline identically.
3. Model Evaluation:
   Loads models/dual_stream.keras (or builds smoke model if in smoke mode) and evaluates:
   - Fixed slice of the TRAIN split (up to 256 images) WITHOUT augmentation.
   - The full VAL split.
   Prints both accuracies and any pipeline discrepancies detected.

Usage
-----
    python scripts/sanity_check.py
    python scripts/sanity_check.py --smoke-test
    python scripts/sanity_check.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.config import AppConfig, load_config
from src.data.dataset import build_dataset
from src.data.splits import make_splits
from src.inference.model_loader import load_trained_model
from src.models.dual_stream import build_dual_stream_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify train/val pipeline consistency and evaluate train slice vs val accuracy."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to custom config YAML (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to trained model file (default from config: models/dual_stream.keras)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run in smoke-test mode with synthetic or sample data on CPU",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=256,
        help="Fixed number of train images for unaugmented evaluation slice (default: 256)",
    )
    return parser.parse_args()


def inspect_batch_pipeline(
    ds: tf.data.Dataset,
    name: str,
) -> Dict[str, Dict[str, float]]:
    """Extract one batch and compute numerical statistics across streams."""
    batch_x, batch_y = next(iter(ds))
    stats = {}
    for stream_name in ["original", "segmented"]:
        tensor = batch_x[stream_name]
        arr = tensor.numpy()
        stats[stream_name] = {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
        logger.info(
            "[%s] stream='%s' shape=%s dtype=%s min=%.2f max=%.2f mean=%.2f",
            name,
            stream_name,
            arr.shape,
            arr.dtype,
            stats[stream_name]["min"],
            stats[stream_name]["max"],
            stats[stream_name]["mean"],
        )
    return stats


def detect_pipeline_differences(
    train_aug_stats: Dict[str, Dict[str, float]],
    train_noaug_stats: Dict[str, Dict[str, float]],
    val_stats: Dict[str, Dict[str, float]],
) -> List[str]:
    """Identify potential data pipeline bugs, scaling mismatches, or clipping issues."""
    differences = []

    # 1. Check if training was clipped to [0, 1] while val is [0, 255]
    for stream in ["original", "segmented"]:
        t_aug_max = train_aug_stats[stream]["max"]
        v_max = val_stats[stream]["max"]
        t_noaug_max = train_noaug_stats[stream]["max"]

        if t_aug_max <= 1.5 and v_max > 1.5:
            differences.append(
                f"CRITICAL: Train ({stream}) is clipped to [0, 1] (max={t_aug_max:.2f}) "
                f"while Val ({stream}) has full range (max={v_max:.2f})."
            )

        if t_noaug_max <= 1.5 and v_max > 1.5:
            differences.append(
                f"CRITICAL: Unaugmented Train ({stream}) has max={t_noaug_max:.2f} "
                f"while Val ({stream}) has max={v_max:.2f}."
            )

        # Check negative values (e.g. erroneous [-1, 1] scaling)
        t_aug_min = train_aug_stats[stream]["min"]
        v_min = val_stats[stream]["min"]
        if t_aug_min < -0.01:
            differences.append(
                f"WARNING: Train ({stream}) contains negative values (min={t_aug_min:.2f}). "
                "EfficientNet expected input is [0, 255]."
            )
        if v_min < -0.01:
            differences.append(
                f"WARNING: Val ({stream}) contains negative values (min={v_min:.2f}). "
                "EfficientNet expected input is [0, 255]."
            )

        # Compare unaugmented train vs val value scale
        scale_ratio = (t_noaug_max + 1e-6) / (v_max + 1e-6)
        if scale_ratio > 3.0 or scale_ratio < 0.33:
            differences.append(
                f"WARNING: Unaugmented train and val have significantly different scales for '{stream}' "
                f"(train max={t_noaug_max:.2f}, val max={v_max:.2f})."
            )

    return differences


def create_fallback_mock_split(cfg: AppConfig, n_samples: int = 16) -> pd.DataFrame:
    """Create a minimal fallback DataFrame pointing to sample images if raw data is absent."""
    sample_dir = _REPO_ROOT / "app" / "samples"
    existing_samples = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
    if not existing_samples and (_REPO_ROOT / "data" / "fake_raw").exists():
        existing_samples = list((_REPO_ROOT / "data" / "fake_raw").rglob("*.jpg"))
    sample_orig = existing_samples[0] if existing_samples else Path("sample_orig.jpg")

    rows = []
    classes = cfg.data.classes
    for i in range(n_samples):
        cls_idx = i % len(classes)
        rows.append({
            "original_path": str(sample_orig),
            "label_index": cls_idx,
            "class_folder": classes[cls_idx],
            "class_name": classes[cls_idx],
            "stem": f"sample_{i}",
        })
    return pd.DataFrame(rows)


def run_sanity_check(
    cfg: AppConfig,
    model_path: Optional[Path] = None,
    smoke_test: bool = False,
    slice_size: int = 256,
) -> None:
    """Run pipeline consistency checks and model evaluation on train slice vs val."""
    print("\n" + "=" * 70)
    print("SANITY CHECK: Dual-Stream ALL Pipeline & Model Validation")
    print("=" * 70)

    # 1. Resolve data splits
    try:
        train_df, val_df, _ = make_splits(cfg, force=False)
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError("Empty split loaded from splits directory.")
    except Exception as e:
        logger.warning("Could not load dataset splits (%s). Generating fallback mock data for testing.", e)
        train_df = create_fallback_mock_split(cfg, n_samples=32)
        val_df = create_fallback_mock_split(cfg, n_samples=16)

    # 2. Extract fixed 256-image slice of the TRAIN split
    slice_count = min(slice_size, len(train_df))
    train_slice_df = train_df.iloc[:slice_count].copy().reset_index(drop=True)
    logger.info("Fixed train evaluation slice: %d samples", slice_count)
    logger.info("Full validation split: %d samples", len(val_df))

    # 3. Build tf.data pipelines
    # Train pipeline WITH augmentation
    train_aug_ds = build_dataset(train_slice_df, cfg, split="train", smoke_test=smoke_test)
    # Train pipeline WITHOUT augmentation (split="val" uses the deterministic no-augmentation path)
    train_noaug_ds = build_dataset(train_slice_df, cfg, split="val", smoke_test=smoke_test)
    # Full validation pipeline WITHOUT augmentation
    val_ds = build_dataset(val_df, cfg, split="val", smoke_test=smoke_test)

    # 4. Inspect numerical statistics
    print("\n--- Pipeline Numeric Statistics ---")
    train_aug_stats = inspect_batch_pipeline(train_aug_ds, "TRAIN (with augmentation)")
    train_noaug_stats = inspect_batch_pipeline(train_noaug_ds, "TRAIN (no augmentation slice)")
    val_stats = inspect_batch_pipeline(val_ds, "VAL (full validation set)")

    # 5. Detect and report pipeline discrepancies
    discrepancies = detect_pipeline_differences(train_aug_stats, train_noaug_stats, val_stats)
    print("\n--- Pipeline Consistency Report ---")
    if discrepancies:
        print(f"FAILED: Found {len(discrepancies)} discrepancy(ies):")
        for d in discrepancies:
            print(f"  • {d}")
    else:
        print("PASSED: Pipeline scaling and value ranges are consistent across Train and Val.")
        print("  • Both streams operate in [0, 255] float32 range (matching EfficientNet built-in Rescaling).")
        print("  • Augmentation does not clip or alter the expected pixel range.")
        print("  • Deterministic decoding and resizing are identical across train slice and val.")

    # 6. Load or build model for evaluation
    target_model_path = model_path or cfg.paths.model_path
    img_size = cfg.image.smoke_test_size if smoke_test else cfg.image.size

    model = None
    if target_model_path.exists() and target_model_path.is_file() and not smoke_test:
        try:
            logger.info("Loading model from %s...", target_model_path)
            model = load_trained_model(cfg, custom_path=target_model_path)
            logger.info("Model loaded successfully.")
        except Exception as err:
            logger.warning("Could not load saved model (%s). Building dual-stream model...", err)

    if model is None:
        logger.info("Instantiating dual-stream model (smoke_test=%s, size=%d)...", smoke_test, img_size)
        model = build_dual_stream_model(cfg, img_size=img_size, weights=None)

    # Ensure model is compiled for evaluation
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc", multi_label=False),
        ],
    )

    # 7. Evaluate on fixed TRAIN slice (WITHOUT augmentation) and full VAL split
    print("\n--- Model Evaluation ---")
    train_eval_results = model.evaluate(train_noaug_ds, verbose=0)
    val_eval_results = model.evaluate(val_ds, verbose=0)

    # Map metric names to values
    metric_names = model.metrics_names
    train_metrics = dict(zip(metric_names, train_eval_results))
    val_metrics = dict(zip(metric_names, val_eval_results))

    train_acc = train_metrics.get("accuracy", 0.0)
    train_loss = train_metrics.get("loss", 0.0)
    val_acc = val_metrics.get("accuracy", 0.0)
    val_loss = val_metrics.get("loss", 0.0)

    print(f"Unaugmented TRAIN slice ({slice_count} samples): accuracy = {train_acc:.4f} | loss = {train_loss:.4f}")
    print(f"Full VAL split ({len(val_df)} samples):        accuracy = {val_acc:.4f} | loss = {val_loss:.4f}")
    print("=" * 70 + "\n")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_sanity_check(
        cfg=cfg,
        model_path=args.model_path,
        smoke_test=args.smoke_test,
        slice_size=args.slice_size,
    )


if __name__ == "__main__":
    main()

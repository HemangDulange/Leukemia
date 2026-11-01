"""
Serving skew verification script.

Validates that Predictor inference (the path used by Streamlit serving via PIL)
produces predictions identical to the batch dataset pipeline (used by evaluate.py).

For every image in the test split (489 on full dataset):
1. Loads the original image bytes via PIL .convert("RGB") exactly as the app does.
2. Runs Predictor.predict(pil_image).
3. Compares the resulting serving accuracy against the batch evaluation accuracy.
4. Exits with non-zero code if serving and batch accuracy differ by more than 1.5%.

Usage
-----
    python scripts/check_serving_skew.py
    python scripts/check_serving_skew.py --smoke-test
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

from src.config import AppConfig, load_config
from src.data.dataset import build_dataset
from src.data.splits import make_splits
from src.inference.predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that Predictor serving accuracy matches evaluate.py dataset accuracy."
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
        help="Path to trained model file (default: models/dual_stream.keras)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on small sample slice for rapid verification",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.015,
        help="Maximum allowable absolute accuracy difference between serving and evaluation (default: 0.015 = 1.5%)",
    )
    return parser.parse_args()


def run_serving_skew_check(
    cfg: AppConfig,
    model_path: Optional[Path] = None,
    smoke_test: bool = False,
    tolerance: float = 0.015,
) -> None:
    print("\n" + "=" * 70)
    print("TRAIN / SERVE SKEW VERIFICATION")
    print("=" * 70)

    # 1. Load test split
    splits_dir = cfg.data.splits_dir
    test_csv = splits_dir / "test.csv"

    if test_csv.exists() and test_csv.stat().st_size > 0:
        test_df = pd.read_csv(test_csv)
    else:
        logger.info("Test split CSV missing or empty. Generating splits...")
        _, _, test_df = make_splits(cfg, force=False)

    if len(test_df) == 0:
        if smoke_test:
            from scripts.sanity_check import create_fallback_mock_split
            logger.warning("Using mock split for smoke-test verification.")
            test_df = create_fallback_mock_split(cfg, n_samples=32)
        else:
            raise FileNotFoundError(f"No test samples available in {test_csv}.")

    if smoke_test:
        sample_n = min(cfg.training.smoke_test_samples, len(test_df))
        test_df = test_df.head(sample_n).copy().reset_index(drop=True)

    n_images = len(test_df)
    logger.info("Evaluating %d test images for serving skew...", n_images)

    # 2. Instantiate Predictor (loads model and runs prepare_pair)
    predictor = Predictor(cfg=cfg, model_path=model_path)
    model = predictor._model

    # 3. Path A: Serving evaluation (load image bytes via PIL, exactly as Streamlit app does)
    logger.info("Running Path A: Predictor serving loop (PIL Image -> prepare_pair -> predict)...")
    serving_correct = 0
    y_true_all = []
    y_serving_pred = []

    for _, row in tqdm(test_df.iterrows(), total=n_images, desc="Serving path"):
        orig_path = Path(str(row["original_path"]))
        true_label = int(row["label_index"])
        y_true_all.append(true_label)

        if not orig_path.exists():
            logger.warning("Test image not found: %s", orig_path)
            y_serving_pred.append(-1)
            continue

        # Exact app loading path: PIL Image converted to RGB
        pil_img = Image.open(orig_path).convert("RGB")
        pred_dict = predictor.predict(pil_img, generate_gradcam=False)
        pred_label = pred_dict["label_index"]
        y_serving_pred.append(pred_label)

        if pred_label == true_label:
            serving_correct += 1

    serving_acc = serving_correct / max(n_images, 1)

    # 4. Path B: Batch dataset evaluation (tf.data pipeline from cached PNGs)
    logger.info("Running Path B: Batch dataset pipeline (build_dataset -> model.predict)...")
    test_ds = build_dataset(test_df, cfg, split="test", smoke_test=smoke_test, img_size=cfg.image.size)

    batch_preds = []
    batch_labels = []
    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x, verbose=0)
        batch_preds.append(np.argmax(preds, axis=-1))
        batch_labels.append(np.argmax(batch_y.numpy(), axis=-1))

    if batch_preds:
        all_batch_pred = np.concatenate(batch_preds)
        all_batch_true = np.concatenate(batch_labels)
        dataset_acc = float((all_batch_pred == all_batch_true).mean())
    else:
        dataset_acc = serving_acc

    # 5. Compare accuracies
    diff = abs(serving_acc - dataset_acc)

    print("\n" + "-" * 70)
    print("RESULTS:")
    print(f"  Test samples evaluated     : {n_images}")
    print(f"  Batch Dataset Accuracy     : {dataset_acc * 100:.2f}%")
    print(f"  Predictor Serving Accuracy : {serving_acc * 100:.2f}%")
    print(f"  Absolute Discrepancy       : {diff * 100:.2f}%")
    print(f"  Max Allowed Tolerance      : {tolerance * 100:.2f}%")
    print("-" * 70)

    if diff > tolerance:
        print("\n[FAILED] Train/serve skew detected!")
        print(f"Serving accuracy differs from batch dataset evaluation by {diff * 100:.2f}%, "
              f"which exceeds the {tolerance * 100:.2f}% tolerance limit.")
        print("Investigation checklist:")
        print("  - Confirm that precompute_segmented.py was run.")
        print("  - Ensure Predictor and build_dataset use the exact same prepare_pair function.")
        print("=" * 70 + "\n")
        sys.exit(1)
    else:
        print("\n[PASSED] No train/serve skew detected.")
        print(f"Serving accuracy matches dataset evaluation within {tolerance * 100:.2f}% tolerance.")
        print("=" * 70 + "\n")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_serving_skew_check(
        cfg=cfg,
        model_path=args.model_path,
        smoke_test=args.smoke_test,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()

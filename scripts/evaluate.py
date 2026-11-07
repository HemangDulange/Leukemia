"""
CLI script: evaluate the dual-stream ALL classifier on the test split.

Functions:
1. Load the frozen test split (data/splits/test.csv) and trained model (models/dual_stream.keras).
2. Compute comprehensive metrics: accuracy, weighted precision/recall/F1, and one-vs-rest AUC.
3. Export metrics to reports/metrics/evaluation.csv and JSON.
4. Export publication-quality 300 DPI figures to reports/figures/:
   - Normalized confusion matrix with cell counts and proportions
   - Multi-class ROC curves per class
   - Accuracy and loss curves (if history.json is found)
5. Optional ablation study (--ablation):
   Compares dual-stream against segmented-only and original-only streams.
6. Optional grayscale evaluation (--gray-test):
   Evaluates the model with both streams converted to grayscale (repeated to 3 channels)
   and prints accuracy and per-class recall next to normal RGB results.

Usage
-----
    python scripts/evaluate.py [--config PATH] [--smoke-test] [--ablation] [--gray-test]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import AppConfig, load_config
from src.data.dataset import build_dataset
from src.data.splits import make_splits
from src.evaluation.metrics import compute_evaluation_metrics, save_evaluation_csv, save_metrics
from src.evaluation.plots import (
    plot_ablation_comparison,
    plot_normalized_confusion_matrix,
    plot_roc_curves,
    plot_training_history,
)
from src.inference.model_loader import load_trained_model
from src.models.dual_stream import build_dual_stream_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def predict_dataset(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    ablation_mode: str = "dual_stream",
    force_grayscale: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect predictions across dataset under a given ablation or grayscale configuration.

    Parameters
    ----------
    model:
        Trained dual-stream Keras model.
    dataset:
        tf.data test dataset yielding ({"original": ..., "segmented": ...}, label).
    ablation_mode:
        One of {'dual_stream', 'original_only', 'segmented_only'}.
    force_grayscale:
        If True, converts both original and segmented inputs to grayscale (repeated across 3 channels).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (y_true, y_pred_proba)
    """
    y_true_list = []
    y_pred_list = []

    for batch_inputs, labels in dataset:
        orig = batch_inputs["original"]
        seg = batch_inputs["segmented"]

        if force_grayscale:
            orig = tf.repeat(tf.image.rgb_to_grayscale(orig), 3, axis=-1)
            seg = tf.repeat(tf.image.rgb_to_grayscale(seg), 3, axis=-1)

        if ablation_mode == "original_only":
            inputs = {"original": orig, "segmented": tf.zeros_like(seg)}
        elif ablation_mode == "segmented_only":
            inputs = {"original": tf.zeros_like(orig), "segmented": seg}
        else:
            inputs = {"original": orig, "segmented": seg}

        preds = model.predict(inputs, verbose=0)
        y_true_list.append(tf.argmax(labels, axis=1).numpy())
        y_pred_list.append(preds)

    y_true = np.concatenate(y_true_list)
    y_pred_proba = np.concatenate(y_pred_list)
    return y_true, y_pred_proba


def run_ablation_study(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: List[str],
    cfg: AppConfig,
) -> pd.DataFrame:
    """Evaluate segmented-only, original-only, and dual-stream configurations."""
    print("\n" + "=" * 65)
    print("RUNNING ABLATION STUDY: COMPARING INPUT STREAM CONTRIBUTIONS")
    print("=" * 65)

    configs = [
        ("Segmented Stream Only", "segmented_only"),
        ("Original Stream Only", "original_only"),
        ("Dual-Stream (Proposed)", "dual_stream"),
    ]

    rows = []
    for label, mode in configs:
        logger.info("Evaluating ablation configuration: %s...", label)
        y_true, y_prob = predict_dataset(model, test_ds, ablation_mode=mode)
        metrics = compute_evaluation_metrics(y_true, y_prob, class_names)
        rows.append(
            {
                "Configuration": label,
                "Accuracy": round(metrics["accuracy"], 4),
                "F1-Score": round(metrics["f1_weighted"], 4),
                "AUC": round(metrics["auc_macro_ovr"], 4),
                "Precision": round(metrics["precision_weighted"], 4),
                "Recall": round(metrics["recall_weighted"], 4),
            }
        )

    ablation_df = pd.DataFrame(rows)

    # Save ablation study CSV
    ablation_csv_path = cfg.paths.metrics_dir / "ablation_study.csv"
    ablation_csv_path.parent.mkdir(parents=True, exist_ok=True)
    ablation_df.to_csv(ablation_csv_path, index=False)
    logger.info("Saved ablation metrics -> %s", ablation_csv_path)

    # Save ablation comparison figure (300 DPI)
    ablation_fig_path = cfg.paths.figures_dir / "ablation_comparison.png"
    plot_ablation_comparison(ablation_df, ablation_fig_path, dpi=300)

    print("\n" + "=" * 65)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("=" * 65)
    print(ablation_df.to_string(index=False))
    print("=" * 65 + "\n")

    return ablation_df


def run_subgroups_evaluation(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    display_names: List[str],
    cfg: AppConfig,
) -> pd.DataFrame:
    """Evaluate accuracy and per-class recall across smear vs non-smear subgroups.

    Subgroups:
    (i) Whole test set
    (ii) Only smear images (.jpg files)
    (iii) Only non-smear images (AML .tiff and Normal .bmp)
    """
    y_pred = np.argmax(y_pred_proba, axis=1)
    paths = test_df["original_path"].astype(str).str.lower()

    # Define subgroup boolean masks
    is_smear = paths.str.endswith((".jpg", ".jpeg"))
    is_nonsmear = paths.str.endswith((".tif", ".tiff", ".bmp"))

    subgroup_defs = [
        ("Whole Test Set", np.ones(len(test_df), dtype=bool)),
        ("Smear Images (.jpg)", is_smear.values),
        ("Non-Smear Images (AML .tiff & Normal .bmp)", is_nonsmear.values),
    ]

    rows = []
    print("\n" + "=" * 80)
    print("SUBGROUP PERFORMANCE EVALUATION (--subgroups)")
    print("=" * 80)

    for label, mask in subgroup_defs:
        count = int(np.sum(mask))
        if count == 0:
            row = {"subgroup": label, "image_count": 0, "accuracy": None}
            for name in display_names:
                row[f"{name}_recall"] = None
            rows.append(row)
            continue

        sub_true = y_true[mask]
        sub_pred = y_pred[mask]
        acc = float(np.mean(sub_true == sub_pred))

        row = {"subgroup": label, "image_count": count, "accuracy": round(acc, 4)}

        # Per-class recall
        for idx, name in enumerate(display_names):
            cls_mask = sub_true == idx
            if np.sum(cls_mask) > 0:
                rec = float(np.mean(sub_pred[cls_mask] == idx))
                row[f"{name}_recall"] = round(rec, 4)
            else:
                row[f"{name}_recall"] = None

        rows.append(row)

    subgroups_df = pd.DataFrame(rows)

    print(subgroups_df.to_string(index=False))
    print("=" * 80 + "\n")

    csv_path = cfg.paths.metrics_dir / "subgroups.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    subgroups_df.to_csv(csv_path, index=False)
    logger.info("Saved subgroup metrics -> %s", csv_path)

    return subgroups_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the dual-stream model on the test split."
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run evaluation on smoke test subset (128px)",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study comparing segmented-only, original-only, and dual-stream",
    )
    parser.add_argument(
        "--gray-test",
        action="store_true",
        help="Evaluate test split with both streams converted to grayscale (repeated to 3 channels)",
    )
    parser.add_argument(
        "--subgroups",
        action="store_true",
        help="Evaluate accuracy and per-class recall on whole test set, smear images (.jpg), and non-smear images",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # 1. Load canonical class names
    if cfg.paths.class_names_file.exists():
        with cfg.paths.class_names_file.open(encoding="utf-8") as fh:
            class_meta = json.load(fh)
        display_names = list(class_meta["index_to_display"].values())
    else:
        display_names = cfg.data.classes

    # 2. Load trained model or instantiate smoke-test model
    model_path = cfg.paths.model_path
    img_size = cfg.image.smoke_test_size if args.smoke_test else cfg.image.size

    if model_path.exists():
        logger.info("Loading model from %s...", model_path)
        model = load_trained_model(cfg, custom_path=model_path)
        if hasattr(model, "input_shape") and isinstance(model.input_shape, list) and model.input_shape[0]:
            img_size = model.input_shape[0][1] or img_size
    elif args.smoke_test:
        logger.warning(
            "Model file not found at %s. Smoke-test mode: building untrained dual-stream model.",
            model_path,
        )
        model = build_dual_stream_model(cfg, img_size=img_size, weights=None)
    else:
        logger.error(
            "Trained model not found at '%s'. Train the model first with `python scripts/train.py`.",
            model_path,
        )
        sys.exit(1)

    # 3. Load fixed test dataset
    _, _, test_df = make_splits(cfg)
    if len(test_df) == 0:
        if args.smoke_test:
            from scripts.sanity_check import create_fallback_mock_split
            logger.warning("Test split has 0 rows. Using fallback samples for smoke-test.")
            test_df = create_fallback_mock_split(cfg, n_samples=16)
        else:
            logger.error("Test split in %s has 0 rows. Prepare data first.", cfg.data.splits_dir)
            sys.exit(1)
    elif args.smoke_test:
        test_df = test_df.head(min(16, len(test_df)))

    test_ds = build_dataset(test_df, cfg, split="test", smoke_test=args.smoke_test, img_size=img_size)

    # 4. Predict on full dual-stream model
    logger.info("Generating predictions on test set (%d samples)...", len(test_df))
    y_true, y_pred_proba = predict_dataset(model, test_ds, ablation_mode="dual_stream")

    # 5. Compute full evaluation metrics
    metrics = compute_evaluation_metrics(y_true, y_pred_proba, display_names)

    # Save metrics JSON & CSV
    json_path = cfg.paths.metrics_dir / "evaluation.json"
    save_metrics(metrics, json_path)

    csv_path = cfg.paths.metrics_dir / "evaluation.csv"
    save_evaluation_csv(metrics, csv_path)

    # 6. Save plots at 300 DPI
    fig_dir = cfg.paths.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Normalized Confusion Matrix (300 DPI)
    plot_normalized_confusion_matrix(
        metrics["confusion_matrix_normalized"],
        metrics["confusion_matrix_raw"],
        display_names,
        fig_dir / "confusion_matrix.png",
        dpi=300,
    )

    # ROC Curves per class (300 DPI)
    plot_roc_curves(
        metrics["roc_curves"],
        fig_dir / "roc_curves.png",
        dpi=300,
    )

    # Training history curves if available (300 DPI)
    hist_json = cfg.paths.metrics_dir / "history.json"
    if hist_json.exists():
        with hist_json.open(encoding="utf-8") as fh:
            hist_data = json.load(fh)
        plot_training_history(hist_data, fig_dir / "training_history.png", dpi=300)

    # 7. Print formatted evaluation summary to console
    print("\n" + "=" * 60)
    print("TEST EVALUATION PERFORMANCE METRICS")
    print("=" * 60)
    print(f"Accuracy                 : {metrics['accuracy']:.4f}")
    print(f"Precision (Weighted)     : {metrics['precision_weighted']:.4f}")
    print(f"Recall (Weighted)        : {metrics['recall_weighted']:.4f}")
    print(f"F1 Score (Weighted)      : {metrics['f1_weighted']:.4f}")
    print(f"AUC (Macro OvR)          : {metrics['auc_macro_ovr']:.4f}")
    print(f"AUC (Weighted OvR)       : {metrics['auc_weighted_ovr']:.4f}")
    print("-" * 60)
    print("Per-Class Report:")
    report_df = pd.DataFrame(metrics["classification_report"]).transpose()
    print(report_df.to_string())
    print("=" * 60 + "\n")

    # 8. Grayscale Robustness Evaluation (--gray-test)
    if args.gray_test:
        logger.info("Executing grayscale evaluation on test split (both streams in grayscale)...")
        y_true_gray, y_pred_proba_gray = predict_dataset(
            model,
            test_ds,
            ablation_mode="dual_stream",
            force_grayscale=True,
        )
        gray_metrics = compute_evaluation_metrics(y_true_gray, y_pred_proba_gray, display_names)

        # Print side-by-side comparison table
        print("\n" + "=" * 70)
        print("GRAYSCALE ROBUSTNESS EVALUATION COMPARISON (--gray-test)")
        print("=" * 70)
        print(f"{'Metric / Subtype':<25} | {'Normal (RGB)':^18} | {'Grayscale Test':^18}")
        print("-" * 70)
        print(
            f"{'Overall Accuracy':<25} | {metrics['accuracy'] * 100:16.2f}% | "
            f"{gray_metrics['accuracy'] * 100:16.2f}%"
        )
        print("-" * 70)
        normal_report = metrics["classification_report"]
        gray_report = gray_metrics["classification_report"]
        for name in display_names:
            norm_rec = normal_report.get(name, {}).get("recall", 0.0) * 100
            gray_rec = gray_report.get(name, {}).get("recall", 0.0) * 100
            print(f"{name + ' Recall':<25} | {norm_rec:16.2f}% | {gray_rec:16.2f}%")
        print("=" * 70 + "\n")

    # 9. Optional Subgroups Evaluation (--subgroups)
    if args.subgroups:
        run_subgroups_evaluation(test_df, y_true, y_pred_proba, display_names, cfg)

    # 10. Optional Ablation Study
    if args.ablation:
        run_ablation_study(model, test_ds, display_names, cfg)


if __name__ == "__main__":
    main()

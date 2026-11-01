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
   Compares dual-stream against segmented-only and original-only streams to prove
   the multi-stream synergistic benefit.

Usage
-----
    python scripts/evaluate.py [--config PATH] [--smoke-test] [--ablation]
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect predictions across dataset under a given ablation configuration.

    Parameters
    ----------
    model:
        Trained dual-stream Keras model.
    dataset:
        tf.data test dataset yielding ({"original": ..., "segmented": ...}, label).
    ablation_mode:
        One of {'dual_stream', 'original_only', 'segmented_only'}.

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

        if ablation_mode == "original_only":
            # Mask out segmented stream with zeros
            inputs = {"original": orig, "segmented": tf.zeros_like(seg)}
        elif ablation_mode == "segmented_only":
            # Mask out original stream with zeros
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

    # 2. Load trained model
    model_path = cfg.paths.model_path
    if not model_path.exists():
        logger.error(
            "Trained model not found at '%s'. Train the model first with `python scripts/train.py`.",
            model_path,
        )
        sys.exit(1)

    logger.info("Loading model from %s...", model_path)
    model = tf.keras.models.load_model(str(model_path))

    # 3. Load fixed test dataset
    _, _, test_df = make_splits(cfg)
    if len(test_df) == 0:
        logger.error("Test split in %s has 0 rows. Prepare data first.", cfg.data.splits_dir)
        sys.exit(1)

    test_ds = build_dataset(test_df, cfg, split="test", smoke_test=args.smoke_test)

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

    # 8. Optional Ablation Study
    if args.ablation:
        run_ablation_study(model, test_ds, display_names, cfg)


if __name__ == "__main__":
    main()

"""Unit tests for evaluation metrics and plotting helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluation.metrics import compute_evaluation_metrics, save_evaluation_csv
from src.evaluation.plots import (
    plot_ablation_comparison,
    plot_normalized_confusion_matrix,
    plot_roc_curves,
    plot_training_history,
)


@pytest.fixture()
def class_names() -> list[str]:
    return ["Hematogones", "Early Pre-B ALL", "Pre-B ALL", "Pro-B ALL"]


@pytest.fixture()
def dummy_predictions():
    # 4 classes, 20 samples
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    # create high probability for true class with some noise
    y_prob = np.zeros((20, 4))
    for i, t in enumerate(y_true):
        y_prob[i, t] = 0.8
        y_prob[i] += np.random.rand(4) * 0.1
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    return y_true, y_prob


class TestEvaluationMetrics:
    def test_compute_metrics_keys(self, dummy_predictions, class_names):
        y_true, y_prob = dummy_predictions
        metrics = compute_evaluation_metrics(y_true, y_prob, class_names)

        assert "accuracy" in metrics
        assert "precision_weighted" in metrics
        assert "recall_weighted" in metrics
        assert "f1_weighted" in metrics
        assert "auc_macro_ovr" in metrics
        assert "classification_report" in metrics
        assert "confusion_matrix_normalized" in metrics
        assert "roc_curves" in metrics

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["auc_macro_ovr"] <= 1.0

    def test_save_evaluation_csv(self, dummy_predictions, class_names, tmp_path):
        y_true, y_prob = dummy_predictions
        metrics = compute_evaluation_metrics(y_true, y_prob, class_names)

        csv_file = tmp_path / "evaluation.csv"
        df = save_evaluation_csv(metrics, csv_file)

        assert csv_file.exists()
        assert "Metric" in df.columns
        assert "Value" in df.columns
        assert len(df) >= 6


class TestEvaluationPlots:
    def test_plot_training_history(self, tmp_path):
        history = {
            "accuracy": [0.7, 0.8, 0.9],
            "val_accuracy": [0.65, 0.75, 0.85],
            "loss": [0.6, 0.4, 0.2],
            "val_loss": [0.7, 0.5, 0.3],
        }
        fig_path = tmp_path / "history.png"
        plot_training_history(history, fig_path, dpi=100)
        assert fig_path.exists()

    def test_plot_normalized_confusion_matrix(self, dummy_predictions, class_names, tmp_path):
        y_true, y_prob = dummy_predictions
        metrics = compute_evaluation_metrics(y_true, y_prob, class_names)

        fig_path = tmp_path / "cm.png"
        plot_normalized_confusion_matrix(
            metrics["confusion_matrix_normalized"],
            metrics["confusion_matrix_raw"],
            class_names,
            fig_path,
            dpi=100,
        )
        assert fig_path.exists()

    def test_plot_roc_curves(self, dummy_predictions, class_names, tmp_path):
        y_true, y_prob = dummy_predictions
        metrics = compute_evaluation_metrics(y_true, y_prob, class_names)

        fig_path = tmp_path / "roc.png"
        plot_roc_curves(metrics["roc_curves"], fig_path, dpi=100)
        assert fig_path.exists()

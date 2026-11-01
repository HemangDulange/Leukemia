"""
Evaluation metrics computation for the ALL dual-stream classifier.

Computes:
- Overall Accuracy
- Weighted Precision, Recall, and F1 Score
- One-vs-Rest (OvR) Macro and Weighted AUC
- Per-class Precision, Recall, F1, and Support
- Normalized and unnormalized Confusion Matrices
- Per-class ROC curves (FPR, TPR, AUC)
- Export to reports/metrics/evaluation.csv
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """Compute comprehensive classification performance metrics.

    Parameters
    ----------
    y_true:
        Integer ground-truth labels of shape ``(N,)``.
    y_pred_proba:
        Softmax probability array of shape ``(N, num_classes)``.
    class_names:
        List of display class names matching the probability columns.

    Returns
    -------
    dict
        Comprehensive metrics dictionary.
    """
    y_pred = np.argmax(y_pred_proba, axis=1)
    num_classes = len(class_names)

    # 1. Scalar summary metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # 2. Multi-class One-vs-Rest AUC
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    if num_classes == 2:
        y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])

    try:
        auc_macro = float(roc_auc_score(y_true_bin, y_pred_proba, average="macro", multi_class="ovr"))
        auc_weighted = float(roc_auc_score(y_true_bin, y_pred_proba, average="weighted", multi_class="ovr"))
    except ValueError as e:
        logger.warning("Could not compute multi-class AUC: %s", e)
        auc_macro = float("nan")
        auc_weighted = float("nan")

    # 3. Per-class ROC curves
    roc_curves: Dict[str, Dict[str, Any]] = {}
    for i, cls_name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            class_auc = float(np.trapz(tpr, fpr)) if len(fpr) > 1 else float("nan")
            roc_curves[cls_name] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": class_auc,
            }
        except Exception:
            roc_curves[cls_name] = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": float("nan")}

    # 4. Classification report & Confusion matrices
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    return {
        "accuracy": acc,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "f1_weighted": f1,
        "auc_macro_ovr": auc_macro,
        "auc_weighted_ovr": auc_weighted,
        "classification_report": report,
        "confusion_matrix_raw": cm_raw.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "roc_curves": roc_curves,
        "class_names": class_names,
    }


def save_evaluation_csv(metrics: Dict[str, Any], path: Path) -> pd.DataFrame:
    """Save overall metrics and per-class performance to CSV.

    Parameters
    ----------
    metrics:
        Output dict from :func:`compute_evaluation_metrics`.
    path:
        Destination CSV file path.

    Returns
    -------
    pd.DataFrame
        Written DataFrame.
    """
    rows = [
        {"Category": "Overall", "Metric": "Accuracy", "Value": round(metrics["accuracy"], 4)},
        {"Category": "Overall", "Metric": "Precision (Weighted)", "Value": round(metrics["precision_weighted"], 4)},
        {"Category": "Overall", "Metric": "Recall (Weighted)", "Value": round(metrics["recall_weighted"], 4)},
        {"Category": "Overall", "Metric": "F1 Score (Weighted)", "Value": round(metrics["f1_weighted"], 4)},
        {"Category": "Overall", "Metric": "AUC (Macro OvR)", "Value": round(metrics["auc_macro_ovr"], 4)},
        {"Category": "Overall", "Metric": "AUC (Weighted OvR)", "Value": round(metrics["auc_weighted_ovr"], 4)},
    ]

    report = metrics.get("classification_report", {})
    class_names = metrics.get("class_names", [])

    for cls in class_names:
        if cls in report:
            cls_data = report[cls]
            rows.append({"Category": f"Class: {cls}", "Metric": "Precision", "Value": round(cls_data.get("precision", 0.0), 4)})
            rows.append({"Category": f"Class: {cls}", "Metric": "Recall", "Value": round(cls_data.get("recall", 0.0), 4)})
            rows.append({"Category": f"Class: {cls}", "Metric": "F1-Score", "Value": round(cls_data.get("f1-score", 0.0), 4)})
            rows.append({"Category": f"Class: {cls}", "Metric": "Support", "Value": int(cls_data.get("support", 0))})

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved evaluation metrics CSV -> %s", path)
    return df


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """Compatibility wrapper returning metrics dictionary."""
    return compute_evaluation_metrics(y_true, y_pred_proba, class_names)


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    """Save metrics to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved -> %s", path)

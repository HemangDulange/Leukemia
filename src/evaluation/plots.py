"""
Evaluation visualisation helpers for publishing and reports.

All figures are generated using publication-quality formatting and saved
at 300 DPI to ``reports/figures/``:
1. Training and validation Accuracy and Loss curves
2. Normalized Confusion Matrix with dual percentage / count cell annotations
3. Per-class One-vs-Rest ROC curves
4. Ablation study comparison bar chart (Segmented-only vs. Original-only vs. Dual-stream)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_training_history(
    history_dict: Dict[str, List[float]],
    out_path: Path,
    dpi: int = 300,
) -> plt.Figure:
    """Plot and save training and validation accuracy / loss curves at 300 DPI.

    Parameters
    ----------
    history_dict:
        History dictionary containing keys: 'accuracy', 'val_accuracy', 'loss', 'val_loss'.
    out_path:
        Target file path for PNG export.
    dpi:
        Resolution for saved figure (default 300).

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=dpi)

    epochs = range(1, len(history_dict.get("accuracy", [])) + 1)

    # Accuracy subplot
    ax_acc = axes[0]
    if "accuracy" in history_dict:
        ax_acc.plot(epochs, history_dict["accuracy"], "b-o", linewidth=2, markersize=4, label="Train Accuracy")
    if "val_accuracy" in history_dict:
        ax_acc.plot(epochs, history_dict["val_accuracy"], "r--s", linewidth=2, markersize=4, label="Val Accuracy")
    ax_acc.set_title("Training and Validation Accuracy", fontsize=13, fontweight="bold", pad=10)
    ax_acc.set_xlabel("Epoch", fontsize=11)
    ax_acc.set_ylabel("Accuracy", fontsize=11)
    ax_acc.grid(True, linestyle=":", alpha=0.6)
    ax_acc.legend(fontsize=10, loc="lower right")
    ax_acc.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Loss subplot
    ax_loss = axes[1]
    if "loss" in history_dict:
        ax_loss.plot(epochs, history_dict["loss"], "b-o", linewidth=2, markersize=4, label="Train Loss")
    if "val_loss" in history_dict:
        ax_loss.plot(epochs, history_dict["val_loss"], "r--s", linewidth=2, markersize=4, label="Val Loss")
    ax_loss.set_title("Training and Validation Loss", fontsize=13, fontweight="bold", pad=10)
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Loss", fontsize=11)
    ax_loss.grid(True, linestyle=":", alpha=0.6)
    ax_loss.legend(fontsize=10, loc="upper right")
    ax_loss.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved training history plot -> %s", out_path)
    plt.close(fig)
    return fig


def plot_normalized_confusion_matrix(
    cm_norm: np.ndarray | List[List[float]],
    cm_raw: np.ndarray | List[List[int]],
    class_names: List[str],
    out_path: Path,
    title: str = "Normalized Confusion Matrix",
    dpi: int = 300,
) -> plt.Figure:
    """Plot and save normalized confusion matrix heatmap at 300 DPI.

    Parameters
    ----------
    cm_norm:
        Normalized confusion matrix (values in [0, 1]).
    cm_raw:
        Raw count confusion matrix.
    class_names:
        Ordered list of display class names.
    out_path:
        Target file path for PNG export.
    dpi:
        Resolution for saved figure (default 300).
    """
    cm_norm_arr = np.asarray(cm_norm, dtype=np.float32)
    cm_raw_arr = np.asarray(cm_raw, dtype=np.int32)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi)
    im = ax.imshow(cm_norm_arr, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Normalized Proportion", rotation=-90, va="bottom", fontsize=11)

    n_classes = len(class_names)
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # Annotate each cell with normalized value and raw count
    thresh = 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            norm_val = cm_norm_arr[i, j]
            raw_val = cm_raw_arr[i, j]
            color = "white" if norm_val > thresh else "black"
            text = f"{norm_val:.2f}\n(n={raw_val})"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10, fontweight="medium")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved normalized confusion matrix -> %s", out_path)
    plt.close(fig)
    return fig


def plot_roc_curves(
    roc_curves: Dict[str, Dict[str, Any]],
    out_path: Path,
    title: str = "One-vs-Rest ROC Curves per Class",
    dpi: int = 300,
) -> plt.Figure:
    """Plot and save multi-class One-vs-Rest ROC curves at 300 DPI.

    Parameters
    ----------
    roc_curves:
        Dictionary mapping class_name -> {'fpr': [...], 'tpr': [...], 'auc': float}.
    out_path:
        Target file path for PNG export.
    dpi:
        Resolution for saved figure (default 300).
    """
    fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi)

    colors = ["#2B6CB0", "#38A169", "#D69E2E", "#E53E3E", "#805AD5"]

    for idx, (cls_name, curve_data) in enumerate(roc_curves.items()):
        fpr = curve_data.get("fpr", [])
        tpr = curve_data.get("tpr", [])
        auc_val = curve_data.get("auc", float("nan"))
        color = colors[idx % len(colors)]
        label = f"{cls_name} (AUC = {auc_val:.4f})" if not np.isnan(auc_val) else cls_name
        ax.plot(fpr, tpr, color=color, linewidth=2.2, label=label)

    # Reference diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.7, label="Chance Baseline")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, labelpad=8)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved ROC curves plot -> %s", out_path)
    plt.close(fig)
    return fig


def plot_ablation_comparison(
    ablation_df: pd.DataFrame,
    out_path: Path,
    title: str = "Ablation Study: Dual-Stream vs. Single Streams",
    dpi: int = 300,
) -> plt.Figure:
    """Plot and save grouped bar chart demonstrating dual-stream performance gain.

    Parameters
    ----------
    ablation_df:
        DataFrame with columns: ``Configuration``, ``Accuracy``, ``F1-Score``, ``AUC``.
    out_path:
        Target file path for PNG export.
    dpi:
        Resolution for saved figure (default 300).
    """
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=dpi)

    configs = ablation_df["Configuration"].tolist()
    x = np.arange(len(configs))
    bar_width = 0.25

    metrics = [
        ("Accuracy", "#3182CE"),
        ("F1-Score", "#38A169"),
        ("AUC", "#DD6B20"),
    ]

    for idx, (metric_name, col) in enumerate(metrics):
        if metric_name in ablation_df.columns:
            vals = ablation_df[metric_name].tolist()
            pos = x + (idx - 1) * bar_width
            bars = ax.bar(pos, vals, width=bar_width, label=metric_name, color=col, alpha=0.9, edgecolor="black", linewidth=0.8)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=11, fontweight="medium")
    ax.set_ylim([0, 1.15])
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved ablation comparison plot -> %s", out_path)
    plt.close(fig)
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    out_path: Path,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Backward compatibility wrapper."""
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    return plot_normalized_confusion_matrix(cm_norm, cm, class_names, out_path, title=title, dpi=300)

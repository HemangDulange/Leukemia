"""
Two-phase training pipeline for the dual-stream ALL classifier.

Phase 1:
- Train the classification head while both feature-extraction backbones are
  completely frozen. Learning rate: 1e-3 (0.001).
Phase 2:
- Unfreeze the top 30 layers of each backbone for fine-tuning with a reduced
  learning rate: 1e-5 (0.00001).

Callbacks:
- EarlyStopping (monitors val_accuracy, restores best weights)
- ReduceLROnPlateau (monitors val_accuracy)
- ModelCheckpoint (monitors val_accuracy, saves best model to models/dual_stream.keras)

Artifacts saved:
- models/dual_stream.keras (best trained model)
- models/class_names.json (canonical class mapping)
- reports/metrics/history.json (combined phase 1 + 2 metrics history)
- reports/figures/training_history.png (accuracy and loss curves)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.config import AppConfig
from src.data.dataset import build_dataset
from src.data.splits import compute_class_weights
from src.evaluation.plots import plot_training_history
from src.models.dual_stream import build_dual_stream_model, unfreeze_backbone_layers
from src.training.callbacks import build_callbacks

logger = logging.getLogger(__name__)


def save_class_names(cfg: AppConfig) -> None:
    """Ensure canonical class mapping is written to models/class_names.json."""
    classes = cfg.data.classes
    display_mapping = {
        "Benign": "Hematogones",
        "Early": "Early Pre-B ALL",
        "Pre": "Pre-B ALL",
        "Pre-B": "Pre-B ALL",
        "Pro": "Pro-B ALL",
    }
    class_meta = {
        "class_order": classes,
        "display_names": {c: display_mapping.get(c, c) for c in classes},
        "index_to_folder": {str(i): c for i, c in enumerate(classes)},
        "index_to_display": {str(i): display_mapping.get(c, c) for i, c in enumerate(classes)},
    }
    out_file = cfg.paths.class_names_file
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(class_meta, fh, indent=2)
    logger.info("Saved canonical class names -> %s", out_file)


def merge_histories(h1: Dict[str, List[float]], h2: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Concatenate two sequential training history dictionaries."""
    merged: Dict[str, List[float]] = {}
    all_keys = set(h1.keys()) | set(h2.keys())
    for k in all_keys:
        v1 = [float(x) for x in h1.get(k, [])]
        v2 = [float(x) for x in h2.get(k, [])]
        merged[k] = v1 + v2
    return merged


def run_training(
    cfg: AppConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    smoke_test: bool = False,
) -> keras.Model:
    """Execute two-phase training and save model and metrics.

    Parameters
    ----------
    cfg:
        Application configuration.
    train_df:
        DataFrame containing training split file paths and labels.
    val_df:
        DataFrame containing validation split file paths and labels.
    smoke_test:
        If True, runs 1 epoch at 128px on a tiny subset without saving heavy weights.

    Returns
    -------
    keras.Model
        The trained dual-stream model.
    """
    img_size = cfg.image.smoke_test_size if smoke_test else cfg.image.size
    best_model_path = cfg.paths.model_path

    # Ensure class names json is up to date
    save_class_names(cfg)

    # 1. Build tf.data datasets
    train_ds = build_dataset(train_df, cfg, split="train", smoke_test=smoke_test)
    val_ds = build_dataset(val_df, cfg, split="val", smoke_test=smoke_test)

    # 2. Compute class weights to counter subtype class imbalance
    class_weights = None
    if len(train_df) > 0 and len(train_df["label_index"].unique()) > 1:
        class_weights = compute_class_weights(train_df, num_classes=cfg.model.num_classes)

    # =========================================================================
    # PHASE 1: Train classification head with backbones frozen at lr 1e-3
    # =========================================================================
    p1_epochs = 1 if smoke_test else cfg.training.phase1_epochs
    p1_lr = cfg.training.phase1_lr

    print("\n" + "=" * 65)
    print(f"PHASE 1: Training Classification Head (Backbones Frozen)")
    print(f"Learning rate: {p1_lr} | Epochs: {p1_epochs} | Input size: {img_size}px")
    print("=" * 65)

    model = build_dual_stream_model(
        cfg=cfg,
        img_size=img_size,
        unfreeze_top_n=0,
        weights="imagenet" if not smoke_test else None,
    )

    # Re-compile explicitly with Phase 1 learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=p1_lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=cfg.training.label_smoothing),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc", multi_label=False),
        ],
    )

    callbacks_p1 = build_callbacks(
        cfg,
        smoke_test=smoke_test,
        checkpoint_path=best_model_path,
        monitor="val_accuracy",
        mode="max",
    )

    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=p1_epochs,
        callbacks=callbacks_p1,
        class_weight=class_weights,
        verbose=1,
    )
    combined_history = {k: [float(x) for x in v] for k, v in h1.history.items()}

    # =========================================================================
    # PHASE 2: Unfreeze top 30 backbone layers at lr 1e-5
    # =========================================================================
    if cfg.training.phase2_epochs > 0:
        p2_epochs = 1 if smoke_test else cfg.training.phase2_epochs
        p2_lr = cfg.training.phase2_lr
        unfreeze_n = cfg.training.unfreeze_top_n

        print("\n" + "=" * 65)
        print(f"PHASE 2: Fine-Tuning Top {unfreeze_n} Backbone Layers")
        print(f"Learning rate: {p2_lr} | Epochs: {p2_epochs}")
        print("=" * 65)

        # Unfreeze top N layers of both backbones & compile with lr 1e-5 and fresh metric instances
        model = unfreeze_backbone_layers(
            model,
            unfreeze_top_n=unfreeze_n,
            learning_rate=p2_lr,
            label_smoothing=cfg.training.label_smoothing,
        )

        callbacks_p2 = build_callbacks(
            cfg,
            smoke_test=smoke_test,
            checkpoint_path=best_model_path,
            monitor="val_accuracy",
            mode="max",
        )

        h2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=p2_epochs,
            callbacks=callbacks_p2,
            class_weight=class_weights,
            verbose=1,
        )
        combined_history = merge_histories(combined_history, h2.history)

    # 3. Save combined history JSON
    history_path = cfg.paths.metrics_dir / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(combined_history, fh, indent=2)
    logger.info("Saved training history -> %s", history_path)

    # 4. Save training history curve plot
    if "accuracy" in combined_history:
        plot_training_history(
            combined_history,
            cfg.paths.figures_dir / "training_history.png",
        )

    # 5. Save model if not smoke test
    if not smoke_test:
        best_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(best_model_path))
        logger.info("Final model saved -> %s", best_model_path)
    else:
        logger.info("[smoke-test] Training completed successfully. Model not saved.")

    return model

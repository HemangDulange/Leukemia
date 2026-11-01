"""
Keras callbacks for the dual-stream ALL training pipeline.

Configures:
- EarlyStopping (monitors val_accuracy, restores best weights)
- ReduceLROnPlateau (monitors val_accuracy)
- ModelCheckpoint (monitors val_accuracy, saves best model directly to models/dual_stream.keras)
- CSVLogger (records per-epoch training metrics)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from tensorflow import keras

from src.config import AppConfig

logger = logging.getLogger(__name__)


def build_callbacks(
    cfg: AppConfig,
    smoke_test: bool = False,
    checkpoint_path: Path | None = None,
    monitor: str = "val_accuracy",
    mode: str = "max",
) -> List[keras.callbacks.Callback]:
    """Construct standard Keras callbacks for training.

    Parameters
    ----------
    cfg:
        Application configuration.
    smoke_test:
        If True, disables filesystem checkpointing for speed.
    checkpoint_path:
        Explicit path where the best model should be saved. Defaults to ``cfg.paths.model_path``.
    monitor:
        Metric to monitor (default: 'val_accuracy').
    mode:
        One of {'min', 'max'} (default: 'max' for accuracy).

    Returns
    -------
    List[keras.callbacks.Callback]
        List of configured Keras callback instances.
    """
    callbacks: List[keras.callbacks.Callback] = []

    # 1. Early Stopping
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=10,
            restore_best_weights=True,
            verbose=1,
        )
    )

    # 2. Reduce LR on Plateau
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        )
    )

    if smoke_test:
        logger.info("Smoke-test mode: skipping ModelCheckpoint and CSVLogger.")
        return callbacks

    # 3. Model Checkpoint saving best weights by val_accuracy
    target_ckpt = checkpoint_path or cfg.paths.model_path
    target_ckpt.parent.mkdir(parents=True, exist_ok=True)

    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            filepath=str(target_ckpt),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            verbose=1,
        )
    )

    # 4. CSV Training Log
    log_path = cfg.paths.metrics_dir / "training_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks.append(
        keras.callbacks.CSVLogger(str(log_path), append=True)
    )

    return callbacks

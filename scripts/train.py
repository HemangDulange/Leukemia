"""
CLI script: train the dual-stream ALL classifier in two progressive phases.

Usage
-----
    # Smoke-test mode (safe for CPU development):
    python scripts/train.py --smoke-test

    # Full two-phase training (Kaggle / Colab GPU):
    python scripts/train.py

    # Custom config YAML:
    python scripts/train.py --config configs/custom.yaml

Options
-------
--config PATH   Custom config YAML (default: configs/config.yaml).
--smoke-test    Run 1 epoch on 64 samples at 128px (CPU-safe).
--force-splits  Regenerate train/val/test CSV splits before training.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf

from src.config import load_config
from src.data.splits import make_splits
from src.training.train import run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the dual-stream ALL classifier in two progressive phases."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to custom config.yaml (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="1 epoch on tiny subset at 128px — safe for local CPU dev machine",
    )
    parser.add_argument(
        "--force-splits",
        action="store_true",
        help="Regenerate train/val/test CSV splits before training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Check hardware accelerators
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info("Found %d GPU accelerator(s): %s", len(gpus), gpus)
    else:
        logger.info("No GPU detected — running on CPU.")

    if args.smoke_test:
        logger.warning(
            "SMOKE-TEST mode active: %d samples, %dpx resolution, 1 epoch. Model will not be saved.",
            cfg.training.smoke_test_samples,
            cfg.image.smoke_test_size,
        )
    else:
        logger.info(
            "Full training mode: Phase 1 (%d epochs @ lr=%s) + Phase 2 (%d epochs @ lr=%s)",
            cfg.training.phase1_epochs,
            cfg.training.phase1_lr,
            cfg.training.phase2_epochs,
            cfg.training.phase2_lr,
        )

    # 1. Load or generate stratified splits
    train_df, val_df, _ = make_splits(cfg, force=args.force_splits)
    if args.smoke_test and (len(train_df) == 0 or len(val_df) == 0):
        from scripts.sanity_check import create_fallback_mock_split
        logger.warning("No dataset found on disk; using fallback mock samples for smoke-test verification.")
        train_df = create_fallback_mock_split(cfg, n_samples=32)
        val_df = create_fallback_mock_split(cfg, n_samples=16)

    # 2. Run two-phase training
    run_training(cfg, train_df, val_df, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()

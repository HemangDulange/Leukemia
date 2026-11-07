"""
CLI script: discover dataset images, compute perceptual hashes, and generate stratified CSV splits.

Usage
-----
    python scripts/prepare_data.py [--config PATH] [--force]

Options
-------
--config PATH   Custom config YAML (default: configs/config.yaml).
--force         Regenerate splits even if CSVs already exist.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import AppConfig, load_config
from src.data.splits import make_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover images, compute perceptual hashes, and generate reproducible splits."
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate splits even if CSVs already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    logger.info("Loading or building dataset splits...")
    train, val, test = make_splits(cfg, force=args.force)
    total_samples = len(train) + len(val) + len(test)

    logger.info(
        "Splits summary -> Total: %d | Train: %d | Val: %d | Test: %d",
        total_samples,
        len(train),
        len(val),
        len(test),
    )


if __name__ == "__main__":
    main()

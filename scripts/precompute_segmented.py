"""
Precompute and cache standardized (original, segmented) PNG pairs.

Eliminates train/serve skew by generating Segmented images using the exact same
HSV morphology pipeline (src/preprocessing/pair.py:prepare_pair) used in serving.

Reads train.csv, val.csv, and test.csv from data/splits/, processes each image:
- Load original blood smear with PIL .convert("RGB")
- Call prepare_pair() -> (orig_224, seg_224)
- Save lossless PNGs to:
    data/processed/orig/<class>/<name>.png
    data/processed/seg/<class>/<name>.png
- Skips already-existing files for fast resuming.
- Displays a tqdm progress bar and prints total counts (expect 3256 per folder on full dataset).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.config import AppConfig, load_config
from src.data.splits import make_splits
from src.preprocessing.pair import prepare_pair

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def count_cached_images(base_dir: Path) -> int:
    """Count existing PNG files in a directory hierarchy."""
    if not base_dir.exists():
        return 0
    return sum(1 for p in base_dir.rglob("*.png") if p.is_file())


def precompute_dataset(
    cfg: AppConfig,
    smoke_test: bool = False,
    force: bool = False,
    df: Optional[pd.DataFrame] = None,
) -> Tuple[int, int]:
    """Precompute and save lossless PNG pairs for all split rows.

    Parameters
    ----------
    cfg:
        Application configuration.
    smoke_test:
        If True, only processes a small sample subset.
    force:
        If True, overwrites existing cached PNG files.
    df:
        Optional pre-filtered DataFrame of rows to process. If None,
        loads train.csv, val.csv, and test.csv.

    Returns
    -------
    Tuple[int, int]
        (total_orig_count, total_seg_count)
    """
    processed_orig_root = cfg.data.processed_dir / "orig"
    processed_seg_root = cfg.data.processed_dir / "seg"
    processed_orig_root.mkdir(parents=True, exist_ok=True)
    processed_seg_root.mkdir(parents=True, exist_ok=True)

    # 1. Gather all rows to process
    if df is not None:
        all_df = df.copy()
    else:
        splits_dir = cfg.data.splits_dir
        dfs = []
        for split_name in ["train", "val", "test"]:
            csv_path = splits_dir / f"{split_name}.csv"
            if csv_path.exists():
                try:
                    split_df = pd.read_csv(csv_path)
                    if len(split_df) > 0:
                        dfs.append(split_df)
                except Exception as e:
                    logger.warning("Could not read %s: %s", csv_path, e)

        if not dfs:
            logger.info("No split CSVs found or they are empty. Generating splits...")
            train_df, val_df, test_df = make_splits(cfg, force=False)
            dfs = [d for d in [train_df, val_df, test_df] if len(d) > 0]

        if not dfs:
            logger.warning("No image data available to precompute.")
            return 0, 0

        all_df = pd.concat(dfs, ignore_index=True)

    if smoke_test:
        sample_n = min(cfg.training.smoke_test_samples, len(all_df))
        all_df = all_df.head(sample_n).copy()
        logger.info("[smoke-test] Precomputing %d samples at %dpx", sample_n, cfg.image.smoke_test_size)

    logger.info("Starting segmentation precomputation for %d files...", len(all_df))
    logger.info("Destination (orig): %s", processed_orig_root)
    logger.info("Destination (seg) : %s", processed_seg_root)

    processed_count = 0
    skipped_count = 0
    target_size = cfg.image.smoke_test_size if smoke_test else cfg.image.size

    for _, row in tqdm(all_df.iterrows(), total=len(all_df), desc="Precomputing pairs"):
        orig_path = Path(str(row["original_path"]))
        class_folder = str(row.get("class_folder") or orig_path.parent.name)
        stem = orig_path.stem

        out_orig_path = processed_orig_root / class_folder / f"{stem}.png"
        out_seg_path = processed_seg_root / class_folder / f"{stem}.png"

        # Skip existing files unless force is requested
        if not force and out_orig_path.exists() and out_seg_path.exists():
            skipped_count += 1
            continue

        if not orig_path.exists():
            logger.warning("Source image not found: %s. Skipping.", orig_path)
            continue

        try:
            # Load with PIL .convert("RGB")
            pil_img = Image.open(orig_path).convert("RGB")

            # Shared preprocessing function: segment full-resolution then resize
            orig_arr, seg_arr = prepare_pair(
                pil_img,
                cfg=cfg,
                target_size=target_size,
                return_mask=False,
            )

            # Convert float32 in [0, 255] to uint8 for lossless PNG storage
            orig_uint8 = np.clip(orig_arr, 0.0, 255.0).astype(np.uint8)
            seg_uint8 = np.clip(seg_arr, 0.0, 255.0).astype(np.uint8)

            out_orig_path.parent.mkdir(parents=True, exist_ok=True)
            out_seg_path.parent.mkdir(parents=True, exist_ok=True)

            Image.fromarray(orig_uint8).save(out_orig_path, format="PNG")
            Image.fromarray(seg_uint8).save(out_seg_path, format="PNG")
            processed_count += 1

        except Exception as err:
            logger.error("Failed to process %s: %s", orig_path, err)

    total_orig = count_cached_images(processed_orig_root)
    total_seg = count_cached_images(processed_seg_root)

    logger.info("=" * 60)
    logger.info("SEGMENTATION PRECOMPUTATION SUMMARY")
    logger.info("=" * 60)
    logger.info("Newly processed : %d", processed_count)
    logger.info("Skipped (cached): %d", skipped_count)
    logger.info("Total in data/processed/orig: %d (expect 3256 on full dataset)", total_orig)
    logger.info("Total in data/processed/seg : %d (expect 3256 on full dataset)", total_seg)
    logger.info("=" * 60)

    return total_orig, total_seg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute and cache standardized (orig, seg) pairs to eliminate train/serve skew."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to custom config YAML (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test on a small subset of samples",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation and overwrite existing PNGs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    precompute_dataset(cfg, smoke_test=args.smoke_test, force=args.force)


if __name__ == "__main__":
    main()

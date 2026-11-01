"""
CLI script: pair images, generate stratified splits, and evaluate HSV segmentation.

Functions:
1. Pair images from data/raw/{Original, Segmented} and generate stratified CSV splits.
2. Compare HSV color threshold segmentation against the dataset's Segmented/ ground
   truth on up to 50 random images.
3. Save side-by-side comparison figures and mean IoU metrics to reports/.

Usage
-----
    python scripts/prepare_data.py [--config PATH] [--force] [--samples N]

Options
-------
--config PATH   Custom config YAML (default: configs/config.yaml).
--force         Regenerate splits even if CSVs already exist.
--samples N     Number of images to evaluate for segmentation IoU (default: 50).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import AppConfig, load_config
from src.data.pairing import build_pairs
from src.data.splits import make_splits
from src.preprocessing.segmentation import segment_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Intersection-over-Union between two binary masks."""
    bin_a = (mask_a > 0).astype(bool)
    bin_b = (mask_b > 0).astype(bool)

    intersection = np.logical_and(bin_a, bin_b).sum()
    union = np.logical_or(bin_a, bin_b).sum()

    if union == 0:
        return 1.0 if np.all(~bin_a) and np.all(~bin_b) else 0.0
    return float(intersection / union)


def evaluate_segmentation(
    cfg: AppConfig,
    pairs_df: pd.DataFrame,
    num_samples: int = 50,
) -> Dict:
    """Compare HSV segmentation against ground-truth segmented images.

    Parameters
    ----------
    cfg:
        Application configuration.
    pairs_df:
        DataFrame containing ``original_path``, ``segmented_path``, and ``class_folder``.
    num_samples:
        Maximum number of paired images to evaluate.

    Returns
    -------
    dict
        Evaluation results containing mean IoU, per-class IoU, and sample details.
    """
    if len(pairs_df) == 0:
        logger.warning("No paired images available to evaluate segmentation.")
        return {"error": "No paired images found in data/raw"}

    sample_size = min(num_samples, len(pairs_df))
    sampled = pairs_df.sample(n=sample_size, random_state=cfg.seed).reset_index(drop=True)
    logger.info("Evaluating segmentation on %d sample images...", sample_size)

    ious: List[float] = []
    class_ious: Dict[str, List[float]] = {}
    comparison_data = []

    for idx, row in sampled.iterrows():
        orig_p = Path(row["original_path"])
        seg_p = Path(row["segmented_path"])
        cls_name = row["class_folder"]

        orig_bgr = cv2.imread(str(orig_p), cv2.IMREAD_COLOR)
        gt_bgr = cv2.imread(str(seg_p), cv2.IMREAD_COLOR)

        if orig_bgr is None or gt_bgr is None:
            continue

        # Pipeline segmentation: BGR -> (binary_mask, segmented_bgr)
        pred_mask, pred_bgr = segment_image(orig_bgr, cfg)

        # Ground truth binary mask (non-black pixels in ground truth segmented image)
        gt_gray = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY)
        _, gt_mask = cv2.threshold(gt_gray, 10, 255, cv2.THRESH_BINARY)

        # Match shapes if slight resizing difference
        if pred_mask.shape != gt_mask.shape:
            gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            gt_bgr = cv2.resize(gt_bgr, (orig_bgr.shape[1], orig_bgr.shape[0]))

        iou = compute_iou(pred_mask, gt_mask)
        ious.append(iou)

        if cls_name not in class_ious:
            class_ious[cls_name] = []
        class_ious[cls_name].append(iou)

        if len(comparison_data) < 6:
            comparison_data.append({
                "class": cls_name,
                "iou": iou,
                "orig_rgb": cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB),
                "gt_rgb": cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB),
                "pred_mask": pred_mask,
                "pred_rgb": cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB),
            })

    mean_iou = float(np.mean(ious)) if ious else 0.0
    median_iou = float(np.median(ious)) if ious else 0.0
    std_iou = float(np.std(ious)) if ious else 0.0

    per_class_summary = {
        cls: {"mean_iou": float(np.mean(vals)), "count": len(vals)}
        for cls, vals in class_ious.items()
    }

    results = {
        "num_evaluated": len(ious),
        "mean_iou": mean_iou,
        "median_iou": median_iou,
        "std_iou": std_iou,
        "per_class": per_class_summary,
    }

    # Save metrics JSON
    metrics_path = cfg.paths.metrics_dir / "segmentation_iou.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Saved segmentation metrics -> %s (Mean IoU: %.4f)", metrics_path, mean_iou)

    # Save side-by-side comparison figure
    if comparison_data:
        save_comparison_figure(comparison_data, cfg.paths.figures_dir / "segmentation_comparison.png")

    return results


def save_comparison_figure(comparison_data: List[Dict], out_path: Path) -> None:
    """Generate and save side-by-side comparison figure."""
    n_rows = len(comparison_data)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.2 * n_rows))

    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    headers = ["Original PBS", "Ground Truth Segmented", "HSV Binary Mask", "HSV Segmented (Ours)"]

    for col, title in enumerate(headers):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold", pad=8)

    for row_idx, item in enumerate(comparison_data):
        ax_orig = axes[row_idx, 0]
        ax_gt = axes[row_idx, 1]
        ax_mask = axes[row_idx, 2]
        ax_pred = axes[row_idx, 3]

        ax_orig.imshow(item["orig_rgb"])
        ax_orig.set_ylabel(f"{item['class']}\nIoU: {item['iou']:.3f}", fontsize=11, fontweight="medium")
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])

        ax_gt.imshow(item["gt_rgb"])
        ax_gt.axis("off")

        ax_mask.imshow(item["pred_mask"], cmap="gray")
        ax_mask.axis("off")

        ax_pred.imshow(item["pred_rgb"])
        ax_pred.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved segmentation comparison figure -> %s", out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair images, generate stratified splits, and evaluate HSV segmentation."
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate splits even if CSVs already exist",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of images to evaluate for segmentation IoU (default: 50)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # 1. Pair images & create splits
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

    # 2. Evaluate segmentation against ground truth
    all_pairs = pd.concat([train, val, test], ignore_index=True)
    if len(all_pairs) > 0:
        results = evaluate_segmentation(cfg, all_pairs, num_samples=args.samples)
        if "mean_iou" in results:
            print("\n" + "=" * 50)
            print("SEGMENTATION EVALUATION RESULTS")
            print(f"Evaluated images : {results['num_evaluated']}")
            print(f"Mean IoU         : {results['mean_iou']:.4f}")
            print(f"Median IoU       : {results['median_iou']:.4f}")
            print(f"Std IoU          : {results['std_iou']:.4f}")
            print("\nPer-class Mean IoU:")
            for cls, metrics in results["per_class"].items():
                print(f"  {cls:<15}: {metrics['mean_iou']:.4f} ({metrics['count']} samples)")
            print("=" * 50)
    else:
        logger.info(
            "Notice: data/raw does not yet contain images. Place the dataset in data/raw/{Original, Segmented}."
        )


if __name__ == "__main__":
    main()

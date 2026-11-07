"""
Generate visual validation grid of HSV segmentation across all 5 classes.

For 6 random images per class (seed 42), creates a 30-image grid displaying:
  Original | Binary Mask | Segmented
Saves the figure to reports/figures/mask_preview.png to verify segmentation
performance on all 5 subtypes (including AML and Normal).

Usage:
    python scripts/preview_masks.py [--config PATH] [--seed 42]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.config import AppConfig, load_config
from src.data.splits import VALID_EXTENSIONS, load_rgb_image
from src.preprocessing.pair import prepare_pair

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_mask_preview(
    cfg: AppConfig,
    images_per_class: int = 6,
    seed: int = 42,
    output_path: Path | None = None,
) -> Path:
    """Generate and save 30-image grid (6 images x 5 classes) of original, mask, segmented."""
    rng = np.random.RandomState(seed)
    classes = cfg.data.classes
    display_names = cfg.data.display_names or {c: c for c in classes}
    raw_dir = cfg.data.raw_dir

    out_file = output_path or (cfg.paths.figures_dir / "mask_preview.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    items_to_plot = []

    for cls_name in classes:
        cls_folder = raw_dir / cls_name
        files: List[Path] = []
        if cls_folder.exists():
            files = [
                p for p in cls_folder.rglob("*")
                if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
            ]
            files.sort(key=lambda p: str(p))

        display_name = display_names.get(cls_name, cls_name)

        if len(files) >= images_per_class:
            chosen_indices = rng.choice(len(files), size=images_per_class, replace=False)
            chosen_files = [files[i] for i in chosen_indices]
        elif len(files) > 0:
            chosen_indices = rng.choice(len(files), size=images_per_class, replace=True)
            chosen_files = [files[i] for i in chosen_indices]
        else:
            chosen_files = []

        for idx, f in enumerate(chosen_files):
            try:
                pil_img = load_rgb_image(f)
                orig_arr, seg_arr, mask_arr = prepare_pair(
                    pil_img,
                    cfg=cfg,
                    target_size=224,
                    return_mask=True,
                )
                items_to_plot.append({
                    "class": cls_name,
                    "display_name": display_name,
                    "filename": f.name,
                    "orig": np.clip(orig_arr, 0, 255).astype(np.uint8),
                    "mask": mask_arr.astype(np.uint8),
                    "seg": np.clip(seg_arr, 0, 255).astype(np.uint8),
                })
            except Exception as e:
                logger.error("Failed to process %s for mask preview: %s", f, e)

        # Fallback synthetic items if no files found
        if not chosen_files:
            logger.warning("No images found for %s, creating synthetic previews", cls_name)
            for i in range(images_per_class):
                synth_arr = np.full((224, 224, 3), 200, dtype=np.uint8)
                synth_mask = np.zeros((224, 224), dtype=np.uint8)
                synth_seg = np.zeros((224, 224, 3), dtype=np.uint8)
                items_to_plot.append({
                    "class": cls_name,
                    "display_name": display_name,
                    "filename": f"synthetic_{cls_name}_{i}.png",
                    "orig": synth_arr,
                    "mask": synth_mask,
                    "seg": synth_seg,
                })

    n_rows = len(items_to_plot)
    fig, axes = plt.subplots(n_rows, 3, figsize=(10, 2.5 * n_rows), dpi=150)
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    axes[0, 0].set_title("Original (224x224)", fontsize=12, fontweight="bold", pad=8)
    axes[0, 1].set_title("HSV Binary Mask", fontsize=12, fontweight="bold", pad=8)
    axes[0, 2].set_title("Segmented Stream (Ours)", fontsize=12, fontweight="bold", pad=8)

    for row_idx, item in enumerate(items_to_plot):
        ax_orig = axes[row_idx, 0]
        ax_mask = axes[row_idx, 1]
        ax_seg = axes[row_idx, 2]

        ax_orig.imshow(item["orig"])
        ax_orig.set_ylabel(
            f"{item['display_name']}\n{item['filename'][:16]}",
            fontsize=9,
            fontweight="semibold",
            rotation=0,
            labelpad=50,
            va="center",
        )
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])

        ax_mask.imshow(item["mask"], cmap="gray")
        ax_mask.axis("off")

        ax_seg.imshow(item["seg"])
        ax_seg.axis("off")

    plt.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %d-image mask preview grid -> %s", n_rows, out_file)
    return out_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 30-image mask preview grid across all 5 classes."
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Target output path (default: reports/figures/mask_preview.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    generate_mask_preview(cfg=cfg, seed=args.seed, output_path=args.output)


if __name__ == "__main__":
    main()

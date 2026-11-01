"""
Generate sample peripheral blood smear test images for app demonstration.

For each of the 4 leukemia classes:
1. Filters test images from data/splits/test.csv.
2. Shuffles with a fixed seed (42).
3. Evaluates candidates with Predictor and selects the first correctly classified test image.
4. Copies the original images to /kaggle/working/samples/ (or app/samples/ locally):
   - Hematogones_sample.jpg  (Benign)
   - EarlyPreB_sample.jpg     (Early Pre-B ALL)
   - PreB_sample.jpg          (Pre-B ALL)
   - ProB_sample.jpg          (Pro-B ALL)
5. Packages them into samples.zip for easy download and deployment.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Optional

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
from PIL import Image

from src.config import AppConfig, load_config
from src.inference.predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Canonical output filename mapping
_SAMPLE_FILENAMES: Dict[str, str] = {
    "Benign": "Hematogones_sample.jpg",
    "Early": "EarlyPreB_sample.jpg",
    "Pre": "PreB_sample.jpg",
    "Pre-B": "PreB_sample.jpg",
    "Pro": "ProB_sample.jpg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select correctly classified test images per class and build samples.zip."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to custom config YAML (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to trained model file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to /kaggle/working/samples if on Kaggle, else app/samples)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic selection (default: 42)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run in smoke-test mode with synthetic/fallback samples",
    )
    return parser.parse_args()


def make_samples(
    cfg: AppConfig,
    model_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    seed: int = 42,
    smoke_test: bool = False,
) -> Path:
    # Determine target output directory
    if output_dir is None:
        kaggle_dir = Path("/kaggle/working/samples")
        if kaggle_dir.parent.exists():
            dest_dir = kaggle_dir
        else:
            dest_dir = _REPO_ROOT / "app" / "samples"
    else:
        dest_dir = Path(output_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory for bundled samples: %s", dest_dir)

    # 1. Load test split
    test_csv = cfg.data.splits_dir / "test.csv"
    if test_csv.exists() and test_csv.stat().st_size > 0:
        test_df = pd.read_csv(test_csv)
    else:
        test_df = pd.DataFrame()

    predictor = Predictor(cfg=cfg, model_path=model_path)
    classes = cfg.data.classes

    selected_files = []

    for cls_idx, class_name in enumerate(classes):
        target_filename = _SAMPLE_FILENAMES.get(class_name, f"{class_name}_sample.jpg")
        target_path = dest_dir / target_filename

        cls_candidates = test_df[test_df["label_index"] == cls_idx] if len(test_df) > 0 else pd.DataFrame()

        chosen_src: Optional[Path] = None

        if len(cls_candidates) > 0:
            shuffled = cls_candidates.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            for _, row in shuffled.iterrows():
                src_p = Path(str(row["original_path"]))
                if not src_p.exists():
                    continue

                try:
                    pil_img = Image.open(src_p).convert("RGB")
                    pred = predictor.predict(pil_img, generate_gradcam=False)
                    if pred["label_index"] == cls_idx:
                        chosen_src = src_p
                        logger.info("Found correctly classified test sample for %s: %s", class_name, src_p.name)
                        break
                except Exception as err:
                    logger.debug("Error predicting %s: %s", src_p, err)

            # If no candidate classified correctly (e.g. untrained smoke weights), take the first candidate
            if chosen_src is None:
                for _, row in shuffled.iterrows():
                    src_p = Path(str(row["original_path"]))
                    if src_p.exists():
                        chosen_src = src_p
                        logger.warning("No correctly predicted sample for %s; using first test image: %s", class_name, src_p.name)
                        break

        # Fallback if no test images exist on disk (local CPU dev environment)
        if chosen_src is None:
            # Check app/samples or generate synthetic sample image
            logger.warning("Generating synthetic placeholder sample for %s -> %s", class_name, target_filename)
            synthetic_img = Image.new("RGB", (224, 224), color=(200, 180, 220))
            synthetic_img.save(target_path, "JPEG")
            selected_files.append(target_path)
            continue

        shutil.copy2(chosen_src, target_path)
        selected_files.append(target_path)
        logger.info("Copied %s -> %s", chosen_src.name, target_path)

    # 2. Also ensure local app/samples has copies if dest_dir was /kaggle/working/samples
    local_samples_dir = _REPO_ROOT / "app" / "samples"
    if dest_dir != local_samples_dir:
        local_samples_dir.mkdir(parents=True, exist_ok=True)
        for f in selected_files:
            shutil.copy2(f, local_samples_dir / f.name)
        logger.info("Mirrored samples to local app/samples/: %s", local_samples_dir)

    # 3. Create samples.zip containing the four files
    zip_dest = dest_dir.parent / "samples.zip"
    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for f in selected_files:
            zip_file.write(f, arcname=f.name)

    logger.info("Created samples zip archive: %s (%d files)", zip_dest, len(selected_files))
    return zip_dest


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    make_samples(
        cfg=cfg,
        model_path=args.model_path,
        output_dir=args.output_dir,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()

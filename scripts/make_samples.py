"""
Select the highest-confidence correctly classified test image for each leukemia subtype.

Evaluates every image in the test split using Predictor (with unified prepare_pair),
computes per-class test accuracy, ranks correct predictions by confidence,
and copies the top-1 sample per class to /kaggle/working/samples/ (and app/samples/):
  - Hematogones_sample.jpg  (Benign)
  - EarlyPreB_sample.jpg     (Early Pre-B ALL)
  - PreB_sample.jpg          (Pre-B ALL)
  - ProB_sample.jpg          (Pro-B ALL)

Finally, packages the 4 sample images into /kaggle/working/samples.zip.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tensorflow as tf
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.config import AppConfig, load_config
from src.data.splits import make_splits
from src.inference.predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Canonical filename mapping for each class index and class folder name
_SAMPLE_FILENAMES_BY_INDEX: Dict[int, str] = {
    0: "Hematogones_sample.jpg",
    1: "EarlyPreB_sample.jpg",
    2: "PreB_sample.jpg",
    3: "ProB_sample.jpg",
}

_SAMPLE_FILENAMES_BY_CLASS: Dict[str, str] = {
    "Benign": "Hematogones_sample.jpg",
    "Early": "EarlyPreB_sample.jpg",
    "Pre": "PreB_sample.jpg",
    "Pre-B": "PreB_sample.jpg",
    "Pro": "ProB_sample.jpg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select highest-confidence correct test image per class and build samples.zip."
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
        help="Output directory (defaults to /kaggle/working/samples if available, else app/samples)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run in smoke-test mode with mock/fallback samples",
    )
    return parser.parse_args()


def resolve_output_dir(output_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """Resolve target samples directory and zip archive path."""
    if output_dir is not None:
        dest_dir = Path(output_dir)
    elif Path("/kaggle/working").exists():
        dest_dir = Path("/kaggle/working/samples")
    else:
        # If /kaggle/working cannot be created (e.g. Windows non-root), use local app/samples
        try:
            candidate = Path("/kaggle/working/samples")
            candidate.mkdir(parents=True, exist_ok=True)
            dest_dir = candidate
        except OSError:
            dest_dir = _REPO_ROOT / "app" / "samples"

    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_dest = dest_dir.parent / "samples.zip"
    return dest_dir, zip_dest


def make_samples(
    cfg: AppConfig,
    model_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    smoke_test: bool = False,
) -> Tuple[List[Path], Path]:
    """Evaluate test split, pick highest-confidence correct image per class, and bundle zip.

    Parameters
    ----------
    cfg:
        Application configuration.
    model_path:
        Optional path to model weights.
    output_dir:
        Optional target directory for samples.
    smoke_test:
        If True, restricts test size and uses mock data if needed.

    Returns
    -------
    Tuple[List[Path], Path]
        List of copied sample files and path to samples.zip archive.
    """
    dest_dir, zip_dest = resolve_output_dir(output_dir)
    logger.info("Destination directory for sample images: %s", dest_dir)
    logger.info("Destination for samples zip archive   : %s", zip_dest)

    # 1. Load reproducible test split using existing make_splits helper (seed 42)
    _, _, test_df = make_splits(cfg, force=False)

    if len(test_df) == 0:
        if smoke_test:
            from scripts.sanity_check import create_fallback_mock_split
            logger.warning("Test split is empty. Using fallback mock split for smoke-test.")
            test_df = create_fallback_mock_split(cfg, n_samples=16)
        else:
            raise FileNotFoundError(
                f"Test split in {cfg.data.splits_dir} is empty. Run scripts/prepare_data.py first."
            )
    elif smoke_test:
        sample_n = min(cfg.training.smoke_test_samples, len(test_df))
        test_df = test_df.sample(n=sample_n, random_state=cfg.seed).reset_index(drop=True)

    n_images = len(test_df)
    logger.info("Evaluating %d test images with Predictor...", n_images)

    # 2. Instantiate Predictor
    predictor = Predictor(cfg=cfg, model_path=model_path)
    classes = cfg.data.classes
    display_mapping = getattr(predictor, "_display_mapping", {
        "Benign": "Hematogones",
        "Early": "Early Pre-B ALL",
        "Pre": "Pre-B ALL",
        "Pre-B": "Pre-B ALL",
        "Pro": "Pro-B ALL",
    })

    num_classes = len(classes)
    correct_candidates: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(num_classes)}
    all_candidates: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(num_classes)}
    total_counts: Dict[int, int] = {i: 0 for i in range(num_classes)}

    # 3. Run Predictor on every image in the TEST split
    for _, row in tqdm(test_df.iterrows(), total=n_images, desc="Predictor test evaluation"):
        src_path = Path(str(row["original_path"]))
        true_label = int(row["label_index"])
        total_counts[true_label] += 1

        if not src_path.exists():
            logger.warning("File missing: %s", src_path)
            continue

        try:
            pil_img = Image.open(src_path).convert("RGB")
            pred = predictor.predict(pil_img, generate_gradcam=False)
            pred_label = int(pred["label_index"])
            confidence = float(pred["confidence"])

            candidate_info = {
                "path": src_path,
                "filename": src_path.name,
                "confidence": confidence,
                "pred_label": pred_label,
            }
            all_candidates[true_label].append(candidate_info)

            if pred_label == true_label:
                correct_candidates[true_label].append(candidate_info)
        except Exception as err:
            logger.error("Error predicting %s: %s", src_path, err)

    # 4. Report per-class accuracy, top 5 correct images, and select top-1 sample
    selected_files: List[Path] = []
    local_app_samples = _REPO_ROOT / "app" / "samples"
    local_app_samples.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("TEST SPLIT EVALUATION & SAMPLE SELECTION REPORT")
    print("=" * 70)

    for cls_idx in range(num_classes):
        class_name = classes[cls_idx]
        display_name = display_mapping.get(class_name, class_name)
        target_filename = _SAMPLE_FILENAMES_BY_INDEX.get(
            cls_idx,
            _SAMPLE_FILENAMES_BY_CLASS.get(class_name, f"{class_name}_sample.jpg"),
        )
        target_path = dest_dir / target_filename

        total_cls = total_counts[cls_idx]
        correct_list = correct_candidates[cls_idx]
        n_correct = len(correct_list)
        accuracy = (n_correct / total_cls) if total_cls > 0 else 0.0

        # Sort correctly classified images descending by confidence (highest confidence first)
        correct_list.sort(key=lambda x: x["confidence"], reverse=True)

        print(f"\nSubtype [{cls_idx}]: {class_name} ({display_name})")
        print(f"  Test Accuracy : {accuracy * 100:6.2f}% ({n_correct}/{total_cls})")
        print(f"  Top 5 Correct Predictions (by confidence):")

        top_5 = correct_list[:5]
        if top_5:
            for rank, item in enumerate(top_5, start=1):
                print(f"    {rank}. {item['filename']:<32} | Confidence: {item['confidence'] * 100:6.2f}%")
        else:
            print("    (No correctly classified images found for this class)")

        # Choose the one with the highest confidence for that class (no random choice)
        chosen_item: Optional[Dict[str, Any]] = None
        if correct_list:
            chosen_item = correct_list[0]
        elif all_candidates[cls_idx]:
            # Fallback if no images were predicted correctly (e.g. untrained smoke weights)
            sorted_all = sorted(all_candidates[cls_idx], key=lambda x: x["confidence"], reverse=True)
            chosen_item = sorted_all[0]
            logger.warning(
                "No correctly classified candidate for %s; using highest-confidence candidate %s (predicted as %d)",
                class_name,
                chosen_item["filename"],
                chosen_item["pred_label"],
            )

        if chosen_item is not None and chosen_item["path"].exists():
            if chosen_item["path"].resolve() != target_path.resolve():
                shutil.copy2(chosen_item["path"], target_path)

            # Also mirror to app/samples when running full generation on Kaggle
            mirror_path = local_app_samples / target_filename
            if (
                not smoke_test
                and dest_dir.resolve() != local_app_samples.resolve()
                and chosen_item["path"].resolve() != mirror_path.resolve()
            ):
                shutil.copy2(chosen_item["path"], mirror_path)

            selected_files.append(target_path)
            print(
                f"  Selected Sample: {chosen_item['filename']} "
                f"({chosen_item['confidence'] * 100:.2f}%) -> {target_filename}"
            )
        else:
            # Synthetic fallback for mock smoke tests without files
            logger.warning("Generating placeholder sample for %s -> %s", class_name, target_filename)
            synthetic_img = Image.new("RGB", (224, 224), color=(200, 180, 220))
            if not target_path.exists():
                synthetic_img.save(target_path, "JPEG")
            mirror_path = local_app_samples / target_filename
            if dest_dir.resolve() != local_app_samples.resolve() and not mirror_path.exists():
                synthetic_img.save(mirror_path, "JPEG")
            selected_files.append(target_path)

    print("\n" + "=" * 70)

    # 5. Zip the 4 chosen original JPEGs as samples.zip
    zip_dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for f in selected_files:
            zip_file.write(f, arcname=f.name)

    # Mirror zip to app/samples.zip if dest_dir is outside repo and not in smoke-test mode
    if not smoke_test and zip_dest.parent.resolve() != local_app_samples.parent.resolve():
        try:
            local_zip = local_app_samples.parent / "samples.zip"
            if zip_dest.resolve() != local_zip.resolve():
                shutil.copy2(zip_dest, local_zip)
        except Exception as e:
            logger.debug("Could not mirror zip locally: %s", e)

    print(f"\n[PASSED] Packaged {len(selected_files)} samples into archive:")
    print(f"  {zip_dest} ({zip_dest.stat().st_size / 1024:.1f} KB)\n")

    return selected_files, zip_dest


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    make_samples(
        cfg=cfg,
        model_path=args.model_path,
        output_dir=args.output_dir,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()

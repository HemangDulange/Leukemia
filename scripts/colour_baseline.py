"""
Color-only baseline classifier using a Random Forest.

Trains a Random Forest (200 trees, seed 42) strictly on 4 color/illumination features
extracted from the original images in the TRAIN split, and evaluates on the TEST split:
1. Mean Red (R) channel intensity
2. Mean Green (G) channel intensity
3. Mean Blue (B) channel intensity
4. Black-pixel share (fraction of pixels with max(R, G, B) < 25)

Validates the hypothesis that leukemia subtype and slide origin are strongly tied
to stain coloration, providing an empirical comparison against the dual-stream deep model.

Usage
-----
    python scripts/colour_baseline.py [--config PATH] [--smoke-test]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from src.config import AppConfig, load_config
from src.data.splits import load_rgb_image, make_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_colour_features(img_path: Path | str) -> Tuple[float, float, float, float]:
    """Extract (mean_r, mean_g, mean_b, black_pixel_share) from an original image."""
    img = load_rgb_image(img_path)
    arr = np.array(img, dtype=np.float32)

    mean_r = float(np.mean(arr[..., 0]))
    mean_g = float(np.mean(arr[..., 1]))
    mean_b = float(np.mean(arr[..., 2]))

    # Fraction of pixels with max channel < 25 (dark/black background or border)
    is_black = np.max(arr, axis=-1) < 25.0
    black_share = float(np.mean(is_black))

    return mean_r, mean_g, mean_b, black_share


def extract_features_from_split(
    df: pd.DataFrame,
    desc: str = "Extracting features",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract color feature matrix X (N, 4) and labels y (N,) from a split DataFrame."""
    X_list = []
    y_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        p = Path(str(row["original_path"]))
        label = int(row["label_index"])
        try:
            feats = extract_colour_features(p)
            X_list.append(feats)
            y_list.append(label)
        except Exception as e:
            logger.error("Error extracting features from %s: %s", p, e)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def run_colour_baseline(cfg: AppConfig, smoke_test: bool = False) -> Dict:
    """Train and evaluate Random Forest color baseline on TRAIN and TEST splits."""
    print("\n" + "=" * 70)
    print("COLOUR-ONLY BASELINE (RANDOM FOREST ON MEAN RGB + BLACK-PIXEL SHARE)")
    print("=" * 70)

    # 1. Load canonical class names
    classes = cfg.data.classes
    if cfg.paths.class_names_file.exists():
        with cfg.paths.class_names_file.open(encoding="utf-8") as fh:
            meta = json.load(fh)
        display_names = list(meta["index_to_display"].values())
    else:
        disp_map = cfg.data.display_names or {c: c for c in classes}
        display_names = [disp_map.get(c, c) for c in classes]

    # 2. Load splits
    train_df, _, test_df = make_splits(cfg, force=False)
    if len(train_df) == 0 or len(test_df) == 0:
        logger.error("Empty train or test split. Run scripts/prepare_data.py first.")
        sys.exit(1)

    if smoke_test:
        train_df = train_df.head(min(32, len(train_df)))
        test_df = test_df.head(min(16, len(test_df)))
        logger.info("[smoke-test] Sampled %d train and %d test rows", len(train_df), len(test_df))

    # 3. Extract 4 color features per image
    X_train, y_train = extract_features_from_split(train_df, desc="Train color features")
    X_test, y_test = extract_features_from_split(test_df, desc="Test color features")

    # 4. Train Random Forest (200 trees, seed 42)
    logger.info("Training RandomForestClassifier(n_estimators=200, random_state=42)...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 5. Evaluate on TEST split
    y_pred = rf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(classes))))
    cls_report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(classes))),
        target_names=display_names,
        output_dict=True,
        zero_division=0,
    )

    # 6. Print Results
    feature_names = ["mean_r", "mean_g", "mean_b", "black_pixel_share"]
    importances = rf.feature_importances_

    print("\n--- Model Performance ---")
    print(f"Test Accuracy: {acc * 100:.2f}% ({np.sum(y_test == y_pred)}/{len(y_test)})")
    print("\nFeature Importances:")
    for fname, imp in zip(feature_names, importances):
        print(f"  {fname:<20}: {imp * 100:.2f}%")

    print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
    cm_df = pd.DataFrame(cm, index=display_names, columns=display_names)
    print(cm_df.to_string())

    print("\nClassification Report:")
    report_df = pd.DataFrame(cls_report).transpose()
    print(report_df.to_string())
    print("=" * 70 + "\n")

    # 7. Save to reports/metrics/colour_baseline.csv
    csv_path = cfg.paths.metrics_dir / "colour_baseline.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {"section": "Summary", "key": "test_accuracy", "value": round(acc, 4)},
        {"section": "Summary", "key": "train_samples", "value": len(X_train)},
        {"section": "Summary", "key": "test_samples", "value": len(X_test)},
    ]
    for fname, imp in zip(feature_names, importances):
        rows.append({"section": "Feature Importance", "key": fname, "value": round(float(imp), 4)})

    for cls_name in display_names:
        if cls_name in cls_report:
            rec = cls_report[cls_name]
            rows.append({"section": "Per-Class Recall", "key": cls_name, "value": round(float(rec["recall"]), 4)})
            rows.append({"section": "Per-Class Precision", "key": cls_name, "value": round(float(rec["precision"]), 4)})
            rows.append({"section": "Per-Class F1", "key": cls_name, "value": round(float(rec["f1-score"]), 4)})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(csv_path, index=False)
    logger.info("Saved colour baseline results -> %s", csv_path)

    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report,
        "feature_importances": dict(zip(feature_names, importances.tolist())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Random Forest color-only baseline."
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on small sample slice for rapid verification",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_colour_baseline(cfg, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()

"""Unit tests for src/data: pairing, splits, class weights, and tf.data dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.dataset import build_dataset
from src.data.pairing import build_pairs, report_unmatched_files
from src.data.splits import compute_class_weights, make_splits, print_class_distribution


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture()
def mock_df() -> pd.DataFrame:
    """Create a mock paired DataFrame with 20 samples across 4 classes."""
    data = []
    for idx in range(20):
        cls_idx = idx % 4
        cls_name = ["Benign", "Early", "Pre-B", "Pro"][cls_idx]
        data.append(
            {
                "original_path": f"dummy/path/orig_{idx}.jpg",
                "segmented_path": f"dummy/path/seg_{idx}.jpg",
                "class_folder": cls_name,
                "label_index": cls_idx,
                "filename": f"sample_{idx}.jpg",
            }
        )
    return pd.DataFrame(data)


class TestPairing:
    def test_report_unmatched_files(self):
        orig = {"img1": Path("orig/img1.jpg"), "img2": Path("orig/img2.jpg")}
        seg = {"img2": Path("seg/img2.jpg"), "img3": Path("seg/img3.jpg")}
        unmatched_orig, unmatched_seg = report_unmatched_files(orig, seg, "TestClass")
        assert unmatched_orig == {"img1"}
        assert unmatched_seg == {"img3"}


class TestSplitsAndWeights:
    def test_compute_class_weights_balanced(self, mock_df):
        # 5 samples per class -> equal weights
        weights = compute_class_weights(mock_df, num_classes=4)
        assert len(weights) == 4
        for w in weights.values():
            assert abs(w - 1.0) < 1e-4

    def test_compute_class_weights_imbalanced(self):
        df = pd.DataFrame({
            "label_index": [0, 0, 0, 0, 1, 1, 2, 3],
        })
        weights = compute_class_weights(df, num_classes=4)
        # Class 0 has 4 samples (most frequent) -> lowest weight
        # Class 2 and 3 have 1 sample -> highest weight
        assert weights[0] < weights[1] < weights[2]
        assert weights[2] == weights[3]

    def test_print_class_distribution(self, mock_df, cfg):
        train = mock_df.iloc[:12]
        val = mock_df.iloc[12:16]
        test = mock_df.iloc[16:]
        dist_df = print_class_distribution(train, val, test, cfg.data.classes)
        assert len(dist_df) == 4
        assert "train_count" in dist_df.columns
        assert dist_df["total_count"].sum() == 20


class TestDatasetPipeline:
    def test_dataset_yields_dict_structure(self, cfg, mock_df, tmp_path):
        """Verify dataset yields ({"original": ..., "segmented": ...}, label)."""
        import cv2

        # Create real dummy image files in tmp_path so _py_func can read them
        img_file = tmp_path / "dummy.jpg"
        dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(img_file), dummy_img)

        df = pd.DataFrame([
            {
                "original_path": str(img_file),
                "segmented_path": str(img_file),
                "class_folder": "Benign",
                "label_index": 0,
                "filename": "dummy.jpg",
            }
        ])

        ds = build_dataset(df, cfg, split="val", smoke_test=True)
        for batch_x, batch_y in ds.take(1):
            assert isinstance(batch_x, dict)
            assert "original" in batch_x
            assert "segmented" in batch_x
            assert batch_y.shape[-1] == cfg.model.num_classes

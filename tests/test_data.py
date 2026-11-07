"""Unit tests for src/data: canonical perceptual hash, splits, class weights, and tf.data dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.config import load_config
from src.data.dataset import build_dataset
from src.data.splits import (
    compute_canonical_hash,
    compute_class_weights,
    load_rgb_image,
    make_splits,
    print_class_distribution,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture()
def mock_df() -> pd.DataFrame:
    """Create a mock DataFrame with 25 samples across 5 classes."""
    data = []
    classes = ["AML", "Early_PreB", "Normal", "PreB", "ProB"]
    for idx in range(25):
        cls_idx = idx % 5
        cls_name = classes[cls_idx]
        data.append(
            {
                "original_path": f"dummy/path/orig_{idx}.jpg",
                "class_folder": cls_name,
                "label_index": cls_idx,
                "filename": f"sample_{idx}.jpg",
                "canonical_hash": f"hash_{idx}",
                "group_id": idx,
            }
        )
    return pd.DataFrame(data)


class TestPerceptualHash:
    def test_canonical_hash_invariance(self):
        """Verify canonical hash is identical across rotations and flips."""
        pattern = np.zeros((64, 64, 3), dtype=np.uint8)
        pattern[10:30, 10:30] = [220, 100, 50]
        pattern[40:60, 40:60] = [50, 150, 220]
        img = Image.fromarray(pattern)

        h_orig = compute_canonical_hash(img)
        h_rot90 = compute_canonical_hash(img.rotate(90))
        h_rot180 = compute_canonical_hash(img.rotate(180))
        h_rot270 = compute_canonical_hash(img.rotate(270))
        h_flip_lr = compute_canonical_hash(img.transpose(Image.FLIP_LEFT_RIGHT))
        h_flip_tb = compute_canonical_hash(img.transpose(Image.FLIP_TOP_BOTTOM))

        assert h_orig == h_rot90 == h_rot180 == h_rot270 == h_flip_lr == h_flip_tb


class TestSplitsAndWeights:
    def test_compute_class_weights_balanced(self, mock_df):
        # 5 samples per class across 5 classes -> equal weights
        weights = compute_class_weights(mock_df, num_classes=5)
        assert len(weights) == 5
        for w in weights.values():
            assert abs(w - 1.0) < 1e-4

    def test_compute_class_weights_imbalanced(self):
        df = pd.DataFrame({
            "label_index": [0, 0, 0, 0, 1, 1, 2, 3, 4],
        })
        weights = compute_class_weights(df, num_classes=5)
        assert weights[0] < weights[1]
        assert weights[2] == weights[3] == weights[4]

    def test_print_class_distribution(self, mock_df, cfg):
        train = mock_df.iloc[:15]
        val = mock_df.iloc[15:20]
        test = mock_df.iloc[20:]
        dist_df = print_class_distribution(train, val, test, cfg.data.classes)
        assert len(dist_df) == 5
        assert "train_count" in dist_df.columns
        assert dist_df["total_count"].sum() == 25


class TestDatasetPipeline:
    def test_dataset_yields_dict_structure(self, cfg, tmp_path):
        """Verify dataset yields ({"original": ..., "segmented": ...}, label) with 5 classes."""
        import cv2

        img_file = tmp_path / "dummy.png"
        dummy_img = np.full((64, 64, 3), 200, dtype=np.uint8)
        cv2.imwrite(str(img_file), dummy_img)

        df = pd.DataFrame([
            {
                "original_path": str(img_file),
                "class_folder": "AML",
                "label_index": 0,
                "filename": "dummy.png",
            }
        ])

        ds = build_dataset(df, cfg, split="val", smoke_test=True)
        for batch_x, batch_y in ds.take(1):
            assert isinstance(batch_x, dict)
            assert "original" in batch_x
            assert "segmented" in batch_x
            assert batch_y.shape[-1] == 5

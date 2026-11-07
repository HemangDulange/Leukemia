"""Unit tests for data layer: Pearson correlation dedup, exclude_files filtering, and augmentation."""
from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

from src.config import load_config
from src.data.dataset import build_dataset
from src.data.splits import (
    compute_class_weights,
    compute_near_duplicate_groups,
    discover_dataset_images,
    print_class_distribution,
)
from src.preprocessing.augmentation import apply_color_jitter_and_gray


class TestNearDuplicateGrouping(unittest.TestCase):
    def test_rotations_and_flips_grouped_together(self):
        """Verify an image and its rotations/flips are grouped together (Pearson corr >= 0.995)."""
        pattern = np.zeros((64, 64, 3), dtype=np.uint8)
        pattern[10:30, 10:30] = [220, 100, 50]
        pattern[40:60, 40:60] = [50, 150, 220]

        img = Image.fromarray(pattern)
        img_rot90 = img.rotate(90)
        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)

        diff_pattern = np.zeros((64, 64, 3), dtype=np.uint8)
        diff_pattern[5:55, 25:35] = [200, 200, 200]

        thumbs = np.stack([
            np.array(img),
            np.array(img_rot90),
            np.array(img_flip),
            diff_pattern,
        ], axis=0)

        group_ids, num_groups, images_in_groups, largest_group, num_dup_groups = compute_near_duplicate_groups(
            thumbs, dedup_threshold=0.995
        )

        self.assertEqual(group_ids[0], group_ids[1])
        self.assertEqual(group_ids[0], group_ids[2])
        self.assertNotEqual(group_ids[0], group_ids[3])
        self.assertEqual(images_in_groups, 3)
        self.assertEqual(largest_group, 3)

    def test_safety_threshold_dissolves_large_clusters(self):
        """Verify groups with > 25 images are kept as singletons (all distinct group_ids)."""
        pattern = np.zeros((64, 64, 3), dtype=np.uint8)
        pattern[20:44, 20:44] = [180, 80, 150]
        thumbs = np.repeat(pattern[np.newaxis, ...], 28, axis=0)

        group_ids, num_groups, images_in_groups, largest_group, num_dup_groups = compute_near_duplicate_groups(
            thumbs, dedup_threshold=0.995
        )

        self.assertEqual(len(set(group_ids)), 28)
        self.assertEqual(images_in_groups, 0)
        self.assertEqual(largest_group, 1)

    def test_drop_duplicates_keeps_largest_area_image(self):
        """Verify process_and_deduplicate retains the higher-resolution image in a near-duplicate pair."""
        from src.data.splits import process_and_deduplicate

        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir) / "raw"
            processed_dir = Path(tmp_dir) / "processed"
            splits_dir = Path(tmp_dir) / "splits"
            cls_dir = raw_dir / "AML"
            cls_dir.mkdir(parents=True, exist_ok=True)

            # Create an original image and save at two sizes (1024x768 vs 224x224)
            rng = np.random.RandomState(42)
            base_pattern = np.zeros((768, 1024, 3), dtype=np.uint8)
            base_pattern[200:400, 300:600] = [180, 70, 210]
            img_large = Image.fromarray(base_pattern)
            img_small = img_large.resize((224, 224), Image.Resampling.BILINEAR)

            img_large.save(cls_dir / "large_photo.jpg", "JPEG")
            img_small.save(cls_dir / "small_photo.jpg", "JPEG")

            cfg = load_config()
            cfg.data.raw_dir = raw_dir
            cfg.data.processed_dir = processed_dir
            cfg.data.splits_dir = splits_dir
            cfg.data.classes = ["AML"]
            cfg.data.drop_duplicates = True
            cfg.data.dedup_threshold = 0.995

            df = process_and_deduplicate(cfg)

            # Exactly one image should remain: the large one
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["filename"], "large_photo.jpg")

            # dropped_duplicates.csv should contain small_photo.jpg
            dropped_csv = splits_dir / "dropped_duplicates.csv"
            self.assertTrue(dropped_csv.exists())
            dropped_df = pd.read_csv(dropped_csv)
            self.assertEqual(len(dropped_df), 1)
            self.assertEqual(dropped_df.iloc[0]["filename"], "small_photo.jpg")
            self.assertIn("large_photo.jpg", dropped_df.iloc[0]["kept_path"])

    def test_cross_class_duplicates_are_dropped(self):
        """Verify cross-class duplicates have all members dropped from dataset."""
        from src.data.splits import process_and_deduplicate

        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir) / "raw"
            processed_dir = Path(tmp_dir) / "processed"
            splits_dir = Path(tmp_dir) / "splits"
            (raw_dir / "AML").mkdir(parents=True, exist_ok=True)
            (raw_dir / "Normal").mkdir(parents=True, exist_ok=True)

            # Save the exact same image into both AML and Normal
            pattern = np.zeros((100, 100, 3), dtype=np.uint8)
            pattern[20:60, 20:60] = [200, 100, 150]
            Image.fromarray(pattern).save(raw_dir / "AML" / "shared_cell.jpg")
            Image.fromarray(pattern).save(raw_dir / "Normal" / "shared_cell.jpg")

            cfg = load_config()
            cfg.data.raw_dir = raw_dir
            cfg.data.processed_dir = processed_dir
            cfg.data.splits_dir = splits_dir
            cfg.data.classes = ["AML", "Normal"]
            cfg.data.drop_duplicates = True
            cfg.data.dedup_threshold = 0.995

            df = process_and_deduplicate(cfg)

            # Both members should be dropped due to ambiguous cross-class label
            self.assertEqual(len(df), 0)


class TestExcludeFilesFiltering(unittest.TestCase):
    def test_discover_dataset_images_skips_excluded(self):
        """Verify discover_dataset_images skips any files specified in cfg.data.exclude_files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir)
            cls_dir = raw_dir / "AML"
            cls_dir.mkdir(parents=True, exist_ok=True)

            img1 = cls_dir / "keep_me.jpg"
            img2 = cls_dir / "skip_me.jpg"
            img3 = cls_dir / "skip_me_too.png"

            for p in [img1, img2, img3]:
                Image.new("RGB", (64, 64), color=(100, 100, 100)).save(p)

            cfg = load_config()
            cfg.data.raw_dir = raw_dir
            cfg.data.classes = ["AML"]
            cfg.data.exclude_files = ["AML/skip_me.jpg", "skip_me_too.png"]

            df = discover_dataset_images(cfg)
            discovered_paths = [Path(p).name for p in df["original_path"]]

            self.assertIn("keep_me.jpg", discovered_paths)
            self.assertNotIn("skip_me.jpg", discovered_paths)
            self.assertNotIn("skip_me_too.png", discovered_paths)
            self.assertEqual(len(discovered_paths), 1)


    def test_discover_dataset_images_skips_excluded_class_extensions(self):
        """Verify discover_dataset_images skips files matching exclude_class_extensions per class."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir)
            preb_dir = raw_dir / "PreB"
            aml_dir = raw_dir / "AML"
            preb_dir.mkdir(parents=True, exist_ok=True)
            aml_dir.mkdir(parents=True, exist_ok=True)

            # PreB files: one .jpg (keep) and one .bmp (skip per audit finding)
            Image.new("RGB", (64, 64), color=(100, 100, 100)).save(preb_dir / "smear.jpg")
            Image.new("RGB", (64, 64), color=(10, 10, 10)).save(preb_dir / "dark_cell.bmp")

            # AML file: one .bmp (keep, because exclusion is PreB-only)
            Image.new("RGB", (64, 64), color=(120, 120, 120)).save(aml_dir / "cell.bmp")

            cfg = load_config()
            cfg.data.raw_dir = raw_dir
            cfg.data.classes = ["AML", "PreB"]
            cfg.data.exclude_class_extensions = {"PreB": [".bmp"]}
            cfg.data.exclude_files = []

            df = discover_dataset_images(cfg)
            discovered = [Path(p).name for p in df["original_path"]]

            self.assertIn("smear.jpg", discovered)
            self.assertNotIn("dark_cell.bmp", discovered)
            self.assertIn("cell.bmp", discovered)
            self.assertEqual(len(discovered), 2)


class TestClassWeightsAndDistribution(unittest.TestCase):
    def test_compute_class_weights(self):
        df = pd.DataFrame({"label_index": [0, 0, 0, 0, 1, 1, 2, 3, 4]})
        weights = compute_class_weights(df, num_classes=5)
        self.assertEqual(len(weights), 5)
        self.assertLess(weights[0], weights[1])
        self.assertEqual(weights[2], weights[3])
        self.assertEqual(weights[3], weights[4])

    def test_print_class_distribution(self):
        classes = ["AML", "Early_PreB", "Normal", "PreB", "ProB"]
        train = pd.DataFrame({"class_folder": ["AML"] * 10 + ["Normal"] * 5, "label_index": [0] * 10 + [2] * 5})
        val = pd.DataFrame({"class_folder": ["AML"] * 2 + ["Normal"] * 1, "label_index": [0] * 2 + [2] * 1})
        test = pd.DataFrame({"class_folder": ["AML"] * 2 + ["Normal"] * 1, "label_index": [0] * 2 + [2] * 1})

        dist_df = print_class_distribution(train, val, test, classes)
        self.assertEqual(len(dist_df), 5)
        self.assertEqual(dist_df["total_count"].sum(), 21)


class TestColorRobustness(unittest.TestCase):
    def test_multiplicative_jitter_preserves_black_background(self):
        """Verify color jitter keeps zero-valued background pixels strictly at (0, 0, 0)."""
        cfg = load_config()
        cfg.augmentation.color_jitter = {
            "enabled": True,
            "hue_range": 0.05,
            "saturation_range": [0.6, 1.4],
            "channel_gain_range": [0.85, 1.15],
        }
        cfg.augmentation.random_gray_prob = 0.0

        # Segmented cell simulation: black background (0,0,0) with a colored center cell
        orig = np.full((64, 64, 3), 150.0, dtype=np.float32)
        seg = np.zeros((64, 64, 3), dtype=np.float32)
        seg[20:44, 20:44] = [150.0, 80.0, 200.0]

        tensor_orig = tf.convert_to_tensor(orig, dtype=tf.float32)
        tensor_seg = tf.convert_to_tensor(seg, dtype=tf.float32)

        out_orig, out_seg = apply_color_jitter_and_gray(tensor_orig, tensor_seg, cfg)
        jittered_seg = out_seg.numpy()

        # Background pixels outside [20:44, 20:44] must remain identically 0.0
        bg_slice = jittered_seg[0:15, 0:15]
        np.testing.assert_allclose(bg_slice, 0.0, atol=1e-5)

        # Foreground cell should have non-zero values
        fg_slice = jittered_seg[25:35, 25:35]
        self.assertTrue(np.all(fg_slice > 0.0))

    def test_grayscale_conversion_produces_identical_channels(self):
        """Verify grayscale conversion replicates identical intensity across 3 channels."""
        cfg = load_config()
        cfg.augmentation.color_jitter = {"enabled": False}
        cfg.augmentation.random_gray_prob = 1.0  # Always grayscale

        img = np.full((32, 32, 3), [120.0, 80.0, 200.0], dtype=np.float32)
        tensor_in = tf.convert_to_tensor(img, dtype=tf.float32)
        out_orig, _ = apply_color_jitter_and_gray(tensor_in, tensor_in, cfg)
        gray_arr = out_orig.numpy()

        # Channels 0, 1, and 2 must be identical
        np.testing.assert_allclose(gray_arr[..., 0], gray_arr[..., 1], atol=1e-4)
        np.testing.assert_allclose(gray_arr[..., 1], gray_arr[..., 2], atol=1e-4)


class TestDatasetYieldsDictStructure(unittest.TestCase):
    def test_pipeline_output_shape(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = Path(tmp_dir) / "dummy.png"
            Image.new("RGB", (64, 64), color=(180, 120, 200)).save(img_path)

            df = pd.DataFrame([{
                "original_path": str(img_path),
                "class_folder": "AML",
                "label_index": 0,
                "filename": "dummy.png",
            }])

            cfg = load_config()
            ds = build_dataset(df, cfg, split="val", smoke_test=True, img_size=128)
            for batch_x, batch_y in ds.take(1):
                self.assertIsInstance(batch_x, dict)
                self.assertIn("original", batch_x)
                self.assertIn("segmented", batch_x)
                self.assertEqual(batch_x["original"].shape, (1, 128, 128, 3))
                self.assertEqual(batch_x["segmented"].shape, (1, 128, 128, 3))
                self.assertEqual(batch_y.shape, (1, 5))


if __name__ == "__main__":
    unittest.main()

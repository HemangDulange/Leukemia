"""Unit tests for src/preprocessing/segmentation.py."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.config import load_config
from src.preprocessing.segmentation import (
    apply_hsv_mask,
    build_binary_mask,
    segment_image,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture()
def purple_image_bgr() -> np.ndarray:
    """A solid purple image in BGR — typical Giemsa stain nucleus colour."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Purple/violet in BGR: B=180, G=50, R=140
    img[:, :] = [180, 50, 140]
    return img


@pytest.fixture()
def purple_image_rgb(purple_image_bgr) -> np.ndarray:
    """RGB version of the purple image."""
    return cv2.cvtColor(purple_image_bgr, cv2.COLOR_BGR2RGB)


@pytest.fixture()
def black_image() -> np.ndarray:
    """A solid black image — should produce an empty mask."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


class TestSegmentImage:
    def test_segment_image_returns_tuple(self, cfg, purple_image_bgr):
        mask, seg = segment_image(purple_image_bgr, cfg)
        assert isinstance(mask, np.ndarray)
        assert isinstance(seg, np.ndarray)

    def test_mask_is_binary_and_single_channel(self, cfg, purple_image_bgr):
        mask, _ = segment_image(purple_image_bgr, cfg)
        assert mask.ndim == 2
        assert mask.shape == purple_image_bgr.shape[:2]
        assert mask.dtype == np.uint8
        unique_vals = set(np.unique(mask))
        assert unique_vals.issubset({0, 255})

    def test_segmented_bgr_shape_and_dtype(self, cfg, purple_image_bgr):
        _, seg = segment_image(purple_image_bgr, cfg)
        assert seg.shape == purple_image_bgr.shape
        assert seg.dtype == purple_image_bgr.dtype

    def test_black_image_produces_empty_mask_and_black_output(self, cfg, black_image):
        mask, seg = segment_image(black_image, cfg)
        assert mask.sum() == 0
        assert seg.sum() == 0

    def test_float32_input_preserves_dtype(self, cfg, purple_image_bgr):
        img_f = purple_image_bgr.astype(np.float32)
        mask, seg = segment_image(img_f, cfg)
        assert mask.dtype == np.uint8
        assert seg.dtype == np.float32

    def test_background_is_black(self, cfg, purple_image_bgr):
        # Image half purple, half green (non-nucleus/non-cytoplasm)
        mixed = purple_image_bgr.copy()
        mixed[:, 50:] = [0, 255, 0]  # Green in BGR (Hue ~ 60, not in purple/cytoplasm range)
        mask, seg = segment_image(mixed, cfg)
        # Green side should be masked out to black in seg
        assert np.all(seg[:, 50:] == 0)


class TestBuildBinaryMask:
    def test_returns_single_channel(self, cfg, purple_image_rgb):
        mask = build_binary_mask(purple_image_rgb, cfg)
        assert mask.ndim == 2

    def test_shape_matches_input(self, cfg, purple_image_rgb):
        mask = build_binary_mask(purple_image_rgb, cfg)
        assert mask.shape == purple_image_rgb.shape[:2]

    def test_black_image_produces_empty_mask(self, cfg, black_image):
        mask = build_binary_mask(black_image, cfg)
        assert mask.sum() == 0


class TestApplyHsvMask:
    def test_output_shape_preserved(self, cfg, purple_image_rgb):
        masked = apply_hsv_mask(purple_image_rgb, cfg)
        assert masked.shape == purple_image_rgb.shape

    def test_output_dtype_preserved_uint8(self, cfg, purple_image_rgb):
        masked = apply_hsv_mask(purple_image_rgb, cfg)
        assert masked.dtype == np.uint8

    def test_output_dtype_preserved_float32(self, cfg, purple_image_rgb):
        img_f = purple_image_rgb.astype(np.float32)
        masked = apply_hsv_mask(img_f, cfg)
        assert masked.dtype == np.float32

    def test_black_image_stays_black(self, cfg, black_image):
        masked = apply_hsv_mask(black_image, cfg)
        assert masked.sum() == 0

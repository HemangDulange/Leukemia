"""
Stream-A segmentation: HSV colour thresholding and mask imposition.

Pipeline
--------
BGR image
    │
    ▼
Convert BGR -> HSV
    │
    ├─► Threshold purple/blue nuclei (lower_nuclei, upper_nuclei)
    └─► Threshold pink/purple cytoplasm (lower_cytoplasm, upper_cytoplasm)
    │
Combine masks (bitwise OR)
    │
Morphological open (remove noise) & close (fill holes)
    │
Remove small blobs (connected-component area filtering)
    │
Binary mask (0 or 255)
    │
Impose mask on original (background set to black) -> segmented BGR

All HSV thresholds and morphological parameters are configured via
``configs/config.yaml``.
"""
from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

from src.config import AppConfig, load_config

logger = logging.getLogger(__name__)


def segment_image(
    bgr: np.ndarray,
    cfg: AppConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment a BGR blood-smear image using HSV thresholding and morphology.

    Follows the canonical pipeline:
    1. Convert BGR to HSV colour space.
    2. Threshold for purple/blue nuclei and pink/purple cytoplasm.
    3. Combine masks using bitwise OR.
    4. Apply morphological opening followed by morphological closing.
    5. Remove small connected blobs below ``min_blob_area``.
    6. Impose the cleaned binary mask onto the original BGR image.

    Parameters
    ----------
    bgr:
        BGR image array of shape ``(H, W, 3)`` and dtype ``uint8`` or ``float32``.
    cfg:
        Application configuration. If ``None``, loads default from config.yaml.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(mask, segmented_bgr)``:
        - ``mask``: Single-channel binary mask, dtype ``uint8`` with values 0 or 255.
        - ``segmented_bgr``: Masked BGR image with background pixels set to black (0).
    """
    if cfg is None:
        cfg = load_config()

    seg_cfg = cfg.segmentation

    # Ensure uint8 working image for OpenCV routines
    orig_dtype = bgr.dtype
    if orig_dtype != np.uint8:
        work_bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    else:
        work_bgr = bgr.copy()

    # 1. BGR -> HSV
    hsv = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2HSV)

    # 2. Threshold nuclei and cytoplasm
    lower_nuc = np.array(seg_cfg.lower_nuclei, dtype=np.uint8)
    upper_nuc = np.array(seg_cfg.upper_nuclei, dtype=np.uint8)
    lower_cyto = np.array(seg_cfg.lower_cytoplasm, dtype=np.uint8)
    upper_cyto = np.array(seg_cfg.upper_cytoplasm, dtype=np.uint8)

    mask_nuc = cv2.inRange(hsv, lower_nuc, upper_nuc)
    mask_cyto = cv2.inRange(hsv, lower_cyto, upper_cyto)

    # 3. Combine masks
    mask_combined = cv2.bitwise_or(mask_nuc, mask_cyto)

    # 4. Morphological open (noise removal) & close (fill interior holes)
    k = seg_cfg.morph_kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask_morph = cv2.morphologyEx(
        mask_combined,
        cv2.MORPH_OPEN,
        kernel,
        iterations=seg_cfg.open_iterations,
    )
    mask_morph = cv2.morphologyEx(
        mask_morph,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=seg_cfg.close_iterations,
    )

    # 5. Remove small blobs via connected components
    min_area = seg_cfg.min_blob_area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_morph, connectivity=8)

    binary_mask = np.zeros_like(mask_morph, dtype=np.uint8)
    kept_count = 0
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            binary_mask[labels == label_idx] = 255
            kept_count += 1

    # Safeguard: if all components were smaller than min_area, keep largest non-background component
    if kept_count == 0 and num_labels > 1:
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[largest_label, cv2.CC_STAT_AREA] > 0:
            binary_mask[labels == largest_label] = 255

    # 6. Impose binary mask on original image (background -> black)
    segmented_bgr = cv2.bitwise_and(work_bgr, work_bgr, mask=binary_mask)

    if orig_dtype == np.float32:
        segmented_bgr = segmented_bgr.astype(np.float32)

    return binary_mask, segmented_bgr


def apply_hsv_mask(
    rgb_image: np.ndarray,
    cfg: AppConfig | None = None,
) -> np.ndarray:
    """Apply HSV segmentation to an RGB image and return the masked RGB image.

    Compatibility wrapper around :func:`segment_image` for RGB inputs.

    Parameters
    ----------
    rgb_image:
        RGB image array, shape ``(H, W, 3)``.
    cfg:
        Application configuration.

    Returns
    -------
    np.ndarray
        Masked RGB image with background pixels set to black.
    """
    orig_dtype = rgb_image.dtype
    work_rgb = np.clip(rgb_image, 0, 255).astype(np.uint8) if orig_dtype != np.uint8 else rgb_image
    bgr = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2BGR)

    _, seg_bgr = segment_image(bgr, cfg=cfg)
    seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)

    if orig_dtype == np.float32:
        return seg_rgb.astype(np.float32)
    return seg_rgb


def build_binary_mask(
    rgb_image: np.ndarray,
    cfg: AppConfig | None = None,
) -> np.ndarray:
    """Return the raw single-channel binary mask for an RGB image.

    Compatibility wrapper around :func:`segment_image`.

    Parameters
    ----------
    rgb_image:
        RGB image array, shape ``(H, W, 3)``.
    cfg:
        Application configuration.

    Returns
    -------
    np.ndarray
        Single-channel binary mask, dtype ``uint8`` with values 0 or 255.
    """
    orig_dtype = rgb_image.dtype
    work_rgb = np.clip(rgb_image, 0, 255).astype(np.uint8) if orig_dtype != np.uint8 else rgb_image
    bgr = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2BGR)

    mask, _ = segment_image(bgr, cfg=cfg)
    return mask

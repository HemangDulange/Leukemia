"""
Unified image pair preparation for dual-stream ALL classification.

Enforces a single shared preprocessing pipeline across training, evaluation,
and serving to eliminate train/serve skew:
1. Loads or accepts full-resolution RGB image (from PIL .convert("RGB")).
2. Converts RGB to BGR in one place and segments with existing segment_image().
3. Converts segmented BGR back to RGB.
4. Resizes both original and segmented images to target size using cv2.INTER_AREA.
5. Returns both images as float32 arrays in range [0, 255] (matching Keras
   EfficientNet internal Rescaling layer).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.config import AppConfig, load_config
from src.preprocessing.segmentation import segment_image

logger = logging.getLogger(__name__)

# Prevent PIL DecompressionBombError on large microscopic scans (e.g. 2560x1920)
Image.MAX_IMAGE_PIXELS = None


def prepare_pair(
    rgb_input: Union[np.ndarray, Image.Image],
    cfg: Optional[AppConfig] = None,
    target_size: Optional[int] = None,
    return_mask: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare standardized (original, segmented) pair from a full-resolution RGB image.

    Parameters
    ----------
    rgb_input:
        Full-resolution RGB image as a PIL Image or uint8 NumPy array of shape (H, W, 3).
    cfg:
        Application configuration. If None, loaded from configs/config.yaml.
    target_size:
        Target spatial resolution (e.g. 224). Defaults to cfg.image.size.
    return_mask:
        If True, also returns the resized binary mask (0 or 255 uint8).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray] or Tuple[np.ndarray, np.ndarray, np.ndarray]
        (orig_224, seg_224) or (orig_224, seg_224, mask_224):
        - orig_224: float32 array, shape (size, size, 3), range [0.0, 255.0]
        - seg_224: float32 array, shape (size, size, 3), range [0.0, 255.0]
        - mask_224: (optional) uint8 array, shape (size, size), values {0, 255}
    """
    if cfg is None:
        cfg = load_config()

    size = target_size or cfg.image.size

    # 1. Ensure RGB uint8 NumPy array from input
    if isinstance(rgb_input, Image.Image):
        if rgb_input.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            arr = np.array(rgb_input, dtype=np.float32)
            max_val = float(arr.max()) if arr.size > 0 else 1.0
            if max_val > 255.0:
                arr = (arr / max(max_val, 65535.0)) * 255.0
            rgb_uint8 = np.array(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB"))
        elif rgb_input.mode in ("RGBA", "LA") or (rgb_input.mode == "P" and "transparency" in rgb_input.info):
            rgba = rgb_input.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb_uint8 = np.array(Image.alpha_composite(bg, rgba).convert("RGB"))
        else:
            rgb_uint8 = np.array(rgb_input.convert("RGB"))
    elif isinstance(rgb_input, np.ndarray):
        if rgb_input.dtype != np.uint8:
            rgb_uint8 = np.clip(rgb_input, 0, 255).astype(np.uint8)
        else:
            rgb_uint8 = rgb_input
    else:
        raise TypeError(f"Unsupported image type: {type(rgb_input)}. Expected PIL.Image or np.ndarray.")

    # 2. Convert RGB to BGR for segment_image()
    bgr_full = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

    # 3. Segment full-resolution image using canonical HSV morphology pipeline
    mask_full, seg_bgr_full = segment_image(bgr_full, cfg)

    # 4. Convert segmented BGR back to RGB
    seg_rgb_full = cv2.cvtColor(seg_bgr_full, cv2.COLOR_BGR2RGB)

    # 5. Resize both images to target size with cv2.INTER_AREA (preserves cell details)
    orig_resized = cv2.resize(rgb_uint8, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    seg_resized = cv2.resize(seg_rgb_full, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)

    if return_mask:
        mask_resized = cv2.resize(mask_full, (size, size), interpolation=cv2.INTER_NEAREST)
        return orig_resized, seg_resized, mask_resized

    return orig_resized, seg_resized

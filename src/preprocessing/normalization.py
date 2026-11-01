"""
Image normalisation, resizing, scaling, and stain normalisation.

Implements:
1. Resize and model scaling: Matches backbone expectations (e.g. EfficientNet
   scales to [-1, 1]).
2. Reinhard stain normalisation: Optional colour transfer in Lab colour space
   using a reference blood-smear image from ``configs/config.yaml``.
3. Denormalisation helper for visualisations (Grad-CAM, reports).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf

from src.config import AppConfig, load_config

logger = logging.getLogger(__name__)

# Cached reference image stats: (mean_lab, std_lab)
_CACHED_REF_STATS: Optional[Tuple[np.ndarray, np.ndarray]] = None
_CACHED_REF_PATH: Optional[Path] = None


def compute_lab_stats(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and standard deviation in Lab colour space.

    Parameters
    ----------
    img:
        BGR or RGB uint8 image of shape ``(H, W, 3)``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(mean, std)`` across L, a, b channels.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean = np.mean(lab, axis=(0, 1))
    std = np.std(lab, axis=(0, 1))
    # Prevent zero division
    std = np.maximum(std, 1e-6)
    return mean, std


def get_reference_stats(cfg: AppConfig | None = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load and cache mean and standard deviation for the reference image.

    Parameters
    ----------
    cfg:
        Application configuration.

    Returns
    -------
    Optional[Tuple[np.ndarray, np.ndarray]]
        ``(mean, std)`` in Lab space, or ``None`` if reference image is not set/found.
    """
    global _CACHED_REF_STATS, _CACHED_REF_PATH

    if cfg is None:
        cfg = load_config()

    ref_path = cfg.normalization.reference_image_path
    if not ref_path:
        return None

    ref_path = Path(ref_path)
    if not ref_path.exists():
        logger.warning("Reinhard reference image not found at %s. Skipping stain normalisation.", ref_path)
        return None

    if _CACHED_REF_STATS is not None and _CACHED_REF_PATH == ref_path:
        return _CACHED_REF_STATS

    ref_bgr = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    if ref_bgr is None:
        logger.warning("Could not read reference image %s.", ref_path)
        return None

    _CACHED_REF_STATS = compute_lab_stats(ref_bgr)
    _CACHED_REF_PATH = ref_path
    logger.info("Loaded and cached Reinhard stain stats from %s.", ref_path)
    return _CACHED_REF_STATS


def reinhard_stain_normalization(
    source_bgr: np.ndarray,
    ref_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ref_bgr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Normalise staining appearance of source image to match a reference image.

    Uses Reinhard et al.'s colour transfer method in Lab colour space:
    L' = (L - mean_src) * (std_ref / std_src) + mean_ref

    Parameters
    ----------
    source_bgr:
        Input BGR image of shape ``(H, W, 3)``.
    ref_stats:
        Precomputed ``(mean, std)`` tuple of reference image in Lab space.
    ref_bgr:
        Reference BGR image (used if ref_stats is not provided).

    Returns
    -------
    np.ndarray
        Stain-normalised BGR image, uint8.
    """
    if ref_stats is None:
        if ref_bgr is not None:
            ref_stats = compute_lab_stats(ref_bgr)
        else:
            ref_stats = get_reference_stats()

    if ref_stats is None:
        return source_bgr

    ref_mean, ref_std = ref_stats
    src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_mean, src_std = compute_lab_stats(source_bgr)

    # Transfer colour distribution
    norm_lab = (src_lab - src_mean) * (ref_std / src_std) + ref_mean
    norm_lab = np.clip(norm_lab, 0, 255).astype(np.uint8)

    norm_bgr = cv2.cvtColor(norm_lab, cv2.COLOR_LAB2BGR)
    return norm_bgr


def resize_and_rescale(
    image: np.ndarray,
    target_size: int | Tuple[int, int],
    cfg: AppConfig | None = None,
    is_bgr: bool = True,
) -> np.ndarray:
    """Resize image, apply optional Reinhard normalisation, and scale for model.

    Parameters
    ----------
    image:
        Input image array ``(H, W, 3)`` in ``[0, 255]``.
    target_size:
        Target spatial size (int for square or ``(width, height)``).
    cfg:
        Application configuration.
    is_bgr:
        Whether the input image is in BGR format (True) or RGB format (False).

    Returns
    -------
    np.ndarray
        Float32 image scaled to ``[-1, 1]`` matching EfficientNet backbone input.
    """
    if cfg is None:
        cfg = load_config()

    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    work = np.clip(image, 0, 255).astype(np.uint8) if image.dtype != np.uint8 else image.copy()

    # Optional Reinhard stain normalisation
    if cfg.normalization.use_reinhard:
        if not is_bgr:
            work_bgr = cv2.cvtColor(work, cv2.COLOR_RGB2BGR)
            work_bgr = reinhard_stain_normalization(work_bgr)
            work = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB)
        else:
            work = reinhard_stain_normalization(work)

    # Resize
    resized = cv2.resize(work, target_size, interpolation=cv2.INTER_LINEAR)

    # Return float32 in [0, 255] for EfficientNet internal Rescaling
    return resized.astype(np.float32)


def normalize(image: Union[tf.Tensor, np.ndarray]) -> Union[tf.Tensor, np.ndarray]:
    """Ensure image is float32 in [0, 255] for Keras EfficientNet backbones.

    Keras EfficientNet architectures include built-in Rescaling(1/255) and
    Normalization layers, so the model expects raw pixel values in [0, 255].

    Parameters
    ----------
    image:
        Tensor or array of shape ``(H, W, 3)`` with pixel values in ``[0, 255]``.

    Returns
    -------
    tf.Tensor or np.ndarray
        Float32 tensor or array in ``[0, 255]``.
    """
    if isinstance(image, tf.Tensor):
        return tf.cast(image, tf.float32)
    return np.asarray(image, dtype=np.float32)


def denormalize(image: Union[tf.Tensor, np.ndarray]) -> Union[tf.Tensor, np.ndarray]:
    """Convert float32 image in [0, 255] back to uint8.

    Parameters
    ----------
    image:
        Float32 tensor or array in ``[0, 255]``.

    Returns
    -------
    tf.Tensor or np.ndarray
        Uint8 image with pixel values in ``[0, 255]``.
    """
    if isinstance(image, tf.Tensor):
        return tf.clip_by_value(image, 0.0, 255.0)
    return np.clip(image, 0.0, 255.0).astype(np.uint8)

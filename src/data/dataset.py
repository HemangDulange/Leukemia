"""
tf.data pipelines for the dual-stream ALL classifier.

Builds optimized input pipelines yielding:
    ({"original": x_orig, "segmented": x_seg}, one_hot_label)

Features:
- Stream B: Original blood-smear image (resized, normalized).
- Stream A: Segmented image via HSV color thresholding (clean binary mask imposed, background black).
- Training split: Joint spatial augmentation (RandomFlip, RandomRotation, RandomZoom) across both
  streams, plus RandomContrast on original stream only.
- Validation / Test split: Strictly deterministic preprocessing without augmentation.
- Reproducible seed 42 with prefetching and parallel execution.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import AppConfig
from src.preprocessing.augmentation import augment_pair
from src.preprocessing.normalization import normalize
from src.preprocessing.segmentation import segment_image

logger = logging.getLogger(__name__)

_AUTOTUNE = tf.data.AUTOTUNE


def _load_and_preprocess_pair(
    orig_path: tf.Tensor,
    seg_path: tf.Tensor,
    label: tf.Tensor,
    img_size: int,
    cfg: AppConfig,
) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Load original image, compute HSV segmentation, resize, and normalize."""

    def _py_func(op_bytes, sp_bytes) -> Tuple[np.ndarray, np.ndarray]:
        path_str = op_bytes.numpy().decode() if hasattr(op_bytes, "numpy") else op_bytes.decode()
        orig_bgr = cv2.imread(path_str, cv2.IMREAD_COLOR)

        # Attempt to load pre-segmented image if provided
        seg_path_str = sp_bytes.numpy().decode() if hasattr(sp_bytes, "numpy") else sp_bytes.decode()
        seg_bgr = cv2.imread(seg_path_str, cv2.IMREAD_COLOR) if seg_path_str else None

        if orig_bgr is None:
            # Fallback zero-filled array if file read failed
            orig_bgr = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            seg_bgr = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        elif seg_bgr is None:
            # Segment using HSV color threshold pipeline
            _, seg_bgr = segment_image(orig_bgr, cfg)

        # Convert to RGB
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)

        # Resize to model input shape
        orig_resized = cv2.resize(orig_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        seg_resized = cv2.resize(seg_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        return orig_resized.astype(np.float32), seg_resized.astype(np.float32)

    orig_arr, seg_arr = tf.py_function(
        func=_py_func,
        inp=[orig_path, seg_path],
        Tout=[tf.float32, tf.float32],
    )

    orig_arr.set_shape([img_size, img_size, 3])
    seg_arr.set_shape([img_size, img_size, 3])

    # Ensure float32 in [0, 255] for EfficientNet backbones (which contain internal Rescaling)
    orig_norm = normalize(orig_arr)
    seg_norm = normalize(seg_arr)

    return (orig_norm, seg_norm), label


def build_dataset(
    df: pd.DataFrame,
    cfg: AppConfig,
    split: str,
    smoke_test: bool = False,
) -> tf.data.Dataset:
    """Build a ``tf.data.Dataset`` yielding ({"original": x_orig, "segmented": x_seg}, label).

    Parameters
    ----------
    df:
        DataFrame containing ``original_path``, ``segmented_path``, and ``label_index``.
    cfg:
        Application configuration.
    split:
        One of ``"train"``, ``"val"``, or ``"test"``.
    smoke_test:
        If ``True``, samples a small subset and uses ``cfg.image.smoke_test_size``.

    Returns
    -------
    tf.data.Dataset
        Batched and prefetched dataset yielding:
        ``({"original": batch_orig, "segmented": batch_seg}, batch_labels)``
    """
    is_train = split == "train"
    img_size = cfg.image.smoke_test_size if smoke_test else cfg.image.size
    batch_size = cfg.training.batch_size
    n_classes = cfg.model.num_classes

    if smoke_test:
        df = df.sample(
            n=min(cfg.training.smoke_test_samples, len(df)),
            random_state=cfg.seed,
        ).reset_index(drop=True)
        logger.info("[smoke-test] %s dataset: %d samples at %dpx", split, len(df), img_size)
    else:
        logger.info("%s dataset: %d samples at %dpx", split, len(df), img_size)

    if len(df) == 0:
        logger.warning("Empty dataframe passed for %s split.", split)

    orig_paths = df["original_path"].astype(str).values
    seg_paths = df["segmented_path"].astype(str).values
    labels = df["label_index"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((orig_paths, seg_paths, labels))

    if is_train:
        ds = ds.shuffle(
            buffer_size=max(len(df), 1),
            seed=cfg.seed,
            reshuffle_each_iteration=True,
        )

    # Load and preprocess
    ds = ds.map(
        lambda o, s, l: _load_and_preprocess_pair(o, s, l, img_size, cfg),
        num_parallel_calls=_AUTOTUNE,
    )

    # Augment ONLY the train set
    if is_train:
        ds = ds.map(
            lambda imgs, lbl: (
                augment_pair(
                    (imgs[0], imgs[1]),  # (original, segmented)
                    lbl,
                    cfg,
                    training=True,
                )
            ),
            num_parallel_calls=_AUTOTUNE,
        )
        ds = ds.map(
            lambda imgs, lbl: ({"original": imgs[0], "segmented": imgs[1]}, tf.one_hot(lbl, n_classes)),
            num_parallel_calls=_AUTOTUNE,
        )
    else:
        # No augmentation for val and test
        ds = ds.map(
            lambda imgs, lbl: ({"original": imgs[0], "segmented": imgs[1]}, tf.one_hot(lbl, n_classes)),
            num_parallel_calls=_AUTOTUNE,
        )

    ds = ds.batch(batch_size).prefetch(_AUTOTUNE)
    return ds

"""
tf.data pipelines for the dual-stream ALL classifier.

Builds optimized input pipelines yielding:
    ({"original": x_orig, "segmented": x_seg}, one_hot_label)

Features:
- Stream B: Original blood-smear image from cached processed PNGs.
- Stream A: Segmented image from cached processed PNGs (created by prepare_pair).
  NEVER reads from the dataset's pre-made Segmented/ directory to guarantee train/serve symmetry.
- Training split: Joint spatial augmentation (RandomFlip, RandomRotation, RandomZoom) across both
  streams, plus RandomContrast on original stream only.
- Validation / Test split: Strictly deterministic without augmentation.
- Reproducible seed 42 with prefetching and parallel execution.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

from src.config import AppConfig
from src.preprocessing.augmentation import augment_pair
from src.preprocessing.normalization import normalize
from src.preprocessing.pair import prepare_pair

logger = logging.getLogger(__name__)

_AUTOTUNE = tf.data.AUTOTUNE


def _load_and_preprocess_pair(
    orig_path: tf.Tensor,
    seg_path: tf.Tensor,
    label: tf.Tensor,
    img_size: int,
    cfg: AppConfig,
) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Load cached orig/seg PNGs, or generate on-the-fly using prepare_pair if cache is missing."""

    def _py_func(op_bytes, sp_bytes) -> Tuple[np.ndarray, np.ndarray]:
        op_str = op_bytes.numpy().decode() if hasattr(op_bytes, "numpy") else op_bytes.decode()
        sp_str = sp_bytes.numpy().decode() if hasattr(sp_bytes, "numpy") else sp_bytes.decode()

        # 1. Try reading the cached PNG files directly
        orig_bgr = cv2.imread(op_str, cv2.IMREAD_COLOR) if op_str else None
        seg_bgr = cv2.imread(sp_str, cv2.IMREAD_COLOR) if sp_str else None

        if orig_bgr is not None and seg_bgr is not None:
            orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
            seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)

            if orig_rgb.shape[0] != img_size or orig_rgb.shape[1] != img_size:
                orig_rgb = cv2.resize(orig_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
            if seg_rgb.shape[0] != img_size or seg_rgb.shape[1] != img_size:
                seg_rgb = cv2.resize(seg_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)

            return orig_rgb.astype(np.float32), seg_rgb.astype(np.float32)

        # 2. Fallback: load original image and segment on-the-fly via prepare_pair
        if op_str and Path(op_str).exists():
            try:
                pil_img = Image.open(op_str).convert("RGB")
                orig_arr, seg_arr = prepare_pair(pil_img, cfg=cfg, target_size=img_size)
                return orig_arr, seg_arr
            except Exception as e:
                logger.error("Error reading/preparing %s: %s", op_str, e)

        # 3. Fallback: zero-filled array if file read failed
        return np.zeros((img_size, img_size, 3), dtype=np.float32), np.zeros((img_size, img_size, 3), dtype=np.float32)

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
    img_size: Optional[int] = None,
) -> tf.data.Dataset:
    """Build a ``tf.data.Dataset`` yielding ({"original": x_orig, "segmented": x_seg}, label).

    Parameters
    ----------
    df:
        DataFrame containing ``original_path`` and ``label_index``.
    cfg:
        Application configuration.
    split:
        One of ``"train"``, ``"val"``, or ``"test"``.
    smoke_test:
        If ``True``, samples a small subset and uses ``cfg.image.smoke_test_size``.
    img_size:
        Optional explicit image resolution. If None, uses smoke_test_size or cfg.image.size.

    Returns
    -------
    tf.data.Dataset
        Batched and prefetched dataset yielding:
        ``({"original": batch_orig, "segmented": batch_seg}, batch_labels)``
    """
    is_train = split == "train"
    if img_size is None:
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

    # Resolve paths to cached data/processed/orig/ and data/processed/seg/
    orig_paths_list = []
    seg_paths_list = []
    processed_orig_root = cfg.data.processed_dir / "orig"
    processed_seg_root = cfg.data.processed_dir / "seg"

    for _, row in df.iterrows():
        raw_path = Path(str(row["original_path"]))
        class_folder = str(row.get("class_folder") or raw_path.parent.name)
        stem = raw_path.stem

        cached_orig = processed_orig_root / class_folder / f"{stem}.png"
        cached_seg = processed_seg_root / class_folder / f"{stem}.png"

        if cached_orig.exists() and cached_seg.exists():
            orig_paths_list.append(str(cached_orig))
            seg_paths_list.append(str(cached_seg))
        else:
            # Fallback to source path; _load_and_preprocess_pair will call prepare_pair on-the-fly
            orig_paths_list.append(str(raw_path))
            seg_paths_list.append("")

    orig_paths = np.array(orig_paths_list, dtype=object)
    seg_paths = np.array(seg_paths_list, dtype=object)
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

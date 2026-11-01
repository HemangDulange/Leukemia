"""
Training-time augmentation using Keras layers (flip, rotation, zoom, contrast).

Reference implementation from notebooks:
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10)
    ])

For the dual-stream architecture:
- Geometric transforms (Flip, Rotation, Zoom) are applied jointly across both
  Stream A (segmented) and Stream B (original) to ensure exact spatial alignment.
- Contrast adjustment is applied to Stream B (original image) to avoid
  distorting the masked boundary of Stream A.
- Augmentations are active during training only.
"""
from __future__ import annotations

import logging
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers

from src.config import AppConfig, load_config

logger = logging.getLogger(__name__)

# Cache for the compiled augmentation model
_GEOMETRIC_AUGMENTER: tf.keras.Sequential | None = None
_CONTRAST_LAYER: layers.RandomContrast | None = None


def build_augmentation_layers(cfg: AppConfig | None = None) -> tf.keras.Sequential:
    """Build standard Keras sequential augmentation pipeline for single images.

    Matches reference notebook architecture:
    [RandomFlip, RandomRotation, RandomZoom, RandomContrast]

    Parameters
    ----------
    cfg:
        Application configuration.

    Returns
    -------
    tf.keras.Sequential
        Keras sequential model containing augmentation layers.
    """
    if cfg is None:
        cfg = load_config()

    aug = cfg.augmentation
    flip_mode = "horizontal_and_vertical" if (aug.horizontal_flip and aug.vertical_flip) else (
        "horizontal" if aug.horizontal_flip else ("vertical" if aug.vertical_flip else None)
    )

    layer_list = []
    if flip_mode:
        layer_list.append(layers.RandomFlip(flip_mode, seed=cfg.seed))
    if aug.rotation_range > 0:
        layer_list.append(layers.RandomRotation(aug.rotation_range, seed=cfg.seed))
    if aug.zoom_range > 0:
        layer_list.append(layers.RandomZoom(aug.zoom_range, seed=cfg.seed))
    if aug.contrast_range > 0:
        layer_list.append(layers.RandomContrast(aug.contrast_range, seed=cfg.seed))

    return tf.keras.Sequential(layer_list, name="data_augmentation")


def _get_pair_augmenters(cfg: AppConfig) -> Tuple[tf.keras.Sequential, layers.RandomContrast]:
    """Get or create cached augmentation layers for dual-stream pair processing."""
    global _GEOMETRIC_AUGMENTER, _CONTRAST_LAYER

    if _GEOMETRIC_AUGMENTER is None or _CONTRAST_LAYER is None:
        aug = cfg.augmentation
        flip_mode = "horizontal_and_vertical" if (aug.horizontal_flip and aug.vertical_flip) else (
            "horizontal" if aug.horizontal_flip else "vertical"
        )
        geom_layers = [
            layers.RandomFlip(flip_mode, seed=cfg.seed),
            layers.RandomRotation(aug.rotation_range, fill_mode="constant", fill_value=0.0, seed=cfg.seed),
            layers.RandomZoom(aug.zoom_range, fill_mode="constant", fill_value=0.0, seed=cfg.seed),
        ]
        _GEOMETRIC_AUGMENTER = tf.keras.Sequential(geom_layers, name="joint_geometric_augmenter")
        _CONTRAST_LAYER = layers.RandomContrast(aug.contrast_range, seed=cfg.seed)

    return _GEOMETRIC_AUGMENTER, _CONTRAST_LAYER


def augment_pair(
    imgs: Tuple[tf.Tensor, tf.Tensor],
    label: tf.Tensor,
    cfg: AppConfig,
    training: bool = True,
) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Apply Keras layer augmentations to an (original, segmented) pair.

    Jointly transforms Stream original and Stream segmented along channels
    so that geometric distortions (flip, rotation, zoom) match exactly.
    Then applies contrast augmentation to the original stream only.
    Both streams are maintained strictly in [0.0, 255.0].

    Parameters
    ----------
    imgs:
        Tuple of ``(stream_orig, stream_seg)`` tensors with shape ``(H, W, 3)`` in ``[0, 255]``.
    label:
        Label tensor (passed through unchanged).
    cfg:
        Application configuration.
    training:
        Whether to apply augmentations (True for training, False for eval).

    Returns
    -------
    Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]
        Augmented ``((stream_orig, stream_seg), label)`` in ``[0, 255]``.
    """
    if not training:
        return imgs, label

    stream_orig, stream_seg = imgs
    geom_model, contrast_layer = _get_pair_augmenters(cfg)

    # Concatenate along channel dimension: (H, W, 6)
    # This allows joint geometric transformations with identical random parameters
    concat_img = tf.concat([stream_orig, stream_seg], axis=-1)
    # Add batch dimension for Keras layer: (1, H, W, 6)
    batched = tf.expand_dims(concat_img, axis=0)

    aug_batched = geom_model(batched, training=True)
    aug_concat = tf.squeeze(aug_batched, axis=0)

    # Split back into Stream Orig (0:3) and Stream Seg (3:6)
    aug_orig = aug_concat[..., 0:3]
    aug_seg = aug_concat[..., 3:6]

    # Contrast layer on original stream only
    aug_orig_batched = tf.expand_dims(aug_orig, axis=0)
    aug_orig_contrast = contrast_layer(aug_orig_batched, training=True)
    aug_orig = tf.squeeze(aug_orig_contrast, axis=0)

    # Clip both streams to valid [0.0, 255.0] range
    aug_orig = tf.clip_by_value(aug_orig, 0.0, 255.0)
    aug_seg = tf.clip_by_value(aug_seg, 0.0, 255.0)

    return (aug_orig, aug_seg), label

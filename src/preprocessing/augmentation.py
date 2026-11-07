"""
Training-time augmentation using Keras layers and colour robustness operations.

For the dual-stream architecture:
- Geometric transforms (Flip, Rotation, Zoom) are applied jointly across both
  Stream A (segmented) and Stream B (original) to ensure exact spatial alignment.
- Contrast adjustment is applied to Stream B (original image).
- Colour robustness (Color Jitter & Random Grayscale):
  - Applied to BOTH streams (train only, after segmentation).
  - Hue shift +-0.05
  - Saturation x0.6-1.4
  - Per-channel gain 0.85-1.15 (multiplicative only, so the black background of the segmented stream stays black)
  - Random grayscale (p=0.3, converted to grayscale and repeated to 3 channels)
  - Both switchable off via config.
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
    """Build standard Keras sequential augmentation pipeline for single images."""
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


def apply_color_jitter_and_gray(
    stream_orig: tf.Tensor,
    stream_seg: tf.Tensor,
    cfg: AppConfig,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply multiplicative color jitter and random grayscale to both streams."""
    aug_cfg = cfg.augmentation
    jitter_cfg = aug_cfg.color_jitter if isinstance(aug_cfg.color_jitter, dict) else {}
    jitter_enabled = jitter_cfg.get("enabled", True) if isinstance(jitter_cfg, dict) else bool(jitter_cfg)

    # 1. Multiplicative Color Jitter (preserves black 0 background on segmented stream)
    if jitter_enabled:
        hue_range = float(jitter_cfg.get("hue_range", 0.05))
        sat_range = jitter_cfg.get("saturation_range", [0.6, 1.4])
        gain_range = jitter_cfg.get("channel_gain_range", [0.85, 1.15])

        # Random hue shift in [-hue_range, hue_range]
        if hue_range > 0:
            hue_delta = tf.random.uniform([], -hue_range, hue_range)
            stream_orig = tf.image.adjust_hue(stream_orig / 255.0, hue_delta) * 255.0
            stream_seg = tf.image.adjust_hue(stream_seg / 255.0, hue_delta) * 255.0

        # Random saturation factor in [sat_min, sat_max]
        if sat_range and len(sat_range) == 2 and sat_range[1] > sat_range[0]:
            sat_factor = tf.random.uniform([], float(sat_range[0]), float(sat_range[1]))
            stream_orig = tf.image.adjust_saturation(stream_orig / 255.0, sat_factor) * 255.0
            stream_seg = tf.image.adjust_saturation(stream_seg / 255.0, sat_factor) * 255.0

        # Multiplicative per-channel gain in [gain_min, gain_max]
        if gain_range and len(gain_range) == 2:
            channel_gain = tf.random.uniform([1, 1, 3], float(gain_range[0]), float(gain_range[1]))
            stream_orig = stream_orig * channel_gain
            stream_seg = stream_seg * channel_gain

    # 2. Random Grayscale (repeated to 3 channels)
    gray_prob = float(getattr(aug_cfg, "random_gray_prob", 0.3))
    if gray_prob > 0.0:
        do_gray = tf.random.uniform([]) < gray_prob

        def _to_gray(im: tf.Tensor) -> tf.Tensor:
            g = tf.image.rgb_to_grayscale(im)
            return tf.repeat(g, 3, axis=-1)

        stream_orig, stream_seg = tf.cond(
            do_gray,
            lambda: (_to_gray(stream_orig), _to_gray(stream_seg)),
            lambda: (stream_orig, stream_seg),
        )

    # Maintain strictly in [0.0, 255.0]
    stream_orig = tf.clip_by_value(stream_orig, 0.0, 255.0)
    stream_seg = tf.clip_by_value(stream_seg, 0.0, 255.0)

    return stream_orig, stream_seg


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
    Then applies multiplicative color jitter and random grayscale to both streams.
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

    # 1. Joint geometric transformations (Flip, Rotation, Zoom)
    concat_img = tf.concat([stream_orig, stream_seg], axis=-1)
    batched = tf.expand_dims(concat_img, axis=0)

    aug_batched = geom_model(batched, training=True)
    aug_concat = tf.squeeze(aug_batched, axis=0)

    # Split back into Stream Orig (0:3) and Stream Seg (3:6)
    aug_orig = aug_concat[..., 0:3]
    aug_seg = aug_concat[..., 3:6]

    # 2. Contrast layer on original stream only
    aug_orig_batched = tf.expand_dims(aug_orig, axis=0)
    aug_orig_contrast = contrast_layer(aug_orig_batched, training=True)
    aug_orig = tf.squeeze(aug_orig_contrast, axis=0)

    # 3. Colour robustness (Color jitter & random grayscale on both streams)
    aug_orig, aug_seg = apply_color_jitter_and_gray(aug_orig, aug_seg, cfg)

    return (aug_orig, aug_seg), label

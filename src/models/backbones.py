"""
Feature-extraction backbone factory supporting EfficientNetB0–B5.

Builds pretrained backbones with ImageNet weights, include_top=False,
and Global Average Pooling. Supports unique layer name prefixing for
multi-stream networks and partial unfreezing (top N layers) for fine-tuning.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple, Type

import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)

# Registry for EfficientNetB0-B5 architectures
_BACKBONE_REGISTRY: Dict[str, Type[keras.Model]] = {
    "EfficientNetB0": keras.applications.EfficientNetB0,
    "EfficientNetB1": keras.applications.EfficientNetB1,
    "EfficientNetB2": keras.applications.EfficientNetB2,
    "EfficientNetB3": keras.applications.EfficientNetB3,
    "EfficientNetB4": keras.applications.EfficientNetB4,
    "EfficientNetB5": keras.applications.EfficientNetB5,
    "ResNet50": keras.applications.ResNet50,
    "MobileNetV2": keras.applications.MobileNetV2,
}


def build_backbone(
    name: str = "EfficientNetB0",
    img_size: int | Tuple[int, int] = 224,
    stream_prefix: str = "",
    weights: str | None = "imagenet",
    unfreeze_top_n: int = 0,
) -> keras.Model:
    """Build a feature-extraction backbone with ImageNet weights and Global Average Pooling.

    Parameters
    ----------
    name:
        Backbone architecture name (e.g. 'EfficientNetB0', 'EfficientNetB5').
    img_size:
        Input spatial resolution. Integer or (height, width).
    stream_prefix:
        Prefix prepended to each layer name to guarantee unique layer names
        when multiple backbones exist in a dual-stream model.
    weights:
        Pretrained weights to load (default: 'imagenet', or None).
    unfreeze_top_n:
        Number of top layers to leave trainable. If 0 (default), all backbone
        layers are frozen.

    Returns
    -------
    keras.Model
        Model taking input tensor (H, W, 3) and outputting feature vector (embedding_dim,).
    """
    if name not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unsupported backbone '{name}'. Supported architectures: {list(_BACKBONE_REGISTRY.keys())}"
        )

    if isinstance(img_size, int):
        input_shape = (img_size, img_size, 3)
    else:
        input_shape = (img_size[0], img_size[1], 3)

    backbone_fn = _BACKBONE_REGISTRY[name]
    model_name = f"{stream_prefix}_backbone" if stream_prefix else f"{name.lower()}_backbone"

    # Instantiate base model with ImageNet weights, no classification top, and global average pooling
    try:
        base_model = backbone_fn(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            pooling="avg",
            name=model_name,
        )
    except TypeError:
        base_model = backbone_fn(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            pooling="avg",
        )
        base_model._name = model_name

    # Prefix all layer names to ensure 100% uniqueness across streams
    if stream_prefix:
        for layer in base_model.layers:
            layer._name = f"{stream_prefix}_{layer.name}"

    # Configure freezing / unfreezing of top N layers
    set_trainable_layers(base_model, unfreeze_top_n=unfreeze_top_n)

    logger.info(
        "Built backbone: %s | input_shape=%s | unfreeze_top_n=%d | total_layers=%d | trainable_weights=%d",
        model_name,
        input_shape,
        unfreeze_top_n,
        len(base_model.layers),
        len(base_model.trainable_weights),
    )

    return base_model


def set_trainable_layers(
    model: keras.Model,
    unfreeze_top_n: int = 0,
) -> None:
    """Freeze the model or unfreeze its top N layers for fine-tuning.

    Preserves BatchNormalization in inference mode during fine-tuning
    to prevent destruction of learned mean/variance statistics.

    Parameters
    ----------
    model:
        Keras Model whose layers will be frozen/unfrozen.
    unfreeze_top_n:
        Number of top layers to unfreeze. 0 means all layers are frozen.
    """
    if unfreeze_top_n <= 0:
        model.trainable = False
        for layer in model.layers:
            layer.trainable = False
    else:
        model.trainable = True
        n_layers = len(model.layers)
        freeze_until = max(0, n_layers - unfreeze_top_n)

        for i, layer in enumerate(model.layers):
            if i < freeze_until:
                layer.trainable = False
            else:
                # Do not unfreeze BatchNorm layers during transfer fine-tuning
                if isinstance(layer, keras.layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True

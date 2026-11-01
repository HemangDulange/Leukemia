"""
Dual-stream classifier for Acute Lymphoblastic Leukemia subtypes.

Architecture:
-------------
Input "original"  (Stream B) ---> Backbone B (orig_*) ---> GAP vector (D,)
                                                                   │
                                                               Concatenate (2D,)
                                                                   │
Input "segmented" (Stream A) ---> Backbone A (seg_*)  ---> GAP vector (D,)
                                                                   │
                                                           BatchNormalization
                                                                   │
                                                              Dropout(0.4)
                                                                   │
                                                            Dense(4, softmax)

Both streams employ separate pretrained backbones (e.g. EfficientNetB0-B5) with
unique layer names, global average pooling, and support for freezing or unfreezing
the top N layers for progressive fine-tuning.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import tensorflow as tf
from tensorflow import keras

from src.config import AppConfig, load_config
from src.models.backbones import build_backbone, set_trainable_layers

logger = logging.getLogger(__name__)


def build_dual_stream_model(
    cfg: AppConfig | None = None,
    img_size: int | None = None,
    unfreeze_top_n: int = 0,
    weights: str | None = "imagenet",
) -> keras.Model:
    """Construct and compile the dual-stream classification model.

    Parameters
    ----------
    cfg:
        Application configuration. If None, loaded from configs/config.yaml.
    img_size:
        Spatial image size. Defaults to ``cfg.image.size`` (or smoke_test_size).
    unfreeze_top_n:
        Number of top layers to unfreeze in each backbone for fine-tuning.
        If 0 (default), backbones are frozen and only head weights are trained.
    weights:
        Pretrained weights for backbones (default 'imagenet').

    Returns
    -------
    keras.Model
        Compiled dual-stream Keras Model with inputs:
        ``{"original": (B, H, W, 3), "segmented": (B, H, W, 3)}``
        and softmax output: ``(B, num_classes)``.
    """
    if cfg is None:
        cfg = load_config()

    size = img_size or cfg.image.size
    input_shape = (size, size, cfg.image.channels)

    # 1. Inputs for the two streams
    input_orig = keras.Input(shape=input_shape, name="original")
    input_seg = keras.Input(shape=input_shape, name="segmented")

    # 2. Two SEPARATE backbones with unique layer names
    backbone_orig = build_backbone(
        name=cfg.model.backbone,
        img_size=size,
        stream_prefix="orig",
        weights=weights,
        unfreeze_top_n=unfreeze_top_n,
    )
    backbone_seg = build_backbone(
        name=cfg.model.backbone,
        img_size=size,
        stream_prefix="seg",
        weights=weights,
        unfreeze_top_n=unfreeze_top_n,
    )

    # 3. Feature extraction (each backbone already has Global Average Pooling)
    feat_orig = backbone_orig(input_orig)
    feat_seg = backbone_seg(input_seg)

    # 4. Fusion via Concatenate
    concat = keras.layers.Concatenate(name="concat_fusion")([feat_orig, feat_seg])

    # 5. BatchNorm
    bn = keras.layers.BatchNormalization(name="batch_norm")(concat)

    # 6. Dropout(0.4)
    dropout = keras.layers.Dropout(cfg.model.dropout_rate, name="dropout")(bn)

    # 7. Dense softmax
    output = keras.layers.Dense(
        cfg.model.num_classes,
        activation="softmax",
        name="output",
    )(dropout)

    # Assemble Model
    model = keras.Model(
        inputs={"original": input_orig, "segmented": input_seg},
        outputs=output,
        name="dual_stream_all_classifier",
    )

    optimizer = keras.optimizers.Adam(learning_rate=cfg.training.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=cfg.training.label_smoothing
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc", multi_label=False),
        ],
    )

    trainable_count = sum(tf.size(v).numpy() for v in model.trainable_variables)
    total_count = model.count_params()
    logger.info(
        "Dual-stream model built: %d total parameters, %d trainable parameters (unfreeze_top_n=%d)",
        total_count,
        trainable_count,
        unfreeze_top_n,
    )

    return model


def unfreeze_backbone_layers(
    model: keras.Model,
    unfreeze_top_n: int = 20,
    learning_rate: float = 1e-5,
) -> keras.Model:
    """Unfreeze the top N layers of both backbones for fine-tuning and recompile with lower LR.

    Parameters
    ----------
    model:
        Trained dual-stream Keras model.
    unfreeze_top_n:
        Number of top layers to unfreeze in each backbone.
    learning_rate:
        Fine-tuning learning rate (typically 1e-5 to 1e-6).

    Returns
    -------
    keras.Model
        Recompiled model with partially unfrozen backbones.
    """
    for layer in model.layers:
        if isinstance(layer, keras.Model) and ("orig_backbone" in layer.name or "seg_backbone" in layer.name):
            set_trainable_layers(layer, unfreeze_top_n=unfreeze_top_n)

    # Recompile with lower learning rate for gentle fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=model.loss,
        metrics=model.metrics,
    )

    trainable_count = sum(tf.size(v).numpy() for v in model.trainable_variables)
    logger.info(
        "Model fine-tuning configured: unfreezing top %d layers | trainable parameters: %d | LR: %e",
        unfreeze_top_n,
        trainable_count,
        learning_rate,
    )
    return model

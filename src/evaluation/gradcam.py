"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for the Dual-Stream ALL Classifier.

Generates visual explanations from the last convolutional layer of the
segmented-stream backbone (Stream A) to highlight the intracellular regions
(nuclei, cytoplasmic borders, chromatin texture) driving the model's prediction.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


def find_segmented_conv_layer(model: keras.Model) -> Tuple[keras.Model, keras.layers.Layer]:
    """Locate the segmented-stream backbone and its final convolutional layer.

    Parameters
    ----------
    model:
        Trained dual-stream Keras functional model.

    Returns
    -------
    Tuple[keras.Model, keras.layers.Layer]
        ``(seg_backbone, last_conv_layer)``
    """
    # 1. Locate segmented backbone submodel
    seg_backbone = None
    for layer in model.layers:
        if "seg" in layer.name and hasattr(layer, "layers"):
            seg_backbone = layer
            break

    if seg_backbone is None:
        # Fallback to first submodel if naming differs
        submodels = [l for l in model.layers if hasattr(l, "layers")]
        if len(submodels) >= 2:
            seg_backbone = submodels[1]
        elif len(submodels) == 1:
            seg_backbone = submodels[0]
        else:
            raise ValueError("No submodel backbone found inside dual-stream model.")

    # 2. Search in reverse for the last 4D conv layer inside the backbone
    last_conv = None
    for layer in reversed(seg_backbone.layers):
        # Look for rank-4 output tensor or conv in name/type
        out_shape = getattr(layer, "output_shape", None)
        is_conv = any(
            t in layer.__class__.__name__.lower()
            for t in ["conv2d", "depthwiseconv2d", "separableconv2d"]
        ) or "conv" in layer.name.lower()

        if is_conv or (out_shape is not None and len(out_shape) == 4):
            last_conv = layer
            break

    if last_conv is None:
        raise ValueError(f"Could not locate convolutional layer in {seg_backbone.name}")

    logger.debug("Grad-CAM target: backbone=%s, layer=%s", seg_backbone.name, last_conv.name)
    return seg_backbone, last_conv


def compute_gradcam_heatmap(
    model: keras.Model,
    orig_tensor: np.ndarray | tf.Tensor,
    seg_tensor: np.ndarray | tf.Tensor,
    class_idx: Optional[int] = None,
) -> np.ndarray:
    """Compute Grad-CAM heatmap from the segmented backbone's final conv layer.

    Parameters
    ----------
    model:
        Trained dual-stream model.
    orig_tensor:
        Preprocessed original image tensor of shape ``(1, H, W, 3)``.
    seg_tensor:
        Preprocessed segmented image tensor of shape ``(1, H, W, 3)``.
    class_idx:
        Target class index. If None, uses top predicted class index.

    Returns
    -------
    np.ndarray
        2D heatmap array of shape ``(H, W)`` with normalized values in ``[0, 1]``.
    """
    orig_tensor = tf.convert_to_tensor(orig_tensor, dtype=tf.float32)
    seg_tensor = tf.convert_to_tensor(seg_tensor, dtype=tf.float32)

    seg_backbone, last_conv = find_segmented_conv_layer(model)

    # Find the original backbone
    orig_backbone = None
    for layer in model.layers:
        if "orig" in layer.name and hasattr(layer, "layers") and layer is not seg_backbone:
            orig_backbone = layer
            break
    if orig_backbone is None:
        submodels = [l for l in model.layers if hasattr(l, "layers") and l is not seg_backbone]
        orig_backbone = submodels[0] if submodels else None

    # Intermediate model for segmented stream: outputs (conv_features, gap_features)
    grad_seg_model = keras.Model(
        inputs=seg_backbone.input,
        outputs=[last_conv.output, seg_backbone.output],
    )

    # Locate fusion and head layers
    concat_layer = next(l for l in model.layers if isinstance(l, keras.layers.Concatenate) or "concat" in l.name.lower())
    bn_layer = next((l for l in model.layers if isinstance(l, keras.layers.BatchNormalization)), None)
    dropout_layer = next((l for l in model.layers if isinstance(l, keras.layers.Dropout)), None)
    output_layer = next(l for l in reversed(model.layers) if isinstance(l, keras.layers.Dense) or "output" in l.name.lower())

    with tf.GradientTape() as tape:
        # Segmented stream forward pass
        conv_outputs, feat_seg = grad_seg_model(seg_tensor, training=False)
        tape.watch(conv_outputs)

        # Original stream forward pass
        feat_orig = orig_backbone(orig_tensor, training=False) if orig_backbone else feat_seg

        # Head forward pass
        fused = concat_layer([feat_orig, feat_seg])
        if bn_layer is not None:
            fused = bn_layer(fused, training=False)
        if dropout_layer is not None:
            fused = dropout_layer(fused, training=False)
        preds = output_layer(fused)

        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))

        loss = preds[:, class_idx]

    # Compute gradients of class score with respect to conv outputs
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        # Fallback uniform heatmap if gradients cannot be traced
        h, w = seg_tensor.shape[1], seg_tensor.shape[2]
        return np.zeros((h, w), dtype=np.float32)

    # Global average pooling of gradients: (channels,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature map activations by pooled gradients
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)

    # Apply ReLU to retain features having positive influence on target class
    cam = tf.maximum(cam, 0)
    max_val = tf.math.reduce_max(cam)

    if max_val > 0:
        cam = cam / max_val

    cam_np = cam.numpy()

    # Resize to original spatial dimensions
    target_h, target_w = int(seg_tensor.shape[1]), int(seg_tensor.shape[2])
    heatmap_resized = cv2.resize(cam_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(heatmap_resized, 0.0, 1.0)


def overlay_heatmap(
    heatmap: np.ndarray,
    rgb_image: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a Grad-CAM heatmap over an RGB image.

    Parameters
    ----------
    heatmap:
        Float array of shape ``(H, W)`` in ``[0, 1]``.
    rgb_image:
        Base RGB image (uint8) of shape ``(H, W, 3)``.
    alpha:
        Heatmap opacity blend factor.
    colormap:
        OpenCV colormap (default ``cv2.COLORMAP_JET``).

    Returns
    -------
    np.ndarray
        Blended RGB uint8 image of shape ``(H, W, 3)``.
    """
    h, w = rgb_image.shape[:2]
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h))

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    blended = (alpha * heatmap_rgb + (1.0 - alpha) * rgb_image).astype(np.uint8)
    return blended

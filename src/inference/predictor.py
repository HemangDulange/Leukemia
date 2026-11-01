"""
Single-image inference engine for the ALL Dual-Stream Classifier.

Loads the trained model and canonical class metadata once, then exposes
``predict(pil_image) -> dict`` which reuses the exact preprocessing pipeline:
1. Converts PIL Image to BGR for OpenCV.
2. Applies HSV color threshold segmentation (segment_image from src.preprocessing.segmentation).
3. Resizes and normalizes both streams with EfficientNet scaling ([-1, 1]).
4. Runs dual-input model forward pass.
5. Computes Grad-CAM heatmap overlay from the segmented stream's final conv layer.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from src.config import AppConfig, load_config
from src.evaluation.gradcam import compute_gradcam_heatmap, overlay_heatmap
from src.preprocessing.normalization import normalize
from src.preprocessing.segmentation import segment_image

logger = logging.getLogger(__name__)


class Predictor:
    """Production predictor wrapping the dual-stream model for single-image inference.

    Parameters
    ----------
    cfg:
        Application configuration. If None, loaded from configs/config.yaml.
    model_path:
        Optional path to override the saved model location.
    """

    def __init__(
        self,
        cfg: Optional[AppConfig] = None,
        model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._cfg = cfg or load_config()
        self._model_path = Path(model_path) if model_path else self._cfg.paths.model_path
        self._class_meta = self._load_class_names()
        self._model = self._load_model()

    def _load_model(self) -> tf.keras.Model:
        """Load and cache the trained Keras model using local weights or Hugging Face Hub."""
        from src.inference.model_loader import load_trained_model
        return load_trained_model(cfg=self._cfg, custom_path=self._model_path)

    def _load_class_names(self) -> Dict[str, Any]:
        """Load canonical class mappings from models/class_names.json."""
        path = self._cfg.paths.class_names_file
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)

        # Fallback to config classes if file does not exist yet
        classes = self._cfg.data.classes
        display_mapping = {
            "Benign": "Hematogones",
            "Early": "Early Pre-B ALL",
            "Pre": "Pre-B ALL",
            "Pre-B": "Pre-B ALL",
            "Pro": "Pro-B ALL",
        }
        return {
            "class_order": classes,
            "display_names": {c: display_mapping.get(c, c) for c in classes},
            "index_to_folder": {str(i): c for i, c in enumerate(classes)},
            "index_to_display": {str(i): display_mapping.get(c, c) for i, c in enumerate(classes)},
        }

    def _preprocess_input(
        self,
        bgr_image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process image through exact preprocessing pipeline.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ``(orig_norm_batch, seg_norm_batch, binary_mask, seg_rgb)``
        """
        size = self._cfg.image.size

        # 1. Exact HSV color threshold segmentation (background -> black)
        binary_mask, seg_bgr = segment_image(bgr_image, self._cfg)

        # 2. Convert to RGB
        orig_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)

        # 3. Resize
        orig_resized = cv2.resize(orig_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        seg_resized = cv2.resize(seg_rgb, (size, size), interpolation=cv2.INTER_LINEAR)

        # 4. Normalize to [-1, 1] for EfficientNet backbones
        orig_norm = normalize(orig_resized.astype(np.float32))
        seg_norm = normalize(seg_resized.astype(np.float32))

        # Add batch dimension: (1, H, W, 3)
        orig_batch = np.expand_dims(orig_norm, axis=0)
        seg_batch = np.expand_dims(seg_norm, axis=0)

        return orig_batch, seg_batch, binary_mask, seg_rgb

    def predict(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        generate_gradcam: bool = True,
    ) -> Dict[str, Any]:
        """Run single-image inference and return comprehensive predictions.

        Parameters
        ----------
        image:
            Input blood-smear image as a PIL Image, uint8 RGB numpy array, or file path.
        generate_gradcam:
            Whether to compute Grad-CAM heatmap overlay from the segmented stream.

        Returns
        -------
        dict
            Prediction results containing:
            - ``predicted_label``: Display name of the top class (e.g. 'Hematogones').
            - ``predicted_class``: Alias for predicted_label.
            - ``predicted_folder``: Internal class folder name ('Benign', 'Early', etc.).
            - ``label_index``: Integer class index in [0, 3].
            - ``class_probabilities``: Dict mapping display name -> probability.
            - ``probabilities``: List of (display_name, probability) tuples.
            - ``segmented_image``: Segmented PIL Image with black background.
            - ``mask``: Binary mask PIL Image (0 or 255).
            - ``original_image``: Resized original PIL Image.
            - ``gradcam_overlay``: Grad-CAM overlay PIL Image (or None).
            - ``confidence``: Probability of top predicted class.
        """
        # Convert input to BGR numpy array for OpenCV routines
        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Could not load image file from {image}")
            orig_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        elif isinstance(image, Image.Image):
            orig_pil = image.convert("RGB")
            rgb_arr = np.array(orig_pil)
            bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected 3-channel image array, got shape {image.shape}")
            # If dtype is float in [0, 1], scale to uint8
            if np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
            # Assume RGB array from PIL/standard loaders
            orig_pil = Image.fromarray(image)
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # 1. Preprocess both streams
        orig_batch, seg_batch, binary_mask, seg_rgb = self._preprocess_input(bgr)

        # 2. Dual-input model forward pass
        proba = self._model.predict(
            {"original": orig_batch, "segmented": seg_batch},
            verbose=0,
        )[0]

        # 3. Extract top class
        label_idx = int(np.argmax(proba))
        folder_name = self._class_meta["index_to_folder"][str(label_idx)]
        display_name = self._class_meta["index_to_display"][str(label_idx)]
        confidence = float(proba[label_idx])

        # 4. Format probability distributions
        prob_dict: Dict[str, float] = {}
        prob_list: List[Tuple[str, float]] = []
        for i, p in enumerate(proba):
            c_name = self._class_meta["index_to_display"][str(i)]
            val = float(p)
            prob_dict[c_name] = round(val, 4)
            prob_list.append((c_name, val))

        # 5. Convert visual outputs to PIL images
        size = self._cfg.image.size
        orig_rgb_resized = cv2.resize(np.array(orig_pil), (size, size), interpolation=cv2.INTER_LINEAR)
        seg_rgb_resized = cv2.resize(seg_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(binary_mask, (size, size), interpolation=cv2.INTER_NEAREST)

        segmented_pil = Image.fromarray(seg_rgb_resized)
        mask_pil = Image.fromarray(mask_resized)
        original_pil = Image.fromarray(orig_rgb_resized)

        # 6. Grad-CAM overlay from segmented-stream backbone
        gradcam_pil = None
        if generate_gradcam:
            try:
                heatmap = compute_gradcam_heatmap(
                    model=self._model,
                    orig_tensor=orig_batch,
                    seg_tensor=seg_batch,
                    class_idx=label_idx,
                )
                overlay_rgb = overlay_heatmap(heatmap, orig_rgb_resized, alpha=0.45)
                gradcam_pil = Image.fromarray(overlay_rgb)
            except Exception as e:
                logger.warning("Grad-CAM generation failed: %s", e)

        return {
            "predicted_label": display_name,
            "predicted_class": display_name,
            "predicted_folder": folder_name,
            "label_index": label_idx,
            "confidence": confidence,
            "class_probabilities": prob_dict,
            "probabilities": prob_list,
            "segmented_image": segmented_pil,
            "masked_image": seg_rgb_resized,
            "mask": mask_pil,
            "original_image": original_pil,
            "gradcam_overlay": gradcam_pil,
        }

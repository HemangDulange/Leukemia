"""
Single-image inference engine for the ALL Dual-Stream Classifier.

Loads the trained model and canonical class metadata once, then exposes
``predict(pil_image) -> dict`` which reuses the exact preprocessing pipeline:
1. Calls src.preprocessing.pair:prepare_pair on the full-resolution RGB image.
2. Formats both streams into (1, 224, 224, 3) float32 in [0, 255] (matching training).
3. Runs dual-input model forward pass.
4. Computes Grad-CAM heatmap overlay from the segmented stream's final conv layer.
5. Emits preview PIL images (original, segmented, mask, gradcam) derived from the exact same output.
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
from src.data.splits import load_rgb_image
from src.evaluation.gradcam import compute_gradcam_heatmap, overlay_heatmap
from src.preprocessing.pair import prepare_pair

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None


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
        """Load canonical class mappings from models/class_names.json or config."""
        path = self._cfg.paths.class_names_file
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)

        # Fallback to config classes if file does not exist yet
        classes = self._cfg.data.classes
        display_mapping = self._cfg.data.display_names or {c: c for c in classes}
        return {
            "class_order": classes,
            "display_names": {c: display_mapping.get(c, c) for c in classes},
            "index_to_folder": {str(i): c for i, c in enumerate(classes)},
            "index_to_display": {str(i): display_mapping.get(c, c) for i, c in enumerate(classes)},
        }

    def predict(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        generate_gradcam: bool = True,
    ) -> Dict[str, Any]:
        """Run single-image inference using shared prepare_pair preprocessing.

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
            - ``label_index``: Integer class index in [0, num_classes-1].
            - ``class_probabilities``: Dict mapping display name -> probability.
            - ``probabilities``: List of (display_name, probability) tuples.
            - ``segmented_image``: Segmented PIL Image with black background.
            - ``mask``: Binary mask PIL Image (0 or 255).
            - ``original_image``: Resized original PIL Image.
            - ``gradcam_overlay``: Grad-CAM overlay PIL Image (or None).
            - ``confidence``: Probability of top predicted class.
        """
        # 1. Standardize input to PIL Image in RGB format
        if isinstance(image, (str, Path)):
            pil_image = load_rgb_image(image)
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected 3-channel image array, got shape {image.shape}")
            if np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
                uint8_arr = (image * 255).astype(np.uint8)
            else:
                uint8_arr = image.astype(np.uint8)
            pil_image = Image.fromarray(uint8_arr)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # 2. Shared preprocessing via prepare_pair (eliminates train/serve skew)
        orig_arr, seg_arr, mask_arr = prepare_pair(
            pil_image,
            cfg=self._cfg,
            target_size=self._cfg.image.size,
            return_mask=True,
        )

        # 3. Add batch dimension: (1, H, W, 3) float32 in [0, 255]
        orig_batch = np.expand_dims(orig_arr, axis=0)
        seg_batch = np.expand_dims(seg_arr, axis=0)

        # 4. Dual-input model forward pass
        proba = self._model.predict(
            {"original": orig_batch, "segmented": seg_batch},
            verbose=0,
        )[0]

        # 5. Extract top class
        label_idx = int(np.argmax(proba))
        folder_name = self._class_meta["index_to_folder"][str(label_idx)]
        display_name = self._class_meta["index_to_display"][str(label_idx)]
        confidence = float(proba[label_idx])

        # 6. Format probability distributions
        prob_dict: Dict[str, float] = {}
        prob_list: List[Tuple[str, float]] = []
        for i, p in enumerate(proba):
            c_name = self._class_meta["index_to_display"][str(i)]
            val = float(p)
            prob_dict[c_name] = round(val, 4)
            prob_list.append((c_name, val))

        # 7. Convert visual outputs directly from the prepare_pair results
        orig_uint8 = np.clip(orig_arr, 0.0, 255.0).astype(np.uint8)
        seg_uint8 = np.clip(seg_arr, 0.0, 255.0).astype(np.uint8)

        original_pil = Image.fromarray(orig_uint8)
        segmented_pil = Image.fromarray(seg_uint8)
        mask_pil = Image.fromarray(mask_arr.astype(np.uint8))

        # 8. Grad-CAM overlay from segmented-stream backbone
        gradcam_pil = None
        if generate_gradcam:
            try:
                heatmap = compute_gradcam_heatmap(
                    model=self._model,
                    orig_tensor=orig_batch,
                    seg_tensor=seg_batch,
                    class_idx=label_idx,
                )
                overlay_bgr = overlay_heatmap(heatmap, orig_uint8, alpha=0.45)
                gradcam_pil = Image.fromarray(overlay_bgr)
            except Exception as e:
                logger.warning("Could not generate Grad-CAM overlay: %s", e)

        return {
            "predicted_label": display_name,
            "predicted_class": display_name,
            "predicted_folder": folder_name,
            "label_index": label_idx,
            "confidence": confidence,
            "class_probabilities": prob_dict,
            "probabilities": prob_list,
            "segmented_image": segmented_pil,
            "mask": mask_pil,
            "original_image": original_pil,
            "gradcam_overlay": gradcam_pil,
        }

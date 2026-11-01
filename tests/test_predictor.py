"""Unit tests for the single-image Predictor and Grad-CAM integration."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from src.config import load_config
from src.models.dual_stream import build_dual_stream_model


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture()
def sample_blood_smear_image() -> Image.Image:
    """Generate a synthetic blood-smear PIL Image containing a stained nucleus."""
    # 224x224 RGB image with light plasma background
    img = np.full((224, 224, 3), (235, 230, 240), dtype=np.uint8)

    # Draw simulated pink/purple cytoplasm
    cv2.circle(img, (112, 112), 65, (210, 160, 200), thickness=-1)

    # Draw simulated dark purple/blue Giemsa-stained nucleus
    cv2.circle(img, (112, 112), 45, (130, 60, 170), thickness=-1)

    return Image.fromarray(img, mode="RGB")


@pytest.fixture(scope="module")
def temporary_smoke_model(cfg, tmp_path_factory) -> Path:
    """Build and save a lightweight smoke-test model for testing Predictor without GPU training."""
    model_dir = tmp_path_factory.mktemp("test_models")
    model_path = model_dir / "temp_dual_stream.keras"

    model = build_dual_stream_model(
        cfg=cfg,
        img_size=cfg.image.smoke_test_size,
        weights=None,
    )
    model.save(str(model_path))
    return model_path


class TestPredictor:
    def test_predictor_raises_if_model_missing(self, cfg, tmp_path):
        """Predictor must raise FileNotFoundError if target model path does not exist."""
        from src.inference.predictor import Predictor

        missing_path = tmp_path / "non_existent_model.keras"
        with pytest.raises(FileNotFoundError, match="Saved model not found"):
            Predictor(cfg, model_path=missing_path)

    def test_predict_with_sample_image(self, cfg, temporary_smoke_model, sample_blood_smear_image):
        """Verify predict(pil_image) returns all required keys and valid types."""
        from src.inference.predictor import Predictor

        predictor = Predictor(cfg, model_path=temporary_smoke_model)

        result = predictor.predict(sample_blood_smear_image, generate_gradcam=True)

        # 1. Verify required dictionary keys exist
        assert "predicted_label" in result
        assert "class_probabilities" in result
        assert "segmented_image" in result
        assert "mask" in result
        assert "original_image" in result
        assert "gradcam_overlay" in result
        assert "confidence" in result

        # 2. Verify display name format
        valid_display_names = {"Hematogones", "Early Pre-B ALL", "Pre-B ALL", "Pro-B ALL"}
        assert result["predicted_label"] in valid_display_names

        # 3. Verify class probabilities
        probs = result["class_probabilities"]
        assert isinstance(probs, dict)
        assert len(probs) == cfg.model.num_classes
        assert set(probs.keys()) == valid_display_names
        prob_sum = sum(probs.values())
        assert abs(prob_sum - 1.0) < 1e-2

        # 4. Verify PIL image types
        assert isinstance(result["segmented_image"], Image.Image)
        assert isinstance(result["mask"], Image.Image)
        assert isinstance(result["original_image"], Image.Image)
        assert isinstance(result["gradcam_overlay"], Image.Image)

        # 5. Verify mask dimensions
        assert result["mask"].size == (cfg.image.size, cfg.image.size)
        assert result["segmented_image"].size == (cfg.image.size, cfg.image.size)

"""Unit tests for model construction, input/output shapes, and fine-tuning."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config
from src.models.backbones import build_backbone
from src.models.dual_stream import build_dual_stream_model, unfreeze_backbone_layers


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def smoke_model(cfg):
    """Build a smoke-test-sized model once per module without loading heavy weights."""
    return build_dual_stream_model(
        cfg,
        img_size=cfg.image.smoke_test_size,
        weights=None,
    )


class TestDualStreamModel:
    def test_input_and_output_shapes(self, cfg, smoke_model):
        """Verify inputs have expected shape (None, H, W, 3) and output has shape (None, 4)."""
        size = cfg.image.smoke_test_size

        # Verify dual inputs
        assert len(smoke_model.inputs) == 2
        input_dict = {inp.name.split(":")[0]: inp for inp in smoke_model.inputs}
        assert "original" in input_dict
        assert "segmented" in input_dict

        assert input_dict["original"].shape.as_list() == [None, size, size, 3]
        assert input_dict["segmented"].shape.as_list() == [None, size, size, 3]

        # Verify output shape
        assert smoke_model.output.shape.as_list() == [None, cfg.model.num_classes]

    def test_two_separate_backbones_unique_names(self, smoke_model):
        """Verify the model uses two separate backbones with unique layer names."""
        submodels = [l for l in smoke_model.layers if hasattr(l, "layers")]
        assert len(submodels) >= 2, "Model must contain at least two backbone submodels."

        orig_sub = next(l for l in submodels if "orig" in l.name)
        seg_sub = next(l for l in submodels if "seg" in l.name)

        assert orig_sub is not seg_sub, "Backbones must be two separate object instances."

        orig_layer_names = {l.name for l in orig_sub.layers}
        seg_layer_names = {l.name for l in seg_sub.layers}

        # Names across streams must have zero overlap
        overlap = orig_layer_names & seg_layer_names
        assert len(overlap) == 0, f"Layer names must be unique across streams. Overlapping: {overlap}"

    def test_forward_pass_prediction(self, cfg, smoke_model):
        """Verify forward pass produces correct batch shape and valid softmax probabilities."""
        batch = 3
        size = cfg.image.smoke_test_size
        dummy_orig = np.random.rand(batch, size, size, 3).astype(np.float32)
        dummy_seg = np.random.rand(batch, size, size, 3).astype(np.float32)

        out = smoke_model.predict({"original": dummy_orig, "segmented": dummy_seg}, verbose=0)
        assert out.shape == (batch, cfg.model.num_classes)
        # Softmax probabilities sum to 1.0 across classes
        np.testing.assert_allclose(out.sum(axis=1), np.ones(batch), atol=1e-5)

    def test_freeze_and_unfreeze_top_n_layers(self, cfg):
        """Verify freezing/unfreezing controls trainable parameter counts."""
        size = cfg.image.smoke_test_size

        # Model with completely frozen backbones
        frozen_model = build_dual_stream_model(cfg, img_size=size, unfreeze_top_n=0, weights=None)
        trainable_frozen = len(frozen_model.trainable_weights)

        # Unfreeze top 10 layers for fine-tuning
        unfrozen_model = unfreeze_backbone_layers(frozen_model, unfreeze_top_n=10)
        trainable_unfrozen = len(unfrozen_model.trainable_weights)

        assert trainable_unfrozen > trainable_frozen, "Unfreezing top N layers must increase trainable weights."

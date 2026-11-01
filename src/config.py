"""
Central configuration loader.

Reads configs/config.yaml (or a custom path) and exposes a typed
``AppConfig`` dataclass.  All other modules import from here.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml


# ── Config root ──────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "config.yaml"


# ── Sub-configs ──────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    raw_dir: Path
    splits_dir: Path
    original_subdir: str
    segmented_subdir: str
    classes: List[str]
    val_fraction: float
    test_fraction: float
    processed_dir: Path = field(default_factory=lambda: _REPO_ROOT / "data" / "processed")


@dataclass
class ImageConfig:
    size: int
    smoke_test_size: int
    channels: int


@dataclass
class SegmentationConfig:
    lower_nuclei: List[int]
    upper_nuclei: List[int]
    lower_cytoplasm: List[int]
    upper_cytoplasm: List[int]
    morph_kernel_size: int = 5
    open_iterations: int = 1
    close_iterations: int = 2
    min_blob_area: int = 500

    # Backward-compatible property aliases
    @property
    def lower_purple(self) -> List[int]:
        return self.lower_nuclei

    @property
    def upper_purple(self) -> List[int]:
        return self.upper_nuclei

    @property
    def lower_blue(self) -> List[int]:
        return self.lower_cytoplasm

    @property
    def upper_blue(self) -> List[int]:
        return self.upper_cytoplasm


@dataclass
class NormalizationConfig:
    use_reinhard: bool = False
    reference_image_path: Path | None = None


@dataclass
class AugmentationConfig:
    rotation_range: float
    zoom_range: float
    contrast_range: float = 0.10
    horizontal_flip: bool = True
    vertical_flip: bool = False
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    brightness_range: Tuple[float, float] = (0.85, 1.15)


@dataclass
class ModelConfig:
    backbone: str
    dropout_rate: float
    embedding_dim: int
    num_classes: int


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    label_smoothing: float
    smoke_test_samples: int
    smoke_test_epochs: int
    phase1_epochs: int = 15
    phase1_lr: float = 1e-3
    phase2_epochs: int = 35
    phase2_lr: float = 1e-5
    unfreeze_top_n: int = 30


@dataclass
class PathsConfig:
    models_dir: Path
    model_filename: str
    class_names_file: Path
    reports_dir: Path
    figures_dir: Path
    metrics_dir: Path
    hf_repo_id: str = "HemangDulange/leukemia-all-classifier"
    hf_filename: str = "dual_stream.keras"

    @property
    def model_path(self) -> Path:
        """Full path to the saved model file."""
        return self.models_dir / self.model_filename


@dataclass
class AppConfig:
    """Top-level configuration object.  Constructed by :func:`load_config`."""

    seed: int
    data: DataConfig
    image: ImageConfig
    segmentation: SegmentationConfig
    normalization: NormalizationConfig
    augmentation: AugmentationConfig
    model: ModelConfig
    training: TrainingConfig
    paths: PathsConfig


# ── Loader ───────────────────────────────────────────────────────────────────
def load_config(config_path: Path | str | None = None) -> AppConfig:
    """Load and validate ``config.yaml``, returning a typed ``AppConfig``.

    Parameters
    ----------
    config_path:
        Path to a YAML config file.  Defaults to ``configs/config.yaml``
        relative to the repository root.

    Returns
    -------
    AppConfig
        Fully populated configuration dataclass.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)

    root = _REPO_ROOT

    data_raw = raw["data"]
    processed_val = data_raw.get("processed_dir", "data/processed")
    data = DataConfig(
        raw_dir=root / data_raw["raw_dir"],
        splits_dir=root / data_raw["splits_dir"],
        original_subdir=data_raw["original_subdir"],
        segmented_subdir=data_raw["segmented_subdir"],
        classes=data_raw["classes"],
        val_fraction=data_raw["val_fraction"],
        test_fraction=data_raw["test_fraction"],
        processed_dir=root / processed_val,
    )

    img_raw = raw["image"]
    image = ImageConfig(
        size=img_raw["size"],
        smoke_test_size=img_raw["smoke_test_size"],
        channels=img_raw["channels"],
    )

    seg_raw = raw["segmentation"]
    segmentation = SegmentationConfig(
        lower_nuclei=seg_raw.get("lower_nuclei", seg_raw.get("lower_purple", [100, 40, 40])),
        upper_nuclei=seg_raw.get("upper_nuclei", seg_raw.get("upper_purple", [160, 255, 255])),
        lower_cytoplasm=seg_raw.get("lower_cytoplasm", seg_raw.get("lower_blue", [120, 20, 100])),
        upper_cytoplasm=seg_raw.get("upper_cytoplasm", seg_raw.get("upper_blue", [170, 100, 255])),
        morph_kernel_size=seg_raw.get("morph_kernel_size", 5),
        open_iterations=seg_raw.get("open_iterations", 1),
        close_iterations=seg_raw.get("close_iterations", 2),
        min_blob_area=seg_raw.get("min_blob_area", 500),
    )

    norm_raw = raw.get("normalization", {})
    ref_path = norm_raw.get("reference_image_path")
    normalization = NormalizationConfig(
        use_reinhard=norm_raw.get("use_reinhard", False),
        reference_image_path=root / ref_path if ref_path else None,
    )

    aug_raw = raw["augmentation"]
    augmentation = AugmentationConfig(
        rotation_range=aug_raw.get("rotation_range", 0.10),
        zoom_range=aug_raw.get("zoom_range", 0.10),
        contrast_range=aug_raw.get("contrast_range", 0.10),
        horizontal_flip=aug_raw.get("horizontal_flip", True),
        vertical_flip=aug_raw.get("vertical_flip", False),
        width_shift_range=aug_raw.get("width_shift_range", 0.1),
        height_shift_range=aug_raw.get("height_shift_range", 0.1),
        brightness_range=tuple(aug_raw.get("brightness_range", [0.85, 1.15])),
    )

    mdl_raw = raw["model"]
    model = ModelConfig(
        backbone=mdl_raw["backbone"],
        dropout_rate=mdl_raw["dropout_rate"],
        embedding_dim=mdl_raw["embedding_dim"],
        num_classes=mdl_raw["num_classes"],
    )

    tr_raw = raw["training"]
    training = TrainingConfig(
        batch_size=tr_raw["batch_size"],
        epochs=tr_raw.get("epochs", 50),
        learning_rate=tr_raw.get("learning_rate", 0.001),
        label_smoothing=tr_raw["label_smoothing"],
        smoke_test_samples=tr_raw["smoke_test_samples"],
        smoke_test_epochs=tr_raw["smoke_test_epochs"],
        phase1_epochs=tr_raw.get("phase1_epochs", 15),
        phase1_lr=tr_raw.get("phase1_lr", 0.001),
        phase2_epochs=tr_raw.get("phase2_epochs", 35),
        phase2_lr=tr_raw.get("phase2_lr", 0.00001),
        unfreeze_top_n=tr_raw.get("unfreeze_top_n", 30),
    )

    p_raw = raw["paths"]
    paths = PathsConfig(
        models_dir=root / p_raw["models_dir"],
        model_filename=p_raw["model_filename"],
        class_names_file=root / p_raw["class_names_file"],
        reports_dir=root / p_raw["reports_dir"],
        figures_dir=root / p_raw["figures_dir"],
        metrics_dir=root / p_raw["metrics_dir"],
        hf_repo_id=p_raw.get("hf_repo_id", "HemangDulange/leukemia-all-classifier"),
        hf_filename=p_raw.get("hf_filename", "dual_stream.keras"),
    )

    cfg = AppConfig(
        seed=raw["seed"],
        data=data,
        image=image,
        segmentation=segmentation,
        normalization=normalization,
        augmentation=augmentation,
        model=model,
        training=training,
        paths=paths,
    )

    _apply_seed(cfg.seed)
    return cfg


def _apply_seed(seed: int) -> None:
    """Set Python, NumPy, and TF random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # lazy import so config is usable without TF
        tf.random.set_seed(seed)
    except ImportError:
        pass

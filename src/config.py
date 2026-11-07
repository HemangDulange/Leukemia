"""
Central configuration loader.

Reads configs/config.yaml (or a custom path) and exposes a typed
``AppConfig`` dataclass.  All other modules import from here.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    classes: List[str]
    val_fraction: float
    test_fraction: float
    display_names: Dict[str, str] = field(default_factory=dict)
    processed_dir: Path = field(default_factory=lambda: _REPO_ROOT / "data" / "processed")
    dedup_threshold: float = 0.995
    exclude_files: List[str] = field(default_factory=list)


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
    color_jitter: Dict[str, Any] = field(default_factory=dict)
    random_gray_prob: float = 0.3


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
def _resolve_path(p: str | Path, root: Path) -> Path:
    p_obj = Path(p)
    if p_obj.is_absolute():
        return p_obj
    return root / p_obj


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
    raw_dir_resolved = _resolve_path(data_raw["raw_dir"], root)
    if not raw_dir_resolved.exists() and (root / "data" / "fake_raw").exists():
        raw_dir_resolved = root / "data" / "fake_raw"

    data = DataConfig(
        raw_dir=raw_dir_resolved,
        splits_dir=_resolve_path(data_raw["splits_dir"], root),
        classes=data_raw["classes"],
        display_names=data_raw.get("display_names", {}),
        val_fraction=data_raw["val_fraction"],
        test_fraction=data_raw["test_fraction"],
        processed_dir=_resolve_path(processed_val, root),
        dedup_threshold=float(data_raw.get("dedup_threshold", 0.995)),
        exclude_files=list(data_raw.get("exclude_files", [])),
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
        reference_image_path=_resolve_path(ref_path, root) if ref_path else None,
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
        color_jitter=aug_raw.get("color_jitter", {}),
        random_gray_prob=float(aug_raw.get("random_gray_prob", 0.3)),
    )

    mdl_raw = raw["model"]
    # Derive num_classes dynamically from classes list
    num_classes = len(data.classes)
    model = ModelConfig(
        backbone=mdl_raw["backbone"],
        dropout_rate=mdl_raw["dropout_rate"],
        embedding_dim=mdl_raw["embedding_dim"],
        num_classes=num_classes,
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
        models_dir=_resolve_path(p_raw["models_dir"], root),
        model_filename=p_raw["model_filename"],
        class_names_file=_resolve_path(p_raw["class_names_file"], root),
        reports_dir=_resolve_path(p_raw["reports_dir"], root),
        figures_dir=_resolve_path(p_raw["figures_dir"], root),
        metrics_dir=_resolve_path(p_raw["metrics_dir"], root),
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

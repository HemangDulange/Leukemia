"""
Model loader with local file detection and Hugging Face Hub fallback.

GitHub enforces a 100 MB per-file limit. When deploying to Streamlit
Community Cloud or other cloud runtimes without Git LFS:
1. Checks for local model weights at ``models/dual_stream.keras``.
2. If absent or empty, downloads the weights from a configurable Hugging Face
   Hub repository (e.g. ``HemangDulange/leukemia-all-classifier``) and caches it locally.
3. Automatically sets CPU thread pools to stay within Streamlit Cloud's ~1 GB memory limit.
"""
from __future__ import annotations

import logging
from pathlib import Path
import urllib.request
from typing import Optional

import tensorflow as tf

from src.config import AppConfig, load_config

logger = logging.getLogger(__name__)


def configure_low_memory_runtime() -> None:
    """Tune TensorFlow threading and memory footprint for <= 1 GB RAM cloud environments."""
    try:
        # Constrain thread pools to avoid memory bloat in constrained containers
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(2)
    except Exception as e:
        logger.debug("Could not adjust thread pool settings: %s", e)

    _patch_keras_deserialization_compat()


def _patch_keras_deserialization_compat() -> None:
    """Ensure backwards compatibility when loading models across Keras 3 minor versions."""
    try:
        import keras
        orig_from_config = keras.layers.Layer.from_config
        if getattr(orig_from_config, "_compat_patched", False):
            return

        @classmethod
        def patched_from_config(cls, config):
            cfg = dict(config)
            cfg.pop("quantization_config", None)
            return orig_from_config.__func__(cls, cfg)

        patched_from_config._compat_patched = True
        keras.layers.Layer.from_config = patched_from_config
    except Exception as e:
        logger.debug("Could not patch Keras layer deserialization: %s", e)


def download_from_huggingface(
    repo_id: str,
    filename: str,
    target_path: Path,
) -> Path:
    """Download model weights from Hugging Face Hub with progress tracking.

    Attempts official ``huggingface_hub`` library first, falling back to
    direct streaming HTTP download via standard library ``urllib.request``.

    Parameters
    ----------
    repo_id:
        Hugging Face Hub repository ID (e.g. 'HemangDulange/leukemia-all-classifier').
    filename:
        File to download (e.g. 'dual_stream.keras').
    target_path:
        Local path where downloaded weights should be saved.

    Returns
    -------
    Path
        Destination path of the cached weights.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target_path.with_suffix(".downloading")

    logger.info("Attempting download from Hugging Face: %s/%s -> %s", repo_id, filename, target_path)

    # 1. Try official huggingface_hub client if installed
    try:
        from huggingface_hub import hf_hub_download
        cached_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(target_path.parent),
            local_dir_use_symlinks=False,
        )
        downloaded = Path(cached_file)
        if downloaded != target_path and downloaded.exists():
            downloaded.replace(target_path)
        logger.info("Successfully downloaded weights via huggingface_hub: %s", target_path)
        return target_path
    except ImportError:
        logger.info("huggingface_hub package not available; using standard HTTP stream fallback.")
    except Exception as exc:
        logger.warning("huggingface_hub download failed (%s); attempting direct HTTP stream fallback.", exc)

    # 2. Fallback: Direct streaming download via HTTPS
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    logger.info("Streaming download from URL: %s", url)

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "LeukemiaClassifier-ModelLoader/1.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as response, open(temp_target, "wb") as out_file:
            total_size = int(response.headers.get("content-length", 0))
            downloaded_bytes = 0
            chunk_size = 1024 * 1024  # 1 MB chunk

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded_bytes += len(chunk)
                if total_size > 0:
                    pct = (downloaded_bytes / total_size) * 100
                    logger.debug("Download progress: %.1f%% (%d / %d bytes)", pct, downloaded_bytes, total_size)

        temp_target.replace(target_path)
        size_mb = target_path.stat().st_size / (1024 * 1024)
        logger.info("Download completed successfully: %s (%.2f MB)", target_path, size_mb)
        return target_path

    except Exception as e:
        if temp_target.exists():
            temp_target.unlink()
        raise RuntimeError(
            f"Failed to download model weights from Hugging Face ({url}).\n"
            f"Reason: {e}\n\n"
            "To fix: either run `python scripts/train.py` to train locally/on Kaggle, "
            f"or upload `dual_stream.keras` to Hugging Face repository '{repo_id}'."
        ) from e


def resolve_model_path(
    cfg: Optional[AppConfig] = None,
    custom_path: Optional[Path | str] = None,
) -> Path:
    """Find local model weights or download them from Hugging Face Hub.

    Parameters
    ----------
    cfg:
        Application configuration.
    custom_path:
        Optional path overriding configuration.

    Returns
    -------
    Path
        Verified local path to the weights file.
    """
    if cfg is None:
        cfg = load_config()

    target_path = Path(custom_path) if custom_path else cfg.paths.model_path

    # Check if local model weights exist and are valid (non-empty > 1 KB)
    if target_path.exists() and target_path.is_file() and target_path.stat().st_size > 1024:
        logger.info("Found local model weights: %s (%.2f MB)", target_path, target_path.stat().st_size / (1024 * 1024))
        return target_path

    # Download from Hugging Face Hub
    repo_id = getattr(cfg.paths, "hf_repo_id", "HemangDulange/leukemia-all-classifier")
    filename = getattr(cfg.paths, "hf_filename", "dual_stream.keras")

    logger.warning("Local weights not found at %s. Triggering download from Hugging Face Hub...", target_path)
    return download_from_huggingface(repo_id, filename, target_path)


def load_trained_model(
    cfg: Optional[AppConfig] = None,
    custom_path: Optional[Path | str] = None,
) -> tf.keras.Model:
    """Resolve model path (local or remote) and load Keras model into memory.

    Parameters
    ----------
    cfg:
        Application configuration.
    custom_path:
        Optional explicit model path.

    Returns
    -------
    tf.keras.Model
        Loaded Keras dual-stream model.
    """
    configure_low_memory_runtime()
    model_path = resolve_model_path(cfg=cfg, custom_path=custom_path)
    logger.info("Loading Keras model from %s...", model_path)
    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
    except Exception:
        model = tf.keras.models.load_model(str(model_path))
    logger.info("Keras model loaded successfully.")
    return model

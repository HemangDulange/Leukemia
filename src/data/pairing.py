"""
Pair Original and Segmented blood-smear images by filename.

Walks data/raw/Original/<Class>/ and data/raw/Segmented/<Class>/,
matches images by filename/stem, and reports all unmatched files per class.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from src.config import AppConfig

logger = logging.getLogger(__name__)

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _normalize_stem(name: str) -> str:
    """Normalize stem for pairing (e.g. handle 'Seg-WBC-001' or 'WBC-001_seg')."""
    lower = name.lower()
    if lower.startswith("seg-") or lower.startswith("seg_"):
        return lower[4:]
    if lower.endswith("-seg") or lower.endswith("_seg"):
        return lower[:-4]
    return lower


def _collect_images(folder: Path) -> Dict[str, Path]:
    """Return mapping of normalized stem -> file path for images in folder."""
    if not folder.exists():
        logger.warning("Folder does not exist: %s", folder)
        return {}
    mapping = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES:
            key = _normalize_stem(p.stem)
            mapping[key] = p
    return mapping


def report_unmatched_files(
    orig_files: Dict[str, Path],
    seg_files: Dict[str, Path],
    class_name: str,
) -> Tuple[Set[str], Set[str]]:
    """Log and return unmatched files for a class.

    Parameters
    ----------
    orig_files:
        Dictionary mapping normalized stem -> Path in Original folder.
    seg_files:
        Dictionary mapping normalized stem -> Path in Segmented folder.
    class_name:
        Display or folder class name.

    Returns
    -------
    Tuple[Set[str], Set[str]]
        ``(unmatched_original, unmatched_segmented)`` file stems.
    """
    orig_keys = set(orig_files.keys())
    seg_keys = set(seg_files.keys())

    unmatched_orig = orig_keys - seg_keys
    unmatched_seg = seg_keys - orig_keys

    if unmatched_orig:
        logger.warning(
            "Class '%s': %d Original files have NO Segmented counterpart:\n  %s",
            class_name,
            len(unmatched_orig),
            ", ".join(sorted(list(unmatched_orig)[:10])) + (" ..." if len(unmatched_orig) > 10 else ""),
        )
    if unmatched_seg:
        logger.warning(
            "Class '%s': %d Segmented files have NO Original counterpart:\n  %s",
            class_name,
            len(unmatched_seg),
            ", ".join(sorted(list(unmatched_seg)[:10])) + (" ..." if len(unmatched_seg) > 10 else ""),
        )

    return unmatched_orig, unmatched_seg


def build_pairs(cfg: AppConfig) -> pd.DataFrame:
    """Walk Original and Segmented directories, match pairs, and report unmatched.

    Parameters
    ----------
    cfg:
        Loaded application configuration.

    Returns
    -------
    pd.DataFrame
        Columns: ``original_path``, ``segmented_path``, ``class_folder``,
        ``label_index``, ``filename``. Only matched pairs are included.
    """
    orig_root = cfg.data.raw_dir / cfg.data.original_subdir
    seg_root = cfg.data.raw_dir / cfg.data.segmented_subdir

    rows: List[dict] = []
    total_unmatched_orig = 0
    total_unmatched_seg = 0

    print("\n" + "=" * 60)
    print("IMAGE PAIRING & UNMATCHED FILE REPORT")
    print("=" * 60)

    for label_idx, class_name in enumerate(cfg.data.classes):
        orig_map = _collect_images(orig_root / class_name)
        seg_map = _collect_images(seg_root / class_name)

        unmatched_orig, unmatched_seg = report_unmatched_files(orig_map, seg_map, class_name)
        total_unmatched_orig += len(unmatched_orig)
        total_unmatched_seg += len(unmatched_seg)

        common_keys = sorted(set(orig_map.keys()) & set(seg_map.keys()))
        for key in common_keys:
            rows.append(
                {
                    "original_path": str(orig_map[key]),
                    "segmented_path": str(seg_map[key]),
                    "class_folder": class_name,
                    "label_index": label_idx,
                    "filename": orig_map[key].name,
                }
            )

        print(
            f"  Class {class_name:<10}: {len(common_keys):4d} matched pairs | "
            f"{len(unmatched_orig):3d} unmatched Orig | {len(unmatched_seg):3d} unmatched Seg"
        )

    print("-" * 60)
    print(f"Total matched pairs     : {len(rows)}")
    print(f"Total unmatched Original: {total_unmatched_orig}")
    print(f"Total unmatched Segmented: {total_unmatched_seg}")
    print("=" * 60 + "\n")

    df = pd.DataFrame(
        rows,
        columns=["original_path", "segmented_path", "class_folder", "label_index", "filename"],
    )
    return df

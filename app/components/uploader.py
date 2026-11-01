"""
Image upload component.

Renders a styled upload widget and returns the decoded RGB numpy array
plus the original filename.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image


def render_uploader() -> Optional[Tuple[np.ndarray, str]]:
    """Render the image uploader widget.

    Returns
    -------
    Optional[Tuple[np.ndarray, str]]
        ``(rgb_array, filename)`` if an image was uploaded, else ``None``.
    """
    st.markdown("### 🔬 Upload a Blood-Smear Image")
    uploaded = st.file_uploader(
        label="Drag & drop or browse",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload a Giemsa-stained blood-smear image for ALL classification.",
    )

    if uploaded is None:
        st.info("Upload an image to begin classification.")
        return None

    img_pil = Image.open(uploaded).convert("RGB")
    rgb = np.array(img_pil)
    return rgb, uploaded.name

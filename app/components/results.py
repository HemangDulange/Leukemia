"""
Results display component for the Streamlit application.

Renders:
- Top predicted diagnosis badge with clinical color coding
- Side-by-side visual pipeline: Original PBS, Binary Mask, HSV-Segmented, and Grad-CAM Overlay
- Dynamic probability distribution progress bars
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import streamlit as st

# Color palette keyed by display class name
_CLASS_COLOURS: Dict[str, str] = {
    "Hematogones": "#38A169",       # green  (benign)
    "Early Pre-B ALL": "#ECC94B",   # yellow
    "Pre-B ALL": "#ED8936",         # orange
    "Pro-B ALL": "#E53E3E",         # red
}
_DEFAULT_COLOUR = "#718096"


def render_results(result: Dict[str, Any], filename: str) -> None:
    """Render comprehensive classification results and explainability visuals.

    Parameters
    ----------
    result:
        Output dict from :meth:`src.inference.predictor.Predictor.predict`.
    filename:
        Original uploaded filename.
    """
    predicted = result.get("predicted_label", result.get("predicted_class", "Unknown"))
    confidence = result.get("confidence", 0.0)
    colour = _CLASS_COLOURS.get(predicted, _DEFAULT_COLOUR)

    st.markdown("---")
    st.markdown(f"## Diagnostic Report: `{filename}`")

    # ── Top prediction banner ─────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background: {colour}22;
            border-left: 6px solid {colour};
            border-radius: 6px;
            padding: 1.2rem 1.6rem;
            margin-bottom: 1.2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <div>
                <span style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; color: #A0AEC0;">
                    Predicted ALL Subtype
                </span>
                <div style="font-size: 1.6rem; color: {colour}; font-weight: 800; margin-top: 2px;">
                    {predicted}
                </div>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.85rem; color: #A0AEC0;">Confidence</span>
                <div style="font-size: 1.5rem; font-weight: 700; color: #FAFAFA;">
                    {confidence * 100:.1f}%
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 4-Panel Visual Inspection Pipeline ───────────────────────────────
    st.markdown("### 🔬 Multi-Stream Inspection & Visual Explainability")
    cols = st.columns(4)

    with cols[0]:
        st.image(result["original_image"], caption="1. Original Smear (Stream B)", use_column_width=True)

    with cols[1]:
        st.image(result["mask"], caption="2. HSV Binary Mask", use_column_width=True)

    with cols[2]:
        seg_img = result.get("segmented_image", result.get("masked_image"))
        st.image(seg_img, caption="3. Segmented (Stream A)", use_column_width=True)

    with cols[3]:
        gradcam_img = result.get("gradcam_overlay")
        if gradcam_img is not None:
            st.image(gradcam_img, caption="4. Grad-CAM (Conv Activation)", use_column_width=True)
        else:
            st.image(result["original_image"], caption="4. Grad-CAM Unavailable", use_column_width=True)

    # ── Class Probability Breakdown ───────────────────────────────────────
    st.markdown("### 📊 Subtype Probabilities")

    # Support either dict or list
    if "class_probabilities" in result and isinstance(result["class_probabilities"], dict):
        prob_items = list(result["class_probabilities"].items())
    else:
        prob_items = result.get("probabilities", [])

    for cls_name, prob in prob_items:
        c = _CLASS_COLOURS.get(cls_name, _DEFAULT_COLOUR)
        pct = float(prob) * 100
        is_top = (cls_name == predicted)
        font_weight = "bold" if is_top else "normal"

        st.markdown(
            f"""
            <div style="margin: 8px 0;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="width: 170px; font-size: 0.95rem; font-weight: {font_weight};">{cls_name}</span>
                    <div style="flex: 1; background: #2D3748; border-radius: 6px; height: 20px; overflow: hidden;">
                        <div style="width: {pct:.1f}%; background: {c}; height: 100%; border-radius: 6px; transition: width 0.5s ease-in-out;"></div>
                    </div>
                    <span style="width: 60px; text-align: right; font-size: 0.95rem; font-weight: {font_weight};">{pct:.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

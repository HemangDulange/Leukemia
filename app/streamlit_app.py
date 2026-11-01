"""
Leukemia ALL Classifier — Interactive Streamlit Application.

Layout & Features:
- Sidebar: About project, model architecture specifications, and medical disclaimer.
- Tab 1 ("ALL Classifier"):
  - Image uploader (jpg/png) and bundled sample images.
  - Image validation with error handling.
  - 4-Column visual inspection: Original | Mask | Segmented | Grad-CAM.
  - Predicted class badge with clinical color-coding and 4-class confidence bar chart.
- Tab 2 ("Pipeline"):
  - Visual diagram breaking down every pipeline step from the architecture diagram.
- Tab 3 ("Model Performance"):
  - Normalized confusion matrix, multi-class ROC curves, and ablation study from reports/figures/.
- Uses st.cache_resource for model loading.
"""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
from PIL import Image, UnidentifiedImageError
import streamlit as st

# ── Ensure repository root is on sys.path ─────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import load_config
from src.inference.predictor import Predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Leukemia ALL Classifier",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color Palette by ALL Subtype ──────────────────────────────────────────────
_CLASS_COLOURS = {
    "Hematogones": "#38A169",       # Green (benign control)
    "Early Pre-B ALL": "#ECC94B",   # Yellow (Early)
    "Pre-B ALL": "#ED8936",         # Orange (Pre-B)
    "Pro-B ALL": "#E53E3E",         # Red (Pro-B)
}
_DEFAULT_COLOUR = "#718096"


# ── Cached Model Loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading dual-stream model and backbones...")
def get_cached_predictor() -> Predictor:
    """Load configuration and model once; cached across sessions."""
    cfg = load_config()
    return Predictor(cfg)


# ── Sidebar: About, Model Info, and Medical Disclaimer ────────────────────────
def render_sidebar() -> None:
    st.sidebar.markdown("# 🩸 ALL Classifier")
    st.sidebar.markdown(
        "Dual-stream deep learning classification of **Acute Lymphoblastic Leukemia** "
        "(ALL) subtypes from Giemsa-stained peripheral blood smear (PBS) images."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Model Information")
    st.sidebar.markdown(
        """
- **Architecture**: Dual-Stream CNN
- **Backbone**: EfficientNetB0 (ImageNet weights)
  - *Stream A*: HSV-thresholded segmented smear
  - *Stream B*: Full RGB original smear
- **Fusion**: Feature Concatenation + BN + Dropout(0.4)
- **Input Resolution**: `224 × 224 × 3`
- **Classes**:
  - `Benign`: Hematogones
  - `Early`: Early Pre-B ALL
  - `Pre-B`: Pre-B ALL
  - `Pro`: Pro-B ALL
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚠️ Medical Disclaimer")
    st.sidebar.warning(
        "**Educational and Research Use Only.**\n\n"
        "This system is an academic research prototype. It is **not** a certified medical "
        "diagnostic device and must not be used as a substitute for professional clinical "
        "hematopathology review, flow cytometry, cytogenetics, or medical treatment decisions."
    )


# ── Tab 1: ALL Classifier (Main Inference Area) ───────────────────────────────
def render_classifier_tab(predictor: Optional[Predictor]) -> None:
    st.subheader("🔬 Peripheral Blood Smear Classification")
    st.caption("Upload a Giemsa-stained peripheral blood-smear image or select a bundled sample.")

    sample_dir = _REPO_ROOT / "app" / "sample_images"
    sample_options = ["None (Upload custom image)"]
    if sample_dir.exists():
        sample_files = [f.name for f in sample_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        sample_options.extend(sorted(sample_files))

    col_upload, col_sample = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Blood Smear Image (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a cropped single-cell or high-power field blood smear image.",
        )

    with col_sample:
        selected_sample = st.selectbox(
            "Or choose a bundled sample:",
            options=sample_options,
            index=1 if len(sample_options) > 1 else 0,
        )

    # Determine input image
    pil_image: Optional[Image.Image] = None
    image_name: str = "Image"

    if uploaded_file is not None:
        try:
            # Validate upload
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("Uploaded file exceeds 10 MB limit.")
                return
            pil_image = Image.open(uploaded_file).convert("RGB")
            image_name = uploaded_file.name
        except (UnidentifiedImageError, Exception) as e:
            st.error(f"Invalid image file: {e}. Please upload a valid JPG or PNG image.")
            return
    elif selected_sample != "None (Upload custom image)" and sample_dir.exists():
        sample_path = sample_dir / selected_sample
        if sample_path.exists():
            pil_image = Image.open(sample_path).convert("RGB")
            image_name = selected_sample

    if pil_image is None:
        st.info("👆 Upload an image or select a bundled sample above to begin classification.")
        return

    # Check predictor status
    if predictor is None:
        st.error(
            "Model weights not found. Please train the model (`python scripts/train.py`) "
            "and ensure `models/dual_stream.keras` exists."
        )
        return

    # Run inference
    with st.spinner("Executing dual-stream inference and Grad-CAM backpropagation..."):
        try:
            result = predictor.predict(pil_image, generate_gradcam=True)
        except Exception as e:
            st.error(f"Inference error: {e}")
            logger.exception("Inference failed")
            return

    predicted_label = result.get("predicted_label", result.get("predicted_class", "Unknown"))
    confidence = result.get("confidence", 0.0)
    badge_colour = _CLASS_COLOURS.get(predicted_label, _DEFAULT_COLOUR)

    # 1. Prediction Banner
    st.markdown("---")
    st.markdown(
        f"""
        <div style="
            background: {badge_colour}20;
            border-left: 6px solid {badge_colour};
            border-radius: 8px;
            padding: 1.2rem 1.6rem;
            margin-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <div>
                <span style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1.2px; color: #A0AEC0;">
                    Predicted ALL Diagnosis
                </span>
                <div style="font-size: 1.8rem; font-weight: 800; color: {badge_colour}; margin-top: 2px;">
                    {predicted_label}
                </div>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.85rem; color: #A0AEC0;">Model Confidence</span>
                <div style="font-size: 1.7rem; font-weight: 800; color: #FAFAFA;">
                    {confidence * 100:.1f}%
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Four-Column Visual Pipeline
    st.markdown("### 🔬 Multi-Stream Visual Inspection")
    cols = st.columns(4)

    with cols[0]:
        st.image(result["original_image"], caption="1. Original Smear (Stream B)", use_column_width=True)
    with cols[1]:
        st.image(result["mask"], caption="2. HSV Binary Mask", use_column_width=True)
    with cols[2]:
        st.image(result["segmented_image"], caption="3. Segmented (Stream A)", use_column_width=True)
    with cols[3]:
        gradcam_img = result.get("gradcam_overlay")
        if gradcam_img is not None:
            st.image(gradcam_img, caption="4. Grad-CAM Activation", use_column_width=True)
        else:
            st.image(result["original_image"], caption="4. Grad-CAM (N/A)", use_column_width=True)

    # 3. Confidence Bar Chart
    st.markdown("### 📊 Subtype Probability Distribution")
    prob_dict = result.get("class_probabilities", {})

    for cls_name, prob in prob_dict.items():
        c = _CLASS_COLOURS.get(cls_name, _DEFAULT_COLOUR)
        pct = float(prob) * 100
        is_top = (cls_name == predicted_label)
        font_weight = "bold" if is_top else "normal"

        st.markdown(
            f"""
            <div style="margin: 8px 0;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="width: 170px; font-size: 0.95rem; font-weight: {font_weight};">{cls_name}</span>
                    <div style="flex: 1; background: #2D3748; border-radius: 6px; height: 22px; overflow: hidden;">
                        <div style="width: {pct:.1f}%; background: {c}; height: 100%; border-radius: 6px;"></div>
                    </div>
                    <span style="width: 65px; text-align: right; font-size: 0.95rem; font-weight: {font_weight};">{pct:.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Tab 2: Pipeline Architecture ──────────────────────────────────────────────
def render_pipeline_tab() -> None:
    st.subheader("🧬 Dual-Stream Pipeline Architecture")
    st.markdown(
        """
        The architecture processes peripheral blood-smear images through two parallel pathways
        to capture both isolated intracellular structures and overall contextual morphology:
        """
    )

    st.markdown(
        """
```
                    ┌───────────────────────────────┐
                    │  Original PBS Image (H, W, 3) │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
       ┌─────────────────────────┐     ┌─────────────────────────┐
       │   Stream A: Segmented   │     │   Stream B: Original    │
       ├─────────────────────────┤     ├─────────────────────────┤
       │ 1. BGR ➔ HSV Transform  │     │ 1. Direct Smear Input   │
       │ 2. Dual-Range Threshold │     │ 2. EfficientNet Scaling │
       │    - Nuclei: Purple/Blue│     │    ([-1, 1])            │
       │    - Cyto: Pink/Purple  │     │ 3. Spatial Resizing     │
       │ 3. Morph Open & Close   │     │    (224 × 224)          │
       │ 4. Remove Small Blobs   │     │                         │
       │ 5. Impose Binary Mask   │     │                         │
       │    (Background ➔ Black) │     │                         │
       └────────────┬────────────┘     └────────────┬────────────┘
                    │                               │
                    ▼                               ▼
       ┌─────────────────────────┐     ┌─────────────────────────┐
       │ Backbone A (Seg stream) │     │ Backbone B (Orig stream)│
       │ (EfficientNetB0 + GAP)  │     │ (EfficientNetB0 + GAP)  │
       └────────────┬────────────┘     └────────────┬────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │   Feature Concatenation (2D)  │
                    ├───────────────────────────────┤
                    │   Batch Normalization         │
                    │   Dropout (p = 0.4)           │
                    │   Dense Softmax (4 classes)   │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ Hematogones | Early | Pre |Pro│
                    └───────────────────────────────┘
```
        """
    )

    st.markdown("---")
    st.markdown("### 🔬 Clinical Rationale for Dual Streams")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            **Stream A: Segmented Image (Intracellular Details)**
            - Isolates the lymphoblast nucleus, nucleoli, and cytoplasmic rim.
            - Background erythrocytes, platelets, and staining artifacts are suppressed to black.
            - Forces Backbone A to focus strictly on chromatin texture and nuclear contour.
            """
        )
    with col_b:
        st.markdown(
            """
            **Stream B: Original Smear (Cell Context & Morphology)**
            - Retains overall cell perimeter, nucleus-to-cytoplasm (N:C) ratio, and smear background.
            - Ensures subtle cytoplasmic projections and staining intensity nuances are preserved.
            - Fused feature representations outperform either stream individually.
            """
        )


# ── Tab 3: Model Performance ──────────────────────────────────────────────────
def render_performance_tab() -> None:
    st.subheader("📈 Model Performance & Validation Metrics")
    st.caption("Empirical performance results from test set evaluation and ablation experiments.")

    fig_dir = _REPO_ROOT / "reports" / "figures"
    metrics_dir = _REPO_ROOT / "reports" / "metrics"

    # Display evaluation CSV summary if available
    eval_csv = metrics_dir / "evaluation.csv"
    if eval_csv.exists():
        if pd is not None:
            df_metrics = pd.read_csv(eval_csv)
            st.markdown("#### 📋 Test Set Evaluation Metrics")
            st.dataframe(df_metrics, use_container_width=True)
        else:
            import csv
            with open(eval_csv, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            st.markdown("#### 📋 Test Set Evaluation Metrics")
            st.dataframe(rows, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📊 Evaluation Figures (300 DPI)")

    cols_figs = st.columns(2)

    cm_path = fig_dir / "confusion_matrix.png"
    roc_path = fig_dir / "roc_curves.png"
    history_path = fig_dir / "training_history.png"
    ablation_path = fig_dir / "ablation_comparison.png"

    with cols_figs[0]:
        st.markdown("##### Normalized Confusion Matrix")
        if cm_path.exists():
            st.image(str(cm_path), use_column_width=True)
        else:
            st.info("Run `python scripts/evaluate.py` to generate the confusion matrix.")

    with cols_figs[1]:
        st.markdown("##### Multi-Class One-vs-Rest ROC Curves")
        if roc_path.exists():
            st.image(str(roc_path), use_column_width=True)
        else:
            st.info("Run `python scripts/evaluate.py` to generate ROC curves.")

    st.markdown("---")
    cols_figs_2 = st.columns(2)

    with cols_figs_2[0]:
        st.markdown("##### Training History (Accuracy & Loss)")
        if history_path.exists():
            st.image(str(history_path), use_column_width=True)
        else:
            st.info("Run `python scripts/train.py` to generate training history curves.")

    with cols_figs_2[1]:
        st.markdown("##### Ablation Study (Dual-Stream Synergy)")
        if ablation_path.exists():
            st.image(str(ablation_path), use_column_width=True)
        else:
            st.info("Run `python scripts/evaluate.py --ablation` to generate the ablation comparison.")


# ── Main Application Controller ───────────────────────────────────────────────
def main() -> None:
    render_sidebar()

    # Attempt to load predictor
    predictor: Optional[Predictor] = None
    try:
        predictor = get_cached_predictor()
    except FileNotFoundError:
        predictor = None
    except Exception as exc:
        logger.warning("Could not load predictor: %s", exc)
        predictor = None

    # Main area header
    st.title("🩸 Acute Lymphoblastic Leukemia Classifier")
    st.markdown(
        "A dual-stream deep convolutional neural network for subtype classification of "
        "Giemsa-stained peripheral blood smear (PBS) images."
    )

    tab_classify, tab_pipeline, tab_performance = st.tabs(
        ["🔬 ALL Classifier", "🧬 Pipeline Architecture", "📈 Model Performance"]
    )

    with tab_classify:
        render_classifier_tab(predictor)

    with tab_pipeline:
        render_pipeline_tab()

    with tab_performance:
        render_performance_tab()

    # Footer Disclaimer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #718096; font-size: 0.85rem; padding: 1rem 0;">
            <b>Educational & Research Prototype</b> • Not intended for clinical diagnostic use •
            Leukemia ALL Classifier • Built with TensorFlow & Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

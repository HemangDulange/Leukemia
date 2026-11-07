"""
Leukemia Classifier — Interactive Streamlit Application.

Layout & Features:
- Sidebar: About project, model architecture specifications, and medical disclaimer.
- Tab 1 ("Leukemia Classifier"):
  - Image uploader (jpg/png/bmp/tiff) and optional bundled sample images.
  - Safe fallback if samples directory is empty or model weights are missing.
  - Multi-Column visual inspection: Original | Mask | Segmented | Grad-CAM.
  - Dynamic predicted class badge with subtype color-coding and probability distribution bar chart.
- Tab 2 ("Pipeline"):
  - Architecture flow diagram reflecting dual-stream inputs and [0, 255] rescaling.
  - Truthful clinical and algorithmic rationale.
- Tab 3 ("Model Performance"):
  - Normalized confusion matrix, multi-class ROC curves, and training history from reports/figures/.
- Uses st.cache_resource for predictor and model loading.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
from PIL import Image, UnidentifiedImageError
import streamlit as st

from src.config import load_config
from src.data.splits import load_rgb_image
from src.inference.predictor import Predictor
from src.preprocessing.pair import prepare_pair

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prevent PIL DecompressionBombError
Image.MAX_IMAGE_PIXELS = None

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Leukemia Multi-Class Classifier",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dynamic Palette Helper ────────────────────────────────────────────────────
_DEFAULT_PALETTE = [
    "#E53E3E",  # Red (e.g. AML)
    "#D69E2E",  # Amber/Yellow (e.g. Early Pre-B)
    "#38A169",  # Green (e.g. Normal)
    "#DD6B20",  # Orange (e.g. Pre-B)
    "#805AD5",  # Purple (e.g. Pro-B)
    "#3182CE",  # Blue
    "#319795",  # Teal
]
_DEFAULT_COLOUR = "#718096"


def load_canonical_metadata() -> Dict[str, Any]:
    """Load canonical class list and display names from class_names.json or config."""
    meta_path = _REPO_ROOT / "models" / "class_names.json"
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as e:
            logger.warning("Could not read %s: %s", meta_path, e)

    cfg = load_config()
    classes = cfg.data.classes
    disp = cfg.data.display_names or {c: c for c in classes}
    return {
        "class_order": classes,
        "display_names": {c: disp.get(c, c) for c in classes},
        "index_to_folder": {str(i): c for i, c in enumerate(classes)},
        "index_to_display": {str(i): disp.get(c, c) for i, c in enumerate(classes)},
    }


def get_class_colors(class_meta: Dict[str, Any]) -> Dict[str, str]:
    """Map display names and folder names to consistent UI colors."""
    colors = {}
    display_names = list(class_meta.get("display_names", {}).values())
    for idx, name in enumerate(display_names):
        colors[name] = _DEFAULT_PALETTE[idx % len(_DEFAULT_PALETTE)]
    for folder, disp in class_meta.get("display_names", {}).items():
        if disp in colors:
            colors[folder] = colors[disp]
    return colors


# ── Cached Model Loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading dual-stream model and backbones...")
def get_cached_predictor() -> Optional[Predictor]:
    """Load configuration and model once; cached across sessions. Returns None if weights missing."""
    try:
        cfg = load_config()
        if not cfg.paths.model_path.exists():
            return None
        return Predictor(cfg)
    except Exception as exc:
        logger.warning("Predictor initialization deferred: %s", exc)
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(class_meta: Dict[str, Any]) -> None:
    st.sidebar.markdown("# 🩸 Leukemia Classifier")
    st.sidebar.markdown(
        "Dual-stream deep learning classification of **Leukemia subtypes & Normal controls** "
        "from peripheral blood smear (PBS) microscopic images."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Model Information")

    classes_list = class_meta.get("class_order", [])
    disp_dict = class_meta.get("display_names", {})

    class_items_md = "\n".join([f"  - `{cls}`: {disp_dict.get(cls, cls)}" for cls in classes_list])

    st.sidebar.markdown(
        f"""
- **Architecture**: Dual-Stream CNN
- **Backbone**: EfficientNetB0 (ImageNet weights)
  - *Stream A*: HSV-thresholded segmented smear
  - *Stream B*: Full RGB original smear
- **Fusion**: Feature Concatenation + Dense Head
- **Input Resolution**: `224 × 224 × 3` (Rescaling [0, 255])
- **Classes ({len(classes_list)})**:
{class_items_md}
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


# ── Tab 1: Classifier (Main Inference Area) ───────────────────────────────────
def render_classifier_tab(
    predictor: Optional[Predictor],
    class_meta: Dict[str, Any],
    class_colors: Dict[str, str],
) -> None:
    st.subheader("🔬 Peripheral Blood Smear Classification")
    st.caption("Upload a peripheral blood smear microscopic image or select an available sample.")

    if predictor is None:
        st.warning(
            "⚠️ **Model weights file not found (`models/dual_stream.keras`).**\n\n"
            "You can still upload images to inspect the real-time HSV segmentation pipeline. "
            "To enable subtype classification and Grad-CAM activations, download or train "
            "the model using `python scripts/train.py`."
        )

    sample_dir = _REPO_ROOT / "app" / "samples"
    sample_options = ["None (Upload custom image)"]
    if sample_dir.exists():
        sample_files = [
            f.name for f in sample_dir.iterdir()
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        ]
        sample_options.extend(sorted(sample_files))

    col_upload, col_sample = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Blood Smear Image (JPG, PNG, BMP, TIFF)",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            help="Upload a single-cell or high-power field blood smear image.",
        )

    with col_sample:
        if len(sample_options) > 1:
            selected_sample = st.selectbox(
                "Or choose an available sample:",
                options=sample_options,
                index=1,
            )
        else:
            st.selectbox(
                "Sample images:",
                options=["No bundled samples found"],
                disabled=True,
            )
            selected_sample = "None (Upload custom image)"

    # Determine input image
    pil_image: Optional[Image.Image] = None
    image_name: str = "Image"

    if uploaded_file is not None:
        try:
            if uploaded_file.size > 25 * 1024 * 1024:
                st.error("Uploaded file exceeds 25 MB limit.")
                return
            pil_image = load_rgb_image(uploaded_file)
            image_name = uploaded_file.name

            # Check if image looks already segmented
            img_arr = np.array(pil_image)
            is_near_black = np.all(img_arr < 30, axis=-1)
            black_ratio = float(np.mean(is_near_black))
            if black_ratio > 0.70:
                st.warning("This looks like an already-segmented image. Please upload an original blood smear.")

        except (UnidentifiedImageError, Exception) as e:
            st.error(f"Invalid image file: {e}. Please upload a supported image.")
            return
    elif selected_sample != "None (Upload custom image)" and sample_dir.exists():
        sample_path = sample_dir / selected_sample
        if sample_path.exists():
            try:
                pil_image = load_rgb_image(sample_path)
                image_name = selected_sample
            except Exception as e:
                st.error(f"Failed to read sample image: {e}")
                return

    if pil_image is None:
        st.info("👆 Upload an image or select an available sample above to inspect.")
        return

    # If predictor is not loaded, run segmentation-only preview
    if predictor is None:
        cfg = load_config()
        orig_arr, seg_arr, mask_arr = prepare_pair(pil_image, cfg=cfg, target_size=224, return_mask=True)
        orig_uint8 = np.clip(orig_arr, 0, 255).astype(np.uint8)
        seg_uint8 = np.clip(seg_arr, 0, 255).astype(np.uint8)

        st.markdown("### 🔬 Preprocessing & Segmentation Preview")
        cols = st.columns(3)
        with cols[0]:
            st.image(Image.fromarray(orig_uint8), caption="1. Original Smear (224x224)", use_column_width=True)
        with cols[1]:
            st.image(Image.fromarray(mask_arr.astype(np.uint8)), caption="2. HSV Binary Mask", use_column_width=True)
        with cols[2]:
            st.image(Image.fromarray(seg_uint8), caption="3. Segmented Stream (Stream A)", use_column_width=True)
        return

    # Run full dual-stream inference
    with st.spinner("Executing dual-stream inference and Grad-CAM backpropagation..."):
        try:
            result = predictor.predict(pil_image, generate_gradcam=True)
        except Exception as e:
            st.error(f"Inference error: {e}")
            logger.exception("Inference failed")
            return

    predicted_label = result.get("predicted_label", result.get("predicted_class", "Unknown"))
    confidence = result.get("confidence", 0.0)
    badge_colour = class_colors.get(predicted_label, _DEFAULT_COLOUR)

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
                    Predicted Subtype / Diagnosis
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
        c = class_colors.get(cls_name, _DEFAULT_COLOUR)
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
        The architecture processes peripheral blood smear images through two parallel pathways
        to capture both isolated stained cellular regions and overall contextual morphology:
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
       │ 2. Dual-Range Threshold │     │ 2. Values in [0, 255]   │
       │    - Nuclei: Purple/Blue│     │    (Rescaling Layer     │
       │    - Cyto: Pink/Purple  │     │     inside EfficientNet)│
       │ 3. Morph Open & Close   │     │ 3. Spatial Resizing     │
       │ 4. Remove Small Blobs   │     │    (224 × 224)          │
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
                    │   Dense Softmax Head          │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  Multi-Class Softmax Output   │
                    │  (AML, Early Pre-B, Normal,   │
                    │   Pre-B ALL, Pro-B ALL)       │
                    └───────────────────────────────┘
```
        """
    )

    st.markdown("---")
    st.markdown("### 🔬 Clinical & Algorithmic Rationale")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            **Stream A: Segmented Image (Stained Region Focus)**
            - Applies dual-range HSV color thresholding targeted at purple/violet stained chromatin and cytoplasm.
            - Masks non-stained slide background to black, reducing background illumination variance.
            - Guides Backbone A to attend primarily to stained leukocyte morphology.
            """
        )
    with col_b:
        st.markdown(
            """
            **Stream B: Original Smear (Full Contextual Morphology)**
            - Preserves complete RGB color profile, unmasked cell boundaries, and whole-field context.
            - Retains natural staining gradients and overall cell-to-background contrast.
            - Provides complementary spatial and contextual features alongside the segmented stream.
            """
        )


# ── Tab 3: Model Performance ──────────────────────────────────────────────────
def render_performance_tab() -> None:
    st.subheader("📈 Model Performance & Validation Metrics")
    st.caption("Empirical performance results from test set evaluation and training experiments.")

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
    st.markdown("#### 📊 Evaluation Figures")

    cols_figs = st.columns(2)

    cm_path = fig_dir / "confusion_matrix.png"
    roc_path = fig_dir / "roc_curves.png"
    history_path = fig_dir / "training_history.png"
    ablation_path = fig_dir / "ablation_comparison.png"
    mask_preview_path = fig_dir / "mask_preview.png"

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
        st.markdown("##### 5-Class Segmentation Preview")
        if mask_preview_path.exists():
            st.image(str(mask_preview_path), use_column_width=True)
        elif ablation_path.exists():
            st.markdown("##### Ablation Study")
            st.image(str(ablation_path), use_column_width=True)
        else:
            st.info("Run `python scripts/preview_masks.py` to generate the 5-class mask preview grid.")


# ── Main Application Controller ───────────────────────────────────────────────
def main() -> None:
    class_meta = load_canonical_metadata()
    class_colors = get_class_colors(class_meta)

    render_sidebar(class_meta)

    # Attempt to load predictor
    predictor: Optional[Predictor] = None
    try:
        predictor = get_cached_predictor()
    except Exception as exc:
        logger.warning("Could not load predictor: %s", exc)
        predictor = None

    # Main area header
    st.title("🩸 Leukemia Multi-Class Classifier")
    st.markdown(
        "A dual-stream deep convolutional neural network for subtype classification of "
        "peripheral blood smear (PBS) microscopic images."
    )

    tab_classify, tab_pipeline, tab_performance = st.tabs(
        ["🔬 Leukemia Classifier", "🧬 Pipeline Architecture", "📈 Model Performance"]
    )

    with tab_classify:
        render_classifier_tab(predictor, class_meta, class_colors)

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
            Leukemia Classifier • Built with TensorFlow & Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

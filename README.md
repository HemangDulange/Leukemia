# 🩸 Leukemia ALL Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A dual-stream deep learning classifier for **Acute Lymphoblastic Leukemia (ALL)** blood-smear images, deployed with Streamlit.

The pipeline combines automated **HSV morphological color thresholding** (Stream A) to isolate intracellular Giemsa-stained structures with the **raw smear image** (Stream B) to capture contextual morphology, fusing both representations into a classification head for 4 distinct subtypes.

---

## 🧬 Architecture Overview

```
Original Smear Image (224 × 224 × 3)
         │
         ├─────────────────────────────────────────┐
         ▼                                         ▼
   Stream A (Segmented)                   Stream B (Original)
         │                                         │
    BGR ➔ HSV Transform                       Smear Input
         │                                         │
  Dual-Range Threshold                   EfficientNet Preprocessing
  - Nuclei: Purple/Blue                            │
  - Cyto:   Pink/Purple                       Backbone B
         │                              (EfficientNetB0 + GAP)
  Morph Open & Close                               │
         │                                   Feature Vec B (D,)
  Remove Small Blobs                               │
         │                                         │
  Impose Binary Mask                               │
  (Background ➔ Black)                             │
         │                                         │
    Backbone A                                     │
(EfficientNetB0 + GAP)                             │
         │                                         │
   Feature Vec A (D,)                              │
         │                                         │
         └───────────────────┬─────────────────────┘
                             ▼
                 Concatenate Fusion (2D,)
                             │
                    Batch Normalization
                             │
                      Dropout (p = 0.4)
                             │
                  Dense Softmax (4 Classes)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
      Hematogones (Benign)             Early Pre-B ALL
      Pre-B ALL                        Pro-B ALL
```

---

## 🎯 Target Classes

All models, scripts, and evaluation metrics enforce canonical class ordering via [`models/class_names.json`](models/class_names.json):

| Class Index | Raw Folder Name | Clinical Display Name | Description |
|:-----------:|:---------------:|:---------------------:|:------------|
| **0** | `Benign` | **Hematogones** | Normal B-cell precursors (non-leukemic control) |
| **1** | `Early` | **Early Pre-B ALL** | Early precursor B-cell Acute Lymphoblastic Leukemia |
| **2** | `Pre` | **Pre-B ALL** | Pre-B Acute Lymphoblastic Leukemia |
| **3** | `Pro` | **Pro-B ALL** | Pro-B Acute Lymphoblastic Leukemia |

---

## 🔬 Unified Preprocessing & Zero Train/Serve Skew

To prevent train/serve skew (where the offline model trains on third-party pre-segmented images while live serving generates fresh segmentation masks), the pipeline unifies all image processing in [`src/preprocessing/pair.py`](src/preprocessing/pair.py):

- **Single Pipeline Function (`prepare_pair`)**:
  Both training and real-time Streamlit inference execute the exact same sequence:
  1. Input is loaded via PIL `.convert("RGB")`.
  2. Full-resolution RGB image is passed to `segment_image()` (handling BGR/RGB color channels consistently and applying dual-range HSV color thresholding + morphological filtering).
  3. Both the original smear and segmented cell images are resized to $224 \times 224$ using area interpolation (`cv2.INTER_AREA`).
  4. Pixel values are kept as `float32` in the raw $[0, 255]$ range, matching Keras EfficientNet's internal Rescaling layer.
- **Offline Precomputation (`scripts/precompute_segmented.py`)**:
  Before training, lossless PNG pairs are generated in `data/processed/orig/<class>/` and `data/processed/seg/<class>/`. The dataset loader (`src/data/dataset.py`) reads directly from this cache, never touching raw pre-made `Segmented/` folders.
- **Serving Skew Verification (`scripts/check_serving_skew.py`)**:
  An automated CI/CD check evaluates the test set image bytes through `Predictor` and asserts test accuracy remains within $\le 1.5\%$ of batch evaluation.
- **Pre-segmented Upload Protection**:
  The Streamlit app checks uploaded smears: if $>70\%$ of pixels are near-black, it warns the user: *"This looks like an already-segmented image. Please upload the original blood smear."*

---

## 🚀 Quick Start & Local Setup

### 1. Prerequisites
- **Python 3.10** (recommended)
- Git

### 2. Clone Repository
```bash
git clone https://github.com/HemangDulange/Leukemia.git
cd Leukemia
```

### 3. Install Dependencies
```bash
# Lightweight dependencies for deployment and local inference:
pip install -r requirements.txt

# Or full dependencies for training, data preparation, and evaluation:
pip install -r requirements-train.txt
```

---

## 💻 Running the Streamlit App Locally

To launch the interactive diagnostic interface with upload validation and Grad-CAM explainability:

```bash
streamlit run app/streamlit_app.py
```

The application opens at `http://localhost:8501`.
- **Bundled Samples**: Test immediately with preloaded peripheral blood-smear samples in `app/samples/`.
- **Visual Explainability**: Side-by-side inspection across Original Smear, HSV Mask, Segmented Cell, and Grad-CAM Activation Heatmap.
- **Model Loading**: Automatically searches for local weights at `models/dual_stream.keras`; if absent, safely downloads published weights from Hugging Face Hub.

---

## ☁️ Deployment on Streamlit Community Cloud

This project is tailored for **Streamlit Community Cloud** (under the 1 GB RAM container limit):
- Pinned `tensorflow-cpu` and `opencv-python-headless` (zero GPU/display overhead).
- Peak inference memory is ~450 MB (well below the 1024 MB resource limit).
- Model weights exceeding GitHub's 100 MB limit are resolved automatically via [`src/inference/model_loader.py`](src/inference/model_loader.py).

### Deployment Steps:
1. Push your repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New App"** and select:
   - **Repository**: `HemangDulange/Leukemia`
   - **Branch**: `main`
   - **Main file path**: `app/streamlit_app.py`
4. Under **Advanced Settings**, confirm Python **3.10**.
5. Click **Deploy!**

---

## ⚡ How to Train on Kaggle (Free GPU)

Full training requires GPU compute and should not be run on local laptops without a dedicated GPU.

### Method A: Using the Provided Kaggle Notebook
1. Open the included notebook: [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb).
2. Upload it to [Kaggle Notebooks](https://www.kaggle.com/code).
3. Set **Accelerator** to **GPU T4 x2** or **P100**.
4. In the right panel, click **Add Input** and search for:
   `mehradaria/leukemia` (or `Acute Lymphoblastic Leukemia (ALL) image dataset`).
5. Run all cells. The notebook will:
   - Clone this repository
   - Link `data/raw/Original` and `data/raw/Segmented`
   - Generate stratified splits (70% train, 15% val, 15% test, seed 42)
   - Execute Phase 1 head training (`lr = 1e-3`) and Phase 2 fine-tuning (`lr = 1e-5`)
   - Output `dual_stream.keras`, metrics CSV, and publication plots.

### Method B: Command-Line Training
```bash
# 1. Generate stratified reproducible splits
python scripts/prepare_data.py --force --samples 50

# 2. Local smoke-test (1 epoch, 128px, 64 samples — safe for any laptop):
python scripts/train.py --smoke-test

# 3. Full GPU training (Kaggle / Colab):
python scripts/train.py
```

---

## 📊 Evaluation & Ablation Study

Evaluate the trained weights on the frozen test split:

```bash
# Standard evaluation (Accuracy, Weighted Precision/Recall/F1, and OvR AUC)
python scripts/evaluate.py

# Run ablation study comparing single-stream vs. dual-stream synergy
python scripts/evaluate.py --ablation
```

Outputs generated in `reports/`:
- `reports/metrics/evaluation.csv`: Full metric score summary.
- `reports/metrics/ablation_study.csv`: Empirical performance across Segmented-only, Original-only, and Dual-stream.
- `reports/figures/confusion_matrix.png`: Normalized confusion matrix with cell counts (300 DPI).
- `reports/figures/roc_curves.png`: One-vs-Rest ROC curves per subtype (300 DPI).
- `reports/figures/ablation_comparison.png`: Ablation bar chart (300 DPI).

---

## 📂 Project Structure

```
leukemia-all-classifier/
├── .streamlit/
│   └── config.toml             # Dark theme & upload security settings
├── configs/
│   └── config.yaml             # Single source of truth (paths, thresholds, hyperparameters)
├── data/                       # Dataset directory (gitignored)
│   ├── raw/Original            # Raw blood smear images
│   ├── processed/              # Lossless precomputed PNGs (zero train/serve skew)
│   │   ├── orig/<class>/
│   │   └── seg/<class>/
│   └── splits/{train, val, test}.csv
├── notebooks/
│   └── kaggle_train.ipynb      # End-to-end Kaggle GPU training notebook
├── src/                        # Core modular library
│   ├── config.py               # Typed AppConfig loader
│   ├── data/
│   │   ├── pairing.py          # Matches Original pairs & generates labels
│   │   ├── splits.py           # Stratified split generator & class-weight computation
│   │   └── dataset.py          # Dual-input tf.data pipeline reading processed cache
│   ├── preprocessing/
│   │   ├── pair.py             # Single unified prepare_pair function (train & serve)
│   │   ├── segmentation.py     # BGR->HSV thresholding & morphological filtering
│   │   ├── normalization.py    # EfficientNet scaling & Reinhard stain transfer
│   │   └── augmentation.py     # Joint spatial Keras augmentation layers (train-only)
│   ├── models/
│   │   ├── backbones.py        # Backbone factory (EfficientNetB0-B5, GAP)
│   │   └── dual_stream.py      # Dual-stream functional model & unfreezing utilities
│   ├── training/
│   │   ├── callbacks.py        # EarlyStopping, ModelCheckpoint (val_accuracy)
│   │   └── train.py            # Two-phase training loop (Head -> Top 30 layers)
│   ├── evaluation/
│   │   ├── metrics.py          # Multiclass OvR AUC, classification report, CSV export
│   │   ├── plots.py            # 300 DPI confusion matrices, ROC, and ablation plots
│   │   └── gradcam.py          # Grad-CAM heatmap overlay from segmented backbone
│   └── inference/
│       ├── model_loader.py     # Local model loader with Hugging Face Hub download fallback
│       └── predictor.py        # Production inference engine used by Streamlit
├── scripts/                    # CLI commands
│   ├── prepare_data.py         # Generate stratified CSV splits
│   ├── precompute_segmented.py # Precompute unified PNG pairs
│   ├── train.py                # Train model (calls precompute automatically if needed)
│   ├── evaluate.py             # Evaluate test split performance & metrics
│   ├── check_serving_skew.py   # Verify live Predictor vs offline pipeline parity (<=1.5%)
│   ├── make_samples.py         # Pick 1 sample per class & bundle samples.zip
│   └── predict.py              # CLI batch inference
├── app/
│   ├── streamlit_app.py        # Interactive Streamlit application
│   ├── samples/                # Bundled PBS smear test samples for live app
│   └── components/             # Modular UI components
├── models/
│   ├── class_names.json        # Canonical class ordering & display names
│   └── dual_stream.keras       # Model weights
├── reports/
│   ├── figures/                # High-res 300 DPI diagnostic figures
│   └── metrics/                # evaluation.csv, history.json, ablation_study.csv
├── tests/                      # Automated unit & integration tests
├── requirements.txt            # Lightweight deployment dependencies
├── requirements-train.txt      # Full GPU training dependencies
└── runtime.txt                 # Python 3.10 runtime declaration for Streamlit Cloud
```

---

## ⚠️ Medical Disclaimer

**For Educational, Academic, and Scientific Research Use Only.**

This software is an artificial intelligence research demonstration and is **not** a certified medical diagnostic device. It is not intended to diagnose, treat, cure, or prevent any disease. Clinical decisions regarding leukemia diagnosis must only be made by qualified medical specialists based on bone marrow aspirates, comprehensive flow cytometry immunophenotyping, and cytogenetic analysis.

---


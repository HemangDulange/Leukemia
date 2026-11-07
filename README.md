# 🩸 Leukemia Multi-Class Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A dual-stream deep learning classifier for multi-class **Leukemia Subtypes & Normal Controls** from peripheral blood smear (PBS) microscopic images, deployed with Streamlit.

The pipeline combines automated **HSV morphological color thresholding** (Stream A) targeting Giemsa-stained leukocyte chromatin and cytoplasm with the **unsegmented smear image** (Stream B) to capture contextual morphology, fusing both representations into a classification head for 5 distinct classes.

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
  Dual-Range Threshold                    Pixel range [0, 255]
  - Nuclei: Purple/Blue                   (Rescaling inside backbone)
  - Cyto:   Pink/Purple                            │
         │                                    Backbone B
  Morph Open & Close                     (EfficientNetB0 + GAP)
         │                                         │
  Remove Small Blobs                              │
         │                                Feature Vec B (D,)
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
                  Dense Softmax (5 Classes)
                             │
        ┌─────────────┬──────┴──────┬─────────────┐
        ▼             ▼             ▼             ▼
       AML       Early Pre-B     Normal        Pre-B ALL / Pro-B ALL
```

---

## 🎯 Target Classes & Dataset

The classifier operates on the **Multi-Class Leukemia Dataset (ALL Subtypes, AML, Normal)**:
- **Kaggle Path**: `/kaggle/input/datasets/shimantokumar/multi-class-leukemia-dataset-all-subtypesamlnormal/leukemia_datasets`
- **Total Images**: 9,097 images across 5 flat class folders:
  - `AML`: 2,000 images
  - `Early_PreB`: 1,738 images
  - `Normal`: 2,000 images
  - `PreB`: 1,800 images
  - `ProB`: 1,559 images
- **Mixed Formats & Resolutions**: `.jpg`, `.bmp`, `.tiff`, `.png` ranging from $400 \times 400$ up to $2560 \times 1920$.

Canonical class mapping is configured via `configs/config.yaml` and persisted to `models/class_names.json`:

| Class Index | Raw Folder Name | Clinical Display Name | Description | Count |
|:-----------:|:---------------:|:---------------------:|:------------|:-----:|
| **0** | `AML` | **AML (Myeloid)** | Acute Myeloid Leukemia blast cells | 2,000 |
| **1** | `Early_PreB` | **Early Pre-B ALL** | Early precursor B-cell Acute Lymphoblastic Leukemia | 1,738 |
| **2** | `Normal` | **Normal** | Healthy non-leukemic control leukocytes | 2,000 |
| **3** | `PreB` | **Pre-B ALL** | Pre-B Acute Lymphoblastic Leukemia | 1,800 |
| **4** | `ProB` | **Pro-B ALL** | Pro-B Acute Lymphoblastic Leukemia | 1,559 |

---

## 🔍 Near-Duplicate-Aware Perceptual Hash Splits

To guarantee absolute split independence and prevent data leakage:
1. **Canonical Perceptual Hashing**:
   - Each image is downsampled: $64 \times 64$ RGB $\rightarrow 8 \times 8$ grayscale $\rightarrow$ mean-thresholded 64-bit boolean string.
   - Evaluated across all 8 D4 dihedral symmetries (4 rotations $\times$ horizontal flip), taking the lexicographical minimum bit string.
   - Rotated, mirrored, or identical image variants receive the exact same canonical hash and `group_id`.
2. **Stratified Group Splitting (`StratifiedGroupKFold`)**:
   - 7 folds are evaluated with fixed seed 42 (Fold 0 = Test ~15%, Fold 1 = Validation ~15%, Folds 2–6 = Train ~70%).
   - All near-duplicates belonging to the same group land strictly in the same split.

---

## 🔬 Unified Preprocessing & Zero Train/Serve Skew

All stages of the project enforce one shared preprocessing pipeline via `src/preprocessing/pair.py`:

- **Single Function (`prepare_pair`)**:
  1. Accepts PIL image or uint8 array (safe 8-bit RGB loader supports 16-bit TIFFs, palette, and alpha channels).
  2. Full-resolution HSV morphological segmentation generates clean cell mask.
  3. Original and segmented images resized with `cv2.INTER_AREA` to $224 \times 224$.
  4. Returns `float32` arrays in raw $[0, 255]$ range, matching Keras EfficientNet's internal Rescaling layer.
- **Precomputed Cache (`scripts/precompute_segmented.py`)**:
  - Precomputes and caches lossless PNG pairs to `data/processed/orig/<class>/` and `data/processed/seg/<class>/`.
  - Automatically invoked before training if cache is absent.
- **Mask Validation Grid (`scripts/preview_masks.py`)**:
  - Samples 6 random images per class (seed 42) and renders a 30-image grid (`Original | Mask | Segmented`) to `reports/figures/mask_preview.png`.

---

## 🚀 Quick Start & Local Setup

### 1. Prerequisites
- **Python 3.10** (recommended)
- Git

### 2. Clone Repository
```bash
git clone -b new-dataset https://github.com/HemangDulange/Leukemia.git
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

```bash
streamlit run app/streamlit_app.py
```

The application opens at `http://localhost:8501`:
- **Dynamic 5-Class Interface**: Automatically populates classes, color badges, and probability distributions from `models/class_names.json`.
- **Visual Inspection**: Displays Original Smear, HSV Binary Mask, Segmented Stream, and Grad-CAM Activation Heatmap.
- **Graceful Fallbacks**: Works smoothly even if weights are missing (provides real-time segmentation preview) or if sample folders are empty.

---

## ⚡ How to Train on Kaggle (Free GPU)

Full training requires GPU compute:

1. In Kaggle Notebooks, select **Accelerator**: **GPU T4 x2** or **P100**.
2. Mount the dataset:
   `/kaggle/input/datasets/shimantokumar/multi-class-leukemia-dataset-all-subtypesamlnormal/leukemia_datasets`
3. Execute the CLI pipeline:
   ```bash
   # 1. Discover images and generate duplicate-aware stratified splits:
   python scripts/prepare_data.py --force

   # 2. Visual verification grid of segmentation masks:
   python scripts/preview_masks.py

   # 3. Two-phase training:
   python scripts/train.py

   # 4. Evaluation and publication metrics:
   python scripts/evaluate.py --ablation

   # 5. Extract representative sample images closest to median confidence:
   python scripts/make_samples.py
   ```

---

## 📂 Project Structure

```
leukemia-all-classifier/
├── configs/
│   └── config.yaml             # Single source of truth (classes, display names, hyperparameters)
├── data/
│   ├── processed/              # Lossless precomputed PNGs (orig/ and seg/)
│   └── splits/                 # train.csv, val.csv, test.csv
├── models/
│   ├── class_names.json        # Canonical 5-class ordering & display names
│   └── dual_stream.keras       # Trained dual-stream model weights
├── src/
│   ├── config.py               # Typed AppConfig loader with dynamic num_classes
│   ├── data/
│   │   ├── splits.py           # Perceptual hash near-duplicate grouping & StratifiedGroupKFold
│   │   └── dataset.py          # Dual-input tf.data pipeline reading processed cache
│   ├── preprocessing/
│   │   ├── pair.py             # Single unified prepare_pair function (train & serve)
│   │   ├── segmentation.py     # BGR->HSV thresholding & morphological filtering
│   │   └── augmentation.py     # Joint spatial Keras augmentation layers (train-only)
│   ├── models/
│   │   ├── backbones.py        # EfficientNet backbone factory
│   │   └── dual_stream.py      # Dual-stream functional model & Phase 2 unfreezing
│   ├── training/
│   │   ├── callbacks.py        # EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
│   │   └── train.py            # Two-phase training loop (Phase 1 Head -> Phase 2 Fine-tuning)
│   ├── evaluation/
│   │   ├── metrics.py          # Multiclass OvR AUC, classification report, CSV export
│   │   ├── plots.py            # 300 DPI confusion matrices, ROC, and history plots
│   │   └── gradcam.py          # Grad-CAM heatmap overlay from segmented backbone
│   └── inference/
│       ├── model_loader.py     # Local model loader with Hugging Face Hub download fallback
│       └── predictor.py        # Production inference engine used by Streamlit
├── scripts/
│   ├── prepare_data.py         # Discover images and generate reproducible splits
│   ├── precompute_segmented.py # Precompute unified PNG pairs
│   ├── preview_masks.py        # 30-image mask preview grid across all 5 classes
│   ├── train.py                # Train model (calls precompute automatically)
│   ├── evaluate.py             # Evaluate test split performance & metrics
│   ├── check_serving_skew.py   # Verify live Predictor vs offline pipeline parity
│   └── make_samples.py         # Pick median-confidence samples per class & bundle zip
├── app/
│   └── streamlit_app.py        # Interactive Streamlit application
├── reports/
│   ├── figures/                # High-res diagnostic figures (mask_preview.png, confusion_matrix.png)
│   └── metrics/                # evaluation.csv, history.json, ablation_study.csv
├── tests/
│   ├── make_fake_dataset.py    # Synthetic 5-class dataset generator for smoke testing
│   └── test_data.py            # Unit tests for perceptual hash and data pipelines
└── requirements.txt            # Streamlit Cloud deployment requirements
```

---

## ⚠️ Medical Disclaimer

**For Educational, Academic, and Scientific Research Use Only.**

This software is an artificial intelligence research demonstration and is **not** a certified medical diagnostic device. It is not intended to diagnose, treat, cure, or prevent any disease. Clinical decisions regarding leukemia diagnosis must only be made by qualified medical specialists based on bone marrow aspirates, comprehensive flow cytometry immunophenotyping, and cytogenetic analysis.

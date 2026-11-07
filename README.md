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

## 🎯 Target Classes & Dataset Audit Findings

The classifier operates on the **Multi-Class Leukemia Dataset (ALL Subtypes, AML, Normal)**:
- **Kaggle Path**: `/kaggle/input/datasets/shimantokumar/multi-class-leukemia-dataset-all-subtypesamlnormal/leukemia_datasets`
- **Total Images**: 9,097 images across 5 flat class folders:
  - `AML`: 2,000 images
  - `Early_PreB`: 1,738 images
  - `Normal`: 2,000 images
  - `PreB`: 1,800 images
  - `ProB`: 1,559 images

### Critical Dataset Audit Findings
1. **Pervasive Near-Duplicates**: ~3,100 of the 9,097 images are duplicates (the same photo exists twice, once at $224 \times 224$ and once at $1024 \times 768$). Only ~6,000 distinct biological photos exist.
2. **PreB Domain Contamination**: 16 `.bmp` files inside the `PreB` folder are single dark cells on black backgrounds from a different source than the other PreB smear images.
3. **Format & Illumination Shortcuts**: AML consists of 2,000 close-up TIFFs ($400 \times 400$); ~1,000 Normal images are black-background single-cell BMPs ($450 \times 450$); all other images are wide-field smear JPGs.
4. **Stain Color Shortcut**: A Random Forest trained on **only mean R, G, B and black-pixel share** reaches **97% test accuracy**, proving that models without color robustness rely heavily on stain coloration rather than true cellular morphology.

Canonical class mapping is configured via `configs/config.yaml` and persisted to `models/class_names.json`:

| Class Index | Raw Folder Name | Clinical Display Name | Description | Count |
|:-----------:|:---------------:|:---------------------:|:------------|:-----:|
| **0** | `AML` | **AML (Myeloid)** | Acute Myeloid Leukemia blast cells | 2,000 |
| **1** | `Early_PreB` | **Early Pre-B ALL** | Early precursor B-cell Acute Lymphoblastic Leukemia | 1,738 |
| **2** | `Normal` | **Normal** | Healthy non-leukemic control leukocytes | 2,000 |
| **3** | `PreB` | **Pre-B ALL** | Pre-B Acute Lymphoblastic Leukemia | 1,800 |
| **4** | `ProB` | **Pro-B ALL** | Pro-B Acute Lymphoblastic Leukemia | 1,559 |

---

## 🔍 Near-Duplicate-Aware Pearson Correlation Splits

To guarantee absolute split independence, eliminate duplicate contamination, and prevent data leakage:
1. **Extension-Level Filtering (`data.exclude_class_extensions`)**:
   - Skips files matching specified extensions per class (e.g., `{PreB: [".bmp"]}`), eliminating the 16 non-smear black-background BMPs from PreB.
2. **Pearson Correlation Near-Duplicate Detection**:
   - Generates $64 \times 64$ RGB thumbnails with PIL draft mode for JPEGs using an 8-worker `ThreadPoolExecutor`, simultaneously recording original dimensions `(w, h)`.
   - Caches thumbnails and sizes to `data/processed/thumbs64.npz` for instant reloading.
   - Converted to $32 \times 32$ grayscale (block averaging), zero-mean centered, and L2-normalized.
   - Evaluated across all 8 D4 dihedral symmetries (4 rotations $\times$ horizontal flip) using chunked matrix multiplication ($500 \times 500$ blocks). Pairs with correlation $\ge \text{dedup\_threshold}$ (default `0.995`) form near-duplicate edges clustered with `scipy.sparse.csgraph.connected_components`.
3. **Safety Clustering Guards**:
   - **Cross-Class Duplicate Drop**: If a group contains images from more than one class, all of its members are dropped and logged to eliminate ambiguous data.
   - **Cluster Size Safety**: Groups with $> 25$ images are treated as independent singletons with a logged warning, preventing false clustering on uniform dark backgrounds.
4. **Resolution-Aware Deduplication (`data.drop_duplicates: true`)**:
   - Keeps exactly one image per group: the one with the largest original resolution (`width * height`). Ties are deterministically broken alphabetically by path.
   - Dropped duplicate records are exported to `data/splits/dropped_duplicates.csv`.
5. **Stratified Group Splitting (`StratifiedGroupKFold`)**:
   - 7 folds are evaluated with fixed seed 42 (Fold 0 = Test ~15%, Fold 1 = Validation ~15%, Folds 2–6 = Train ~70%).
   - All near-duplicates belonging to the same group land strictly in the same split.
6. **Dedicated CLI Script (`scripts/prepare_splits.py`)**:
   - Run `python scripts/prepare_splits.py [--force]` to discover images, deduplicate, and generate reproducible `train.csv`, `val.csv`, `test.csv`, and `dropped_duplicates.csv`. (Reused directly by `train.py`).

---

## 🎨 Color Robustness & Disentanglement

Because peripheral blood smear stain colors can vary drastically between laboratories and slide preparation protocols (or correlate artificially with diagnostic subtypes):
1. **Multiplicative Color Jitter (Train-Only)**:
   - Applied to both streams after segmentation:
     - Hue shift: $\pm 0.05$
     - Saturation factor: $\times 0.6 - 1.4$
     - Per-channel gain: $0.85 - 1.15$
   - Strictly multiplicative transformations preserve zero-valued black pixels $(0, 0, 0)$ on the segmented stream without additive bleed.
2. **Random Grayscale Augmentation**:
   - Converts both streams to grayscale with probability $p = 0.30$ and repeats the luminance channel across 3 channels, forcing the network to learn cell morphology and texture rather than relying on stain hue.
3. **Color-Only Baseline Classifier (`scripts/colour_baseline.py`)**:
   - Random Forest (200 trees, seed 42) trained on the TRAIN split and tested on the TEST split using **only 4 color/illumination features**:
     1. Mean Red ($R$) intensity
     2. Mean Green ($G$) intensity
     3. Mean Blue ($B$) intensity
     4. Black-pixel share ($\max(R, G, B) < 25$)
   - Exports feature importances and confusion matrix to `reports/metrics/colour_baseline.csv` to benchmark against deep morphological representations.
4. **Grayscale Robustness Evaluation (`scripts/evaluate.py --gray-test`)**:
   - Runs model evaluation on the test split with both streams forced to grayscale, printing a side-by-side comparison table of overall accuracy and per-class recall against normal RGB images.

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
   # 1. Discover images, deduplicate near-duplicates, and generate stratified splits:
   python scripts/prepare_splits.py --force

   # 2. Visual verification grid of segmentation masks:
   python scripts/preview_masks.py

   # 3. Two-phase training (Phase 1 head -> Phase 2 fine-tuning):
   python scripts/train.py

   # 4. Evaluation (normal test, grayscale robustness, smear vs non-smear subgroups, and ablation):
   python scripts/evaluate.py --gray-test --subgroups --ablation

   # 5. Color-only baseline evaluation:
   python scripts/colour_baseline.py

   # 6. Extract representative sample images closest to median confidence:
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
│   └── splits/                 # train.csv, val.csv, test.csv, dropped_duplicates.csv
├── models/
│   ├── class_names.json        # Canonical 5-class ordering & display names
│   └── dual_stream.keras       # Trained dual-stream model weights
├── src/
│   ├── config.py               # Typed AppConfig loader with dynamic num_classes
│   ├── data/
│   │   ├── splits.py           # Pearson correlation near-duplicate grouping & StratifiedGroupKFold
│   │   └── dataset.py          # Dual-input tf.data pipeline reading processed cache
│   ├── preprocessing/
│   │   ├── pair.py             # Single unified prepare_pair function (train & serve)
│   │   ├── segmentation.py     # BGR->HSV thresholding & morphological filtering
│   │   └── augmentation.py     # Joint spatial & color robustness augmentation (train-only)
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
│   ├── prepare_splits.py       # Discover images, deduplicate, and generate reproducible splits
│   ├── prepare_data.py         # Backward-compatible wrapper for prepare_splits
│   ├── precompute_segmented.py # Precompute unified PNG pairs
│   ├── preview_masks.py        # 30-image mask preview grid across all 5 classes
│   ├── colour_baseline.py      # Random Forest color-only baseline (mean RGB + black share)
│   ├── train.py                # Train model (calls precompute automatically)
│   ├── evaluate.py             # Evaluate test split performance (--gray-test, --subgroups, --ablation)
│   ├── check_serving_skew.py   # Verify live Predictor vs offline pipeline parity
│   └── make_samples.py         # Pick median-confidence samples per class & bundle zip
├── app/
│   └── streamlit_app.py        # Interactive Streamlit application
├── reports/
│   ├── figures/                # High-res diagnostic figures (mask_preview.png, confusion_matrix.png)
│   └── metrics/                # evaluation.csv, history.json, subgroups.csv, colour_baseline.csv
├── tests/
│   ├── make_fake_dataset.py    # Synthetic 5-class dataset generator for smoke testing
│   └── test_data.py            # Unit tests for perceptual hash and data pipelines
└── requirements.txt            # Streamlit Cloud deployment requirements
```

---

## ⚠️ Medical Disclaimer

**For Educational, Academic, and Scientific Research Use Only.**

This software is an artificial intelligence research demonstration and is **not** a certified medical diagnostic device. It is not intended to diagnose, treat, cure, or prevent any disease. Clinical decisions regarding leukemia diagnosis must only be made by qualified medical specialists based on bone marrow aspirates, comprehensive flow cytometry immunophenotyping, and cytogenetic analysis.

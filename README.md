# Leukemia ALL Classifier

A dual-stream deep learning classifier for **Acute Lymphoblastic Leukemia (ALL)**
blood-smear images, deployed with Streamlit.

## Architecture

```
Stream A (segmented)              Stream B (original)
       │                                  │
Original Image                    Original Image
       │                                  │
  HSV Colour                        Pre-process
  Thresholding                           │
       │                            Backbone B
  Binary Mask                            │
       │                           Feature Vec B
  Masked Image                           │
       │                                 │
  Pre-process                            │
       │                                 │
  Backbone A                             │
       │                                 │
 Feature Vec A ──── Concat ──── Feature Vec B
                        │
                Classification Head
                        │
                    4 Classes
```

## Classes

| Folder | Display Name     |
|--------|-----------------|
| Benign | Hematogones      |
| Early  | Early Pre-B ALL  |
| Pre-B  | Pre-B ALL        |
| Pro    | Pro-B ALL        |

## Quick Start

### Install (inference / deploy)
```bash
pip install -r requirements.txt
```

### Install (training)
```bash
pip install -r requirements-train.txt
```

### Prepare data
```bash
python scripts/prepare_data.py
```

### Smoke test (local, CPU-safe)
```bash
python scripts/train.py --smoke-test
```

### Full training (Kaggle / Colab only)
```bash
python scripts/train.py
```

### Run app
```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```
leukemia-all-classifier/
├── configs/config.yaml        # Single source of truth for all settings
├── src/                       # Library code (importable, tested)
│   ├── config.py
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── inference/
├── scripts/                   # CLI entry points
├── app/                       # Streamlit deployment
├── models/                    # Saved weights + class_names.json
├── reports/                   # Figures and metrics
└── tests/
```

## Configuration

All settings live in `configs/config.yaml`. Override any value by passing
`--config path/to/custom.yaml` to any script.

## Requirements

- Python 3.10
- TensorFlow/Keras (CPU build for local; GPU for Kaggle/Colab)
- 8 GB RAM minimum for smoke-test; 16 GB+ recommended for full training

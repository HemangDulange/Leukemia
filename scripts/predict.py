"""
CLI script: run single-image prediction.

Usage
-----
    python scripts/predict.py --image PATH/TO/IMAGE.jpg [--config PATH]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.inference.predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict class for a single blood-smear image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    predictor = Predictor(cfg)

    result = predictor.predict(args.image)
    print(f"\nPredicted class : {result['predicted_class']}")
    print(f"Label index     : {result['label_index']}")
    print("\nClass probabilities:")
    for name, prob in result["probabilities"]:
        bar = "█" * int(prob * 30)
        print(f"  {name:<20} {prob:.4f}  {bar}")


if __name__ == "__main__":
    main()

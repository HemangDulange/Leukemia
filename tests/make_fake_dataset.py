"""
Generate synthetic 5-class leukemia dataset for local smoke-testing.

Creates 5 class folders with 12 images each (total 60 images):
  - Classes: AML, Early_PreB, Normal, PreB, ProB
  - Mixed formats: .jpg, .bmp, .tiff, .png
  - Mixed dimensions: 400x400, 640x480, 1024x768, up to 2560x1920
  - Includes near-duplicate images (e.g. rotated/flipped) to test canonical perceptual hash grouping
  - Includes purple/blue nuclei and pink cytoplasm to test HSV segmentation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["AML", "Early_PreB", "Normal", "PreB", "ProB"]

SIZES = [
    (400, 400),
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 960),
    (1920, 1080),
    (2560, 1920),  # large scan test
]

FORMAT_EXTS = [
    ("JPEG", ".jpg"),
    ("BMP", ".bmp"),
    ("TIFF", ".tiff"),
    ("PNG", ".png"),
]


def draw_synthetic_cell(width: int, height: int, rng: np.random.RandomState) -> Image.Image:
    """Create a synthetic blood smear patch with background, cytoplasm, and nucleus."""
    # Light background (smear slide)
    bg_color = (
        int(rng.randint(220, 245)),
        int(rng.randint(215, 240)),
        int(rng.randint(225, 250)),
    )
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Random position and radii for distinct perceptual hashes across cells
    cx = int(width * rng.uniform(0.3, 0.7))
    cy = int(height * rng.uniform(0.3, 0.7))
    r_cyto_x = int(min(width, height) * rng.uniform(0.18, 0.35))
    r_cyto_y = int(min(width, height) * rng.uniform(0.18, 0.35))

    # Cytoplasm (pink/purple)
    cyto_box = [cx - r_cyto_x, cy - r_cyto_y, cx + r_cyto_x, cy + r_cyto_y]
    draw.ellipse(
        cyto_box,
        fill=(rng.randint(180, 220), rng.randint(130, 170), rng.randint(190, 230)),
    )

    # Nucleus (dark purple/blue)
    r_nuc_x = int(r_cyto_x * rng.uniform(0.5, 0.8))
    r_nuc_y = int(r_cyto_y * rng.uniform(0.5, 0.8))
    nuc_box = [cx - r_nuc_x, cy - r_nuc_y, cx + r_nuc_x, cy + r_nuc_y]
    draw.ellipse(
        nuc_box,
        fill=(rng.randint(80, 120), rng.randint(40, 80), rng.randint(130, 180)),
    )

    # Add random cellular chromatin / granule blobs
    for _ in range(rng.randint(3, 8)):
        gx = cx + int(rng.uniform(-r_nuc_x * 0.7, r_nuc_x * 0.7))
        gy = cy + int(rng.uniform(-r_nuc_y * 0.7, r_nuc_y * 0.7))
        gr = int(min(width, height) * rng.uniform(0.01, 0.04))
        draw.ellipse([gx - gr, gy - gr, gx + gr, gy + gr], fill=(rng.randint(40, 70), rng.randint(20, 50), rng.randint(80, 120)))

    # Add a few erythrocyte-like circles in background
    for _ in range(rng.randint(2, 6)):
        ex = rng.randint(0, width)
        ey = rng.randint(0, height)
        er = rng.randint(15, 30)
        draw.ellipse([ex - er, ey - er, ex + er, ey + er], fill=(220, 160, 160))

    return img


def create_fake_dataset(output_dir: Path, images_per_class: int = 12) -> None:
    """Generate fake dataset for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    total_created = 0
    print(f"Creating fake dataset in {output_dir}...")

    for class_name in CLASSES:
        class_folder = output_dir / class_name
        class_folder.mkdir(parents=True, exist_ok=True)

        base_img: Image.Image | None = None

        for idx in range(images_per_class):
            width, height = SIZES[idx % len(SIZES)]
            fmt, ext = FORMAT_EXTS[idx % len(FORMAT_EXTS)]
            filename = f"sample_{class_name}_{idx:03d}{ext}"
            file_path = class_folder / filename

            # Make image 0 a base image, and image 1 & 2 rotated/flipped duplicates in this class
            if idx == 0:
                img = draw_synthetic_cell(width, height, rng)
                base_img = img
            elif idx == 1 and base_img is not None:
                # Rotated duplicate
                img = base_img.rotate(90, expand=True)
            elif idx == 2 and base_img is not None:
                # Flipped duplicate
                img = base_img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                img = draw_synthetic_cell(width, height, rng)

            # Save with appropriate format
            if fmt == "JPEG":
                img.save(file_path, "JPEG", quality=90)
            elif fmt == "BMP":
                img.save(file_path, "BMP")
            elif fmt == "TIFF":
                img.save(file_path, "TIFF")
            elif fmt == "PNG":
                img.save(file_path, "PNG")

            total_created += 1

        print(f"  Created {images_per_class} images in {class_name}/")

    print(f"\n[DONE] Created {total_created} total images across {len(CLASSES)} classes in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fake multi-class leukemia dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "fake_raw",
        help="Destination directory for fake dataset",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=12,
        help="Number of images per class (default: 12)",
    )
    args = parser.parse_args()
    create_fake_dataset(args.output_dir, args.images_per_class)


if __name__ == "__main__":
    main()

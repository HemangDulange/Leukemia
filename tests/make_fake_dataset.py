"""
Generate synthetic 5-class leukemia dataset for local smoke-testing.

Audit findings reflected in generator:
- Near-duplicates: each class includes the same photo saved at two sizes (1024x768 and 224x224)
  plus a horizontally flipped copy.
- PreB contains one non-smear dark-cell .bmp file to verify exclude_class_extensions filtering.
- AML includes TIFF files, Normal includes single-cell BMP files, and others are smear JPEGs.
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
    (2560, 1920),
]

FORMAT_EXTS = [
    ("JPEG", ".jpg"),
    ("BMP", ".bmp"),
    ("TIFF", ".tiff"),
    ("PNG", ".png"),
]


def draw_synthetic_cell(width: int, height: int, rng: np.random.RandomState) -> Image.Image:
    """Create a synthetic blood smear patch with background, cytoplasm, and nucleus."""
    bg_color = (
        int(rng.randint(220, 245)),
        int(rng.randint(215, 240)),
        int(rng.randint(225, 250)),
    )
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    cx = int(width * rng.uniform(0.3, 0.7))
    cy = int(height * rng.uniform(0.3, 0.7))
    r_cyto_x = int(min(width, height) * rng.uniform(0.18, 0.35))
    r_cyto_y = int(min(width, height) * rng.uniform(0.18, 0.35))

    cyto_box = [cx - r_cyto_x, cy - r_cyto_y, cx + r_cyto_x, cy + r_cyto_y]
    draw.ellipse(
        cyto_box,
        fill=(rng.randint(180, 220), rng.randint(130, 170), rng.randint(190, 230)),
    )

    r_nuc_x = int(r_cyto_x * rng.uniform(0.5, 0.8))
    r_nuc_y = int(r_cyto_y * rng.uniform(0.5, 0.8))
    nuc_box = [cx - r_nuc_x, cy - r_nuc_y, cx + r_nuc_x, cy + r_nuc_y]
    draw.ellipse(
        nuc_box,
        fill=(rng.randint(80, 120), rng.randint(40, 80), rng.randint(130, 180)),
    )

    for _ in range(rng.randint(3, 8)):
        gx = cx + int(rng.uniform(-r_nuc_x * 0.7, r_nuc_x * 0.7))
        gy = cy + int(rng.uniform(-r_nuc_y * 0.7, r_nuc_y * 0.7))
        gr = int(min(width, height) * rng.uniform(0.01, 0.04))
        draw.ellipse(
            [gx - gr, gy - gr, gx + gr, gy + gr],
            fill=(rng.randint(40, 70), rng.randint(20, 50), rng.randint(80, 120)),
        )

    for _ in range(rng.randint(2, 6)):
        ex = rng.randint(0, width)
        ey = rng.randint(0, height)
        er = rng.randint(15, 30)
        draw.ellipse([ex - er, ey - er, ex + er, ey + er], fill=(220, 160, 160))

    return img


def create_fake_dataset(
    output_dir: Path,
    images_per_class: int = 14,
    test_large_cluster: bool = False,
) -> None:
    """Generate fake dataset for testing.

    Each class contains:
    - Base image at 1024x768 (index 0)
    - Near-duplicate at 224x224 (index 1)
    - Near-duplicate flipped copy (index 2)
    - PreB additionally contains one non-smear .bmp file (index 3)
    - Other images have mixed formats matching audit findings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    total_created = 0
    print(f"Creating fake dataset in {output_dir}...")

    for class_name in CLASSES:
        class_folder = output_dir / class_name
        class_folder.mkdir(parents=True, exist_ok=True)

        n_images = 28 if (test_large_cluster and class_name == "AML") else images_per_class
        base_img: Image.Image | None = None

        for idx in range(n_images):
            # Default format and dimensions
            width, height = SIZES[idx % len(SIZES)]
            fmt, ext = ("JPEG", ".jpg")

            if class_name == "AML" and idx > 2:
                fmt, ext = ("TIFF", ".tiff")
                width, height = (400, 400)
            elif class_name == "Normal" and idx > 2 and (idx % 2 == 0):
                fmt, ext = ("BMP", ".bmp")
                width, height = (450, 450)
            elif class_name == "PreB" and idx == 3:
                # PreB non-smear .bmp from different source
                fmt, ext = ("BMP", ".bmp")
                width, height = (450, 450)

            filename = f"sample_{class_name}_{idx:03d}{ext}"
            file_path = class_folder / filename

            if idx == 0:
                # Base image at high resolution (1024x768)
                width, height = (1024, 768)
                fmt, ext = ("JPEG", ".jpg")
                filename = f"sample_{class_name}_{idx:03d}{ext}"
                file_path = class_folder / filename
                img = draw_synthetic_cell(width, height, rng)
                base_img = img
            elif idx == 1 and base_img is not None:
                # Near-duplicate: same image downsampled to 224x224
                width, height = (224, 224)
                fmt, ext = ("JPEG", ".jpg")
                filename = f"sample_{class_name}_{idx:03d}{ext}"
                file_path = class_folder / filename
                img = base_img.resize((width, height), Image.Resampling.BILINEAR)
            elif idx == 2 and base_img is not None:
                # Near-duplicate: horizontal flip of base image
                width, height = base_img.size
                fmt, ext = ("JPEG", ".jpg")
                filename = f"sample_{class_name}_{idx:03d}{ext}"
                file_path = class_folder / filename
                img = base_img.transpose(Image.FLIP_LEFT_RIGHT)
            elif test_large_cluster and class_name == "AML" and base_img is not None:
                rot_deg = (idx * 90) % 360
                img = base_img.rotate(rot_deg, expand=True)
            elif class_name == "PreB" and idx == 3:
                # Single dark cell on black background (.bmp)
                img = Image.new("RGB", (width, height), color=(10, 10, 10))
                draw = ImageDraw.Draw(img)
                cx, cy = width // 2, height // 2
                draw.ellipse([cx - 40, cy - 40, cx + 40, cy + 40], fill=(45, 25, 70))
            else:
                img = draw_synthetic_cell(width, height, rng)

            if fmt == "JPEG":
                img.save(file_path, "JPEG", quality=90)
            elif fmt == "BMP":
                img.save(file_path, "BMP")
            elif fmt == "TIFF":
                img.save(file_path, "TIFF")
            elif fmt == "PNG":
                img.save(file_path, "PNG")

            total_created += 1

        print(f"  Created {n_images} images in {class_name}/")

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
        default=14,
        help="Number of images per class (default: 14)",
    )
    parser.add_argument(
        "--test-large-cluster",
        action="store_true",
        help="Create a class with >25 near-duplicate images to verify cluster safety limit",
    )
    args = parser.parse_args()
    create_fake_dataset(
        args.output_dir,
        args.images_per_class,
        test_large_cluster=args.test_large_cluster,
    )


if __name__ == "__main__":
    main()

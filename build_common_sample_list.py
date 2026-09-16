"""List real WeedMap tiles usable by both RGB and multispectral inputs."""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image


SUBSETS = tuple(f"RedEdge_{number:03d}" for number in range(5)) + tuple(
    f"Sequoia_{number:03d}" for number in range(5, 8)
)
MULTISPECTRAL_CHANNELS = ("G", "R", "RE", "NIR", "NDVI")
SAMPLE_ID_PATTERN = re.compile(r"frame\d{4}")


def load_band(path):
    with Image.open(path) as image:
        band = np.asarray(image)
    if band.ndim == 3:
        band = band[..., 0]
    if band.ndim != 2:
        raise ValueError(f"Expected a single-channel image: {path}")
    return band


def sample_pixel_counts(rgb_path, band_paths, color_path, mask_path):
    """Return valid crop/weed counts, or None for an empty/invalid tile."""
    with Image.open(rgb_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    if not np.any(rgb):
        return None

    image_shape = rgb.shape[:2]
    multispectral_has_data = False
    for path in band_paths:
        band = load_band(path)
        if band.shape != image_shape:
            return None
        multispectral_has_data |= bool(np.any(band))
    if not multispectral_has_data:
        return None

    with Image.open(color_path) as image:
        color = np.asarray(image.convert("RGB"))
    mask = load_band(mask_path)
    if color.shape[:2] != image_shape or mask.shape != image_shape:
        return None

    # Match WeedMapDataset: unknown colors and mask=255 are ignored.
    label = np.full(image_shape, 255, dtype=np.uint8)
    label[np.all(color == (0, 0, 0), axis=-1)] = 0
    label[np.all(color == (0, 255, 0), axis=-1)] = 1
    label[np.all(color == (255, 0, 0), axis=-1)] = 2
    label[mask == 255] = 255

    crop_pixels = int(np.count_nonzero(label == 1))
    weed_pixels = int(np.count_nonzero(label == 2))
    if not np.any(label != 255) or crop_pixels + weed_pixels == 0:
        return None
    return crop_pixels, weed_pixels


def build_common_sample_list(data_root, output_path):
    if not data_root.is_dir():
        raise FileNotFoundError(f"WeedMap data directory does not exist: {data_root}")

    rows = []
    total_crop = total_weed = 0
    subset_counts = {}
    for subset_name in SUBSETS:
        sensor, subset_id = subset_name.split("_")
        subset_root = data_root / subset_name / subset_id
        groundtruth_dir = subset_root / "groundtruth"
        tile_dir = subset_root / "tile"
        mask_dir = subset_root / "mask"
        count = 0

        for color_path in sorted(groundtruth_dir.glob("*_GroundTruth_color.png")):
            prefix = f"{subset_id}_"
            suffix = "_GroundTruth_color"
            stem = color_path.stem
            if not stem.startswith(prefix) or not stem.endswith(suffix):
                continue
            sample_id = stem[len(prefix) : -len(suffix)]
            if not SAMPLE_ID_PATTERN.fullmatch(sample_id):
                continue

            rgb_path = tile_dir / "RGB" / f"{sample_id}.png"
            band_paths = tuple(
                tile_dir / channel / f"{sample_id}.png"
                for channel in MULTISPECTRAL_CHANNELS
            )
            mask_path = mask_dir / f"{sample_id}.png"
            if not all(path.is_file() for path in (rgb_path, *band_paths, mask_path)):
                continue

            try:
                pixel_counts = sample_pixel_counts(
                    rgb_path, band_paths, color_path, mask_path
                )
            except (OSError, ValueError):
                continue
            if pixel_counts is None:
                continue

            rows.append((sensor, subset_id, sample_id))
            crop_pixels, weed_pixels = pixel_counts
            total_crop += crop_pixels
            total_weed += weed_pixels
            count += 1

        subset_counts[subset_name] = count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("sensor", "subset_id", "sample_id"))
        writer.writerows(rows)

    print(f"Common samples: {len(rows)}")
    for subset_name, count in subset_counts.items():
        print(f"  {subset_name}: {count}")
    print(f"Approximate foreground pixels: crop={total_crop:,}, weed={total_weed:,}")
    print(f"CSV saved to: {output_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/weedmap"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("splits/real_weedmap_common_samples.csv"),
    )
    args = parser.parse_args()
    build_common_sample_list(args.data_root, args.output)


if __name__ == "__main__":
    main()

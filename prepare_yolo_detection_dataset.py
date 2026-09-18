"""Convert WeedMap pixel labels to a YOLO detection dataset.

The sample order, filtering, and 80/20 split match train_real_weedmap_unet.py
when it uses the same common-sample CSV and seed. Connected components use
8-neighbor connectivity. Each retained component becomes one detection box.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from weedmap_dataset import WeedMapDataset


def connected_components(binary_mask, min_area):
    """Return (xmin, ymin, xmax, ymax, area) with exclusive max coordinates."""
    runs = []  # (xmin, xmax_exclusive, y)
    parent = []
    rank = []
    previous = []

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left == right:
            return
        if rank[left] < rank[right]:
            left, right = right, left
        parent[right] = left
        if rank[left] == rank[right]:
            rank[left] += 1

    for y, row in enumerate(binary_mask):
        edges = np.flatnonzero(np.diff(np.pad(row, (1, 1))))
        current = []
        for xmin, xmax in edges.reshape(-1, 2):
            index = len(runs)
            runs.append((int(xmin), int(xmax), y))
            parent.append(index)
            rank.append(0)
            current.append(index)

        # Runs in adjacent rows touch under 8-neighbor connectivity when
        # their horizontal intervals overlap or are one pixel apart.
        previous_pos = 0
        for index in current:
            xmin, xmax, _ = runs[index]
            while (
                previous_pos < len(previous)
                and runs[previous[previous_pos]][1] < xmin
            ):
                previous_pos += 1
            pos = previous_pos
            while pos < len(previous) and runs[previous[pos]][0] <= xmax:
                union(index, previous[pos])
                pos += 1
        previous = current

    bounds = {}
    for index, (xmin, xmax, y) in enumerate(runs):
        root = find(index)
        if root not in bounds:
            bounds[root] = [xmin, y, xmax, y + 1, 0]
        box = bounds[root]
        box[0] = min(box[0], xmin)
        box[1] = min(box[1], y)
        box[2] = max(box[2], xmax)
        box[3] = max(box[3], y + 1)
        box[4] += xmax - xmin

    return sorted(
        (tuple(box) for box in bounds.values() if box[4] >= min_area),
        key=lambda box: (box[1], box[0]),
    )


def yolo_lines(
    label,
    min_area,
    max_box_area_ratio=0.25,
    min_box_width=4,
    min_box_height=4,
    skip_border_touching=False,
    target_classes="crop_weed",
):
    if target_classes not in ("crop_weed", "weed_only"):
        raise ValueError(f"Unsupported target classes: {target_classes}")
    height, width = label.shape
    lines = []
    counts = {
        "crop": 0,
        "weed": 0,
        "skipped small boxes": 0,
        "skipped huge boxes": 0,
        "skipped border boxes": 0,
    }
    classes = (
        ((1, 0, "crop"), (2, 1, "weed"))
        if target_classes == "crop_weed"
        else ((2, 0, "weed"),)
    )
    for pixel_class, yolo_class, name in classes:
        for xmin, ymin, xmax, ymax, area in connected_components(
            label == pixel_class, 1
        ):
            box_width_pixels = xmax - xmin
            box_height_pixels = ymax - ymin
            if (
                area < min_area
                or box_width_pixels < min_box_width
                or box_height_pixels < min_box_height
            ):
                counts["skipped small boxes"] += 1
                continue
            if (
                box_width_pixels * box_height_pixels / (width * height)
                > max_box_area_ratio
            ):
                counts["skipped huge boxes"] += 1
                continue
            if skip_border_touching and (
                xmin == 0 or ymin == 0 or xmax == width or ymax == height
            ):
                counts["skipped border boxes"] += 1
                continue
            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)
            box_width = box_width_pixels / width
            box_height = box_height_pixels / height
            lines.append(
                f"{yolo_class} {x_center:.8f} {y_center:.8f} "
                f"{box_width:.8f} {box_height:.8f}"
            )
            counts[name] += 1
    return lines, counts


def prepare_dataset(
    data_root,
    sample_list_csv,
    output_dir,
    seed,
    min_area,
    max_box_area_ratio=0.25,
    min_box_width=4,
    min_box_height=4,
    skip_border_touching=False,
    overwrite=False,
    target_classes="crop_weed",
):
    if target_classes not in ("crop_weed", "weed_only"):
        raise ValueError(f"Unsupported target classes: {target_classes}")
    if min_area < 1:
        raise ValueError("--min-area must be at least 1")
    if not 0 < max_box_area_ratio <= 1:
        raise ValueError("--max-box-area-ratio must be greater than 0 and at most 1")
    if min_box_width < 1:
        raise ValueError("--min-box-width must be at least 1")
    if min_box_height < 1:
        raise ValueError("--min-box-height must be at least 1")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Use --overwrite to regenerate it or choose an empty --output-dir."
            )
        if (
            output_dir.is_symlink()
            or "yolo_weedmap_detect" not in output_dir.resolve().name
        ):
            raise ValueError(
                f"Refusing to delete unsafe output directory: {output_dir}. "
                "The directory name must contain 'yolo_weedmap_detect' and must not be a symlink."
            )
        shutil.rmtree(output_dir)

    # Match the multispectral U-Net experiment's filtered sample order.
    dataset = WeedMapDataset(
        data_root=data_root,
        input_type="multispectral",
        filter_empty=True,
        sample_list_csv=sample_list_csv,
    )
    total = len(dataset)
    train_size = int(0.8 * total)
    if train_size == 0 or train_size == total:
        raise ValueError(f"At least two valid samples are required; found {total}")

    indices = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).tolist()
    splits = {"train": indices[:train_size], "val": indices[train_size:]}
    counts = {
        "crop": 0,
        "weed": 0,
        "empty": 0,
        "skipped small boxes": 0,
        "skipped huge boxes": 0,
        "skipped border boxes": 0,
    }

    for split in splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, sample_indices in splits.items():
        for index in sample_indices:
            input_paths, color_path, mask_path = dataset.samples[index]
            rgb_path = input_paths[0].parent.parent / "RGB" / input_paths[0].name
            if not rgb_path.is_file():
                raise FileNotFoundError(f"RGB image missing: {rgb_path}")

            with Image.open(rgb_path) as image:
                width, height = image.size
            label = dataset._load_label(color_path, mask_path, (height, width))
            lines, image_counts = yolo_lines(
                label,
                min_area,
                max_box_area_ratio,
                min_box_width,
                min_box_height,
                skip_border_touching,
                target_classes,
            )

            # Prefix with the subset name to keep frame IDs from different
            # WeedMap sequences distinct in YOLO's flat split directories.
            subset_name = rgb_path.parents[3].name
            stem = f"{subset_name}_{rgb_path.stem}"
            shutil.copy2(rgb_path, output_dir / "images" / split / f"{stem}.png")
            (output_dir / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            counts["crop"] += image_counts["crop"]
            counts["weed"] += image_counts["weed"]
            for key in (
                "skipped small boxes",
                "skipped huge boxes",
                "skipped border boxes",
            ):
                counts[key] += image_counts[key]
            counts["empty"] += not lines

    yaml_text = (
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
    )
    if target_classes == "weed_only":
        yaml_text += 'nc: 1\nnames: ["weed"]\n'
    else:
        yaml_text += "names:\n  0: crop\n  1: weed\n"
    (output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"train image count: {len(splits['train'])}")
    print(f"val image count: {len(splits['val'])}")
    print(f"crop box count: {counts['crop']}")
    print(f"weed box count: {counts['weed']}")
    print(f"empty image count: {counts['empty']}")
    for key in ("skipped small boxes", "skipped huge boxes", "skipped border boxes"):
        print(f"{key}: {counts[key]}")
    print(f"YOLO dataset: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/weedmap"))
    parser.add_argument(
        "--sample-list-csv",
        type=Path,
        default=Path("splits/real_weedmap_common_samples.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/yolo_weedmap_detect")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.25)
    parser.add_argument("--min-box-width", type=int, default=4)
    parser.add_argument("--min-box-height", type=int, default=4)
    parser.add_argument("--skip-border-touching", action="store_true")
    parser.add_argument(
        "--target-classes", choices=("crop_weed", "weed_only"), default="crop_weed"
    )
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()
    prepare_dataset(
        args.data_root,
        args.sample_list_csv,
        args.output_dir,
        args.seed,
        args.min_area,
        args.max_box_area_ratio,
        args.min_box_width,
        args.min_box_height,
        args.skip_border_touching,
        args.overwrite,
        args.target_classes,
    )


if __name__ == "__main__":
    main()

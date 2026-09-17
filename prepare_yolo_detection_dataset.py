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


def yolo_lines(label, min_area):
    height, width = label.shape
    lines = []
    counts = {"crop": 0, "weed": 0}
    for pixel_class, yolo_class, name in ((1, 0, "crop"), (2, 1, "weed")):
        for xmin, ymin, xmax, ymax, _ in connected_components(
            label == pixel_class, min_area
        ):
            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height
            lines.append(
                f"{yolo_class} {x_center:.8f} {y_center:.8f} "
                f"{box_width:.8f} {box_height:.8f}"
            )
            counts[name] += 1
    return lines, counts


def prepare_dataset(data_root, sample_list_csv, output_dir, seed, min_area):
    if min_area < 1:
        raise ValueError("--min-area must be at least 1")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose an empty --output-dir to avoid mixing old and new labels."
        )

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
    counts = {"crop": 0, "weed": 0, "empty": 0}

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
            lines, image_counts = yolo_lines(label, min_area)

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
            counts["empty"] += not lines

    (output_dir / "data.yaml").write_text(
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: crop\n"
        "  1: weed\n",
        encoding="utf-8",
    )
    print(f"train image count: {len(splits['train'])}")
    print(f"val image count: {len(splits['val'])}")
    print(f"crop box count: {counts['crop']}")
    print(f"weed box count: {counts['weed']}")
    print(f"empty image count: {counts['empty']}")
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
    args = parser.parse_args()
    prepare_dataset(
        args.data_root, args.sample_list_csv, args.output_dir, args.seed, args.min_area
    )


if __name__ == "__main__":
    main()

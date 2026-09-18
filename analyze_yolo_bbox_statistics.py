"""Summarize the existing YOLO detection labels without changing the dataset."""

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CLASS_NAMES = {0: "crop", 1: "weed"}
COLORS = {"crop": "#2e7d32", "weed": "#c62828"}
FIELDS = (
    "split", "image_filename", "class_id", "class_name", "x_center", "y_center",
    "width_norm", "height_norm", "area_norm", "aspect_ratio", "image_width",
    "image_height", "width_px", "height_px", "area_px",
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def check_data_yaml(path):
    """Read the simple data.yaml emitted by prepare_yolo_detection_dataset.py."""
    content = path.read_text(encoding="utf-8")
    settings = {}
    names = {}
    in_names = False
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "names:":
            in_names = True
            continue
        if in_names and line[:1].isspace():
            match = re.fullmatch(r"(\d+)\s*:\s*['\"]?([\w-]+)['\"]?", stripped)
            if match:
                names[int(match.group(1))] = match.group(2)
            continue
        in_names = False
        key, separator, value = stripped.partition(":")
        if separator:
            settings[key] = value.strip().strip("'\"")
    if settings.get("train") != "images/train" or settings.get("val") != "images/val":
        raise ValueError(f"{path}: expected train/val image paths")
    if names != CLASS_NAMES:
        raise ValueError(f"{path}: expected class names {CLASS_NAMES}, got {names}")


def read_label(path, split, image_filename, image_width, image_height):
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields, got {len(fields)}")
        try:
            class_id = int(fields[0])
            x_center, y_center, width_norm, height_norm = map(float, fields[1:])
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: invalid YOLO label") from error
        if class_id not in CLASS_NAMES:
            raise ValueError(f"{path}:{line_number}: unknown class {class_id}")
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in
                   (x_center, y_center, width_norm, height_norm)) or width_norm == 0 or height_norm == 0:
            raise ValueError(f"{path}:{line_number}: coordinates must be in [0, 1] with positive size")
        if (x_center - width_norm / 2 < -1e-6 or x_center + width_norm / 2 > 1 + 1e-6
                or y_center - height_norm / 2 < -1e-6 or y_center + height_norm / 2 > 1 + 1e-6):
            raise ValueError(f"{path}:{line_number}: box extends outside the image")
        width_px = width_norm * image_width
        height_px = height_norm * image_height
        rows.append({
            "split": split,
            "image_filename": image_filename,
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "x_center": x_center,
            "y_center": y_center,
            "width_norm": width_norm,
            "height_norm": height_norm,
            "area_norm": width_norm * height_norm,
            "aspect_ratio": width_norm / height_norm,
            "image_width": image_width,
            "image_height": image_height,
            "width_px": width_px,
            "height_px": height_px,
            "area_px": width_px * height_px,
        })
    return rows


def collect(dataset_dir):
    check_data_yaml(dataset_dir / "data.yaml")
    rows, boxes_per_image = [], []
    for split in ("train", "val"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        images = sorted(path for path in image_dir.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"No images found in {image_dir}")
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        stems = [path.stem for path in images]
        if len(stems) != len(set(stems)):
            raise ValueError(f"Duplicate image stems in {image_dir}")
        missing = set(stems) - labels.keys()
        extra = labels.keys() - set(stems)
        if missing or extra:
            raise ValueError(f"{split}: missing labels={sorted(missing)}, orphan labels={sorted(extra)}")
        for image_path in images:
            with Image.open(image_path) as image:
                image_width, image_height = image.size
            image_rows = read_label(labels[image_path.stem], split, image_path.name,
                                    image_width, image_height)
            rows.extend(image_rows)
            boxes_per_image.append(len(image_rows))
    return rows, boxes_per_image


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_statistics(rows, boxes_per_image):
    print(f"total boxes: {len(rows)}")
    for name in ("crop", "weed"):
        print(f"{name} boxes: {sum(row['class_name'] == name for row in rows)}")
    for split in ("train", "val"):
        print(f"{split} boxes: {sum(row['split'] == split for row in rows)}")
    print(f"average boxes per image: {mean(boxes_per_image):.4f}")
    print(f"boxes per image (min / median / max): "
          f"{min(boxes_per_image)} / {median(boxes_per_image):g} / {max(boxes_per_image)}")
    for name in ("crop", "weed"):
        class_rows = [row for row in rows if row["class_name"] == name]
        print(f"{name} bbox statistics (mean / median / min / max):")
        for field in ("width_px", "height_px", "area_px", "aspect_ratio", "area_norm"):
            values = [row[field] for row in class_rows]
            if values:
                print(f"  {field}: {mean(values):.6g} / {median(values):.6g} / "
                      f"{min(values):.6g} / {max(values):.6g}")
            else:
                print(f"  {field}: n/a (no boxes)")


def plot_class_histogram(rows, field, xlabel, title, path):
    values = [row[field] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        if values:
            low, high = min(values), max(values)
            bins = np.geomspace(low, high, 45) if high > low else np.linspace(low * 0.9, high * 1.1, 3)
            for name in ("crop", "weed"):
                class_values = [row[field] for row in rows if row["class_name"] == name]
                if class_values:
                    ax.hist(class_values, bins=bins, alpha=0.55, color=COLORS[name],
                            label=f"{name} (n={len(class_values)})")
            ax.set_xscale("log")
            ax.legend()
        ax.set(xlabel=xlabel, ylabel="Box count", title=title)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def plot_scatter(rows, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        for name, marker in (("crop", "o"), ("weed", "x")):
            class_rows = [row for row in rows if row["class_name"] == name]
            if class_rows:
                ax.scatter([row["width_px"] for row in class_rows],
                           [row["height_px"] for row in class_rows],
                           s=14, alpha=0.35, marker=marker, color=COLORS[name],
                           label=f"{name} (n={len(class_rows)})", rasterized=True)
        ax.set(xlabel="BBox width (px)", ylabel="BBox height (px)",
               title="YOLO bbox width vs height")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def plot_boxes_per_image(counts, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        bins = np.arange(min(counts), max(counts) + 2) - 0.5
        ax.hist(counts, bins=bins, color="#3f6ea8", edgecolor="white")
        ax.set(xlabel="Boxes per image", ylabel="Image count",
               title="YOLO boxes per image")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/yolo_weedmap_detect"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/yolo_bbox_statistics.csv"))
    parser.add_argument("--assets-dir", type=Path, default=Path("reports/assets"))
    parser.add_argument("--plot-prefix", default="")
    args = parser.parse_args()

    rows, boxes_per_image = collect(args.dataset_dir)
    write_csv(rows, args.output_csv)
    print_statistics(rows, boxes_per_image)
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    plot_prefix = f"{args.plot_prefix}_" if args.plot_prefix else ""
    plot_class_histogram(rows, "area_px", "BBox area (px²)", "YOLO bbox area distribution",
                         args.assets_dir / f"{plot_prefix}yolo_bbox_area_distribution.png")
    plot_scatter(rows, args.assets_dir / f"{plot_prefix}yolo_bbox_width_height_scatter.png")
    plot_boxes_per_image(boxes_per_image,
                         args.assets_dir / f"{plot_prefix}yolo_boxes_per_image_distribution.png")
    plot_class_histogram(rows, "area_norm", "BBox / image area ratio",
                         "YOLO bbox / image area ratio distribution",
                         args.assets_dir / f"{plot_prefix}yolo_bbox_area_ratio_distribution.png")
    print(f"CSV saved: {args.output_csv}")
    print(f"Plots saved: {args.assets_dir}")


if __name__ == "__main__":
    main()

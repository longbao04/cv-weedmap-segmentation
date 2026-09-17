"""Draw YOLO detection labels on an RGB image for visual inspection."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


CLASS_STYLES = {0: ("crop", "green"), 1: ("weed", "red")}


def read_boxes(label_path, image_width, image_height):
    """Read normalized YOLO boxes and return pixel rectangles with class IDs."""
    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 fields, got {len(fields)}")
        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = map(float, fields[1:])
        except ValueError as error:
            raise ValueError(f"{label_path}:{line_number}: invalid YOLO label") from error
        if class_id not in CLASS_STYLES:
            raise ValueError(f"{label_path}:{line_number}: unknown class {class_id}")
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in
                   (x_center, y_center, width, height)) or width == 0 or height == 0:
            raise ValueError(f"{label_path}:{line_number}: coordinates must be in [0, 1] with positive size")

        left = x_center - width / 2
        top = y_center - height / 2
        right = x_center + width / 2
        bottom = y_center + height / 2
        # Label coordinates are rounded to eight decimals by the dataset converter.
        if min(left, top) < -1e-6 or max(right, bottom) > 1 + 1e-6:
            raise ValueError(f"{label_path}:{line_number}: box extends outside the image")
        left, top = max(left, 0), max(top, 0)
        right, bottom = min(right, 1), min(bottom, 1)
        boxes.append((class_id, left * image_width, top * image_height,
                      (right - left) * image_width, (bottom - top) * image_height))
    return boxes


def visualize(dataset_dir, split, sample_name, output):
    image_path = dataset_dir / "images" / split / f"{sample_name}.png"
    label_path = dataset_dir / "labels" / split / f"{sample_name}.txt"
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"Label not found: {label_path}")

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        image_width, image_height = rgb.size
        boxes = read_boxes(label_path, image_width, image_height)

    fig, ax = plt.subplots(figsize=(image_width / 100, image_height / 100), dpi=100)
    try:
        ax.set_position((0, 0, 1, 1))
        ax.imshow(rgb, extent=(0, image_width, image_height, 0))
        for class_id, left, top, width, height in boxes:
            ax.add_patch(Rectangle((left, top), width, height, fill=False,
                                   edgecolor=CLASS_STYLES[class_id][1], linewidth=1.5))
        ax.set_xlim(0, image_width)
        ax.set_ylim(image_height, 0)
        ax.axis("off")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=100, pad_inches=0)
    finally:
        plt.close(fig)

    crop_count = sum(class_id == 0 for class_id, *_ in boxes)
    weed_count = sum(class_id == 1 for class_id, *_ in boxes)
    print(f"image path: {image_path}")
    print(f"label path: {label_path}")
    print(f"total boxes: {len(boxes)}")
    print(f"crop boxes: {crop_count}")
    print(f"weed boxes: {weed_count}")
    print(f"output path: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/yolo_weedmap_detect"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-name", default="RedEdge_004_frame0070")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/yolo_label_visualization_RedEdge_004_frame0070.png"))
    args = parser.parse_args()
    visualize(args.dataset_dir, args.split, args.sample_name, args.output)


if __name__ == "__main__":
    main()

"""Summarize YOLO prediction counts across inference conf and NMS IoU settings."""

import argparse
import csv
import math
import sys
from pathlib import Path

import yaml
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
FIELDS = (
    "conf", "iou", "num_images", "total_pred_boxes", "mean_boxes_per_image",
    "crop_pred_boxes", "weed_pred_boxes", "mean_confidence",
    "mean_crop_confidence", "mean_weed_confidence",
)


def checked_thresholds(parser, values, option):
    if not values or any(not math.isfinite(value) or not 0 < value < 1 for value in values):
        parser.error(f"{option} requires one or more values strictly between 0 and 1")
    if len(values) != len(set(values)):
        parser.error(f"{option} contains duplicate values")
    return values


def class_ids(data_yaml):
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Data YAML not found: {data_yaml}")
    with data_yaml.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    names = data.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict):
        raise ValueError(f"Expected class names in {data_yaml}")
    by_name = {str(name).lower(): int(index) for index, name in names.items()}
    if set(by_name) != {"crop", "weed"} or len(names) != 2:
        raise ValueError(f"Expected exactly crop and weed classes in {data_yaml}")
    return by_name


def validation_images(image_dir, sample_limit):
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Validation image directory not found: {image_dir}")
    paths = sorted(path for path in image_dir.iterdir()
                   if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"No validation images found in {image_dir}")
    return paths[:sample_limit]


def mean(values):
    return sum(values) / len(values) if values else ""


def summarize(results, conf, iou, ids):
    all_confidences = []
    crop_confidences = []
    weed_confidences = []
    for result in results:
        for class_id, confidence in zip(result.boxes.cls.cpu().tolist(),
                                        result.boxes.conf.cpu().tolist()):
            class_id = int(class_id)
            if class_id == ids["crop"]:
                crop_confidences.append(confidence)
            elif class_id == ids["weed"]:
                weed_confidences.append(confidence)
            else:
                raise ValueError(f"Unexpected predicted class ID: {class_id}")
            all_confidences.append(confidence)
    return {
        "conf": conf,
        "iou": iou,
        "num_images": len(results),
        "total_pred_boxes": len(all_confidences),
        "mean_boxes_per_image": len(all_confidences) / len(results),
        "crop_pred_boxes": len(crop_confidences),
        "weed_pred_boxes": len(weed_confidences),
        "mean_confidence": mean(all_confidences),
        "mean_crop_confidence": mean(crop_confidences),
        "mean_weed_confidence": mean(weed_confidences),
    }


def print_table(rows):
    labels = ("conf", "iou", "images", "boxes", "boxes/img", "crop", "weed",
              "mean conf", "crop conf", "weed conf")
    cells = []
    for row in rows:
        cells.append([
            f"{row['conf']:.2f}", f"{row['iou']:.2f}", str(row["num_images"]),
            str(row["total_pred_boxes"]), f"{row['mean_boxes_per_image']:.2f}",
            str(row["crop_pred_boxes"]), str(row["weed_pred_boxes"]),
            *(f"{row[key]:.3f}" if row[key] != "" else "—" for key in
              ("mean_confidence", "mean_crop_confidence", "mean_weed_confidence")),
        ])
    widths = [max(len(label), *(len(row[i]) for row in cells))
              for i, label in enumerate(labels)]
    print("  ".join(label.rjust(width) for label, width in zip(labels, widths)))
    print("  ".join("-" * width for width in widths))
    for row in cells:
        print("  ".join(value.rjust(width) for value, width in zip(row, widths)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "runs/detect/runs/yolo_weedmap_compare/r010_5epochs/weights/best.pt")
    parser.add_argument("--data-yaml", type=Path, default=ROOT / "data/yolo_weedmap_detect_r010/data.yaml")
    parser.add_argument("--image-dir", type=Path, default=ROOT / "data/yolo_weedmap_detect_r010/images/val")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "outputs/yolo_postprocessing_threshold_summary.csv")
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--conf-values", nargs="+", type=float, default=[0.25, 0.30, 0.40, 0.50])
    parser.add_argument("--iou-values", nargs="+", type=float, default=[0.50, 0.60, 0.70])
    args = parser.parse_args()
    if args.sample_limit < 1:
        parser.error("--sample-limit must be positive")
    checked_thresholds(parser, args.conf_values, "--conf-values")
    checked_thresholds(parser, args.iou_values, "--iou-values")

    try:
        if not args.model.is_file():
            raise FileNotFoundError(f"Model weights not found: {args.model}")
        ids = class_ids(args.data_yaml)
        paths = validation_images(args.image_dir, args.sample_limit)
        model = YOLO(str(args.model))
        model_names = {str(name).lower(): int(index) for index, name in model.names.items()}
        if model_names != ids:
            raise ValueError(f"Model classes {model.names} do not match {args.data_yaml}")

        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        rows = []
        for conf in args.conf_values:
            for iou in args.iou_values:
                results = model.predict(source=images, conf=conf, iou=iou,
                                        batch=4, verbose=False)
                if len(results) != len(paths):
                    raise RuntimeError("Prediction count does not match sampled image count")
                rows.append(summarize(results, conf, iou, ids))

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"YOLO post-processing threshold analysis ({len(paths)} sorted validation images)")
        print_table(rows)
        print(f"\nSaved: {args.output_csv}")
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

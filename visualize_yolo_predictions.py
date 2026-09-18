"""Compare two trained YOLO detectors on the same WeedMap validation image."""

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parent
MODELS = (
    {
        "name": "crop+weed r010",
        "dataset": ROOT / "data/yolo_weedmap_detect_r010",
        "weights": ROOT / "runs/detect/runs/yolo_weedmap_compare/r010_5epochs/weights/best.pt",
        "classes": {0: "crop", 1: "weed"},
    },
    {
        "name": "weed-only r010",
        "dataset": ROOT / "data/yolo_weedmap_detect_weed_only_r010",
        "weights": ROOT / "runs/detect/runs/yolo_weedmap_compare/weed_only_r010_5epochs/weights/best.pt",
        "classes": {0: "weed"},
    },
)
COLORS = {"crop": "#55e05a", "weed": "#ff4e55"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def validation_images(dataset):
    image_dir = dataset / "images/val"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Validation image directory not found: {image_dir}")
    paths = sorted(path for path in image_dir.iterdir()
                   if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"No validation images found in {image_dir}")
    return paths


def read_gt_boxes(label_path, width, height, classes):
    if not label_path.is_file():
        raise FileNotFoundError(f"Ground truth label not found: {label_path}")
    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 YOLO label fields")
        try:
            class_id = int(fields[0])
            xc, yc, w, h = map(float, fields[1:])
        except ValueError as error:
            raise ValueError(f"{label_path}:{line_number}: invalid YOLO label") from error
        if class_id not in classes:
            raise ValueError(f"{label_path}:{line_number}: unexpected class ID {class_id}")
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (xc, yc, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"{label_path}:{line_number}: invalid normalized box")
        x1, y1 = max(0, xc - w / 2) * width, max(0, yc - h / 2) * height
        x2, y2 = min(1, xc + w / 2) * width, min(1, yc + h / 2) * height
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"{label_path}:{line_number}: box is outside the image")
        boxes.append((classes[class_id], x1, y1, x2, y2, None))
    return boxes


def predict_boxes(weights, image_path, classes, conf, iou):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    result = model.predict(source=str(image_path), conf=conf, iou=iou,
                           device="cpu", verbose=False)[0]
    boxes = []
    for xyxy, class_id, confidence in zip(result.boxes.xyxy.cpu().tolist(),
                                           result.boxes.cls.cpu().tolist(),
                                           result.boxes.conf.cpu().tolist()):
        class_id = int(class_id)
        if class_id not in classes:
            raise ValueError(f"{weights}: unexpected predicted class ID {class_id}")
        boxes.append((classes[class_id], *xyxy, confidence))
    return boxes


def draw_panel(ax, image, boxes, title):
    ax.imshow(image)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    for class_name, x1, y1, x2, y2, confidence in boxes:
        color = COLORS[class_name]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor=color, linewidth=1.5))
        label = class_name if confidence is None else f"{class_name} {confidence:.2f}"
        ax.text(x1, max(0, y1 - 2), label, color="white", fontsize=7,
                va="bottom", bbox={"facecolor": color, "edgecolor": "none", "pad": 1})


def print_stats(name, filename, gt_boxes, predicted_boxes):
    confidences = [box[5] for box in predicted_boxes]
    print(f"\n{name}")
    print(f"  image filename: {filename}")
    print(f"  number of GT boxes: {len(gt_boxes)}")
    print(f"  number of predicted boxes: {len(predicted_boxes)}")
    print(f"  number of predicted crop boxes: {sum(box[0] == 'crop' for box in predicted_boxes)}")
    print(f"  number of predicted weed boxes: {sum(box[0] == 'weed' for box in predicted_boxes)}")
    print(f"  mean confidence: {sum(confidences) / len(confidences):.4f}" if confidences
          else "  mean confidence: N/A (no predictions)")
    print(f"  max confidence: {max(confidences):.4f}" if confidences
          else "  max confidence: N/A (no predictions)")


def visualize(sample_index, conf, iou):
    missing = [model["weights"] for model in MODELS if not model["weights"].is_file()]
    if missing:
        raise FileNotFoundError("YOLO model weight file(s) missing: " + ", ".join(map(str, missing)))

    image_lists = [validation_images(model["dataset"]) for model in MODELS]
    if [path.name for path in image_lists[0]] != [path.name for path in image_lists[1]]:
        raise ValueError("Validation image filenames differ between the two YOLO datasets")
    if not 0 <= sample_index < len(image_lists[0]):
        raise ValueError(f"sample-index must be between 0 and {len(image_lists[0]) - 1}")

    chosen_paths = [paths[sample_index] for paths in image_lists]
    with Image.open(chosen_paths[0]) as loaded:
        image = loaded.convert("RGB")
    with Image.open(chosen_paths[1]) as loaded:
        other_image = loaded.convert("RGB")
    if image.size != other_image.size or image.tobytes() != other_image.tobytes():
        raise ValueError(f"Validation images differ for {chosen_paths[0].name}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    try:
        for row, (model, image_path) in enumerate(zip(MODELS, chosen_paths)):
            label_path = model["dataset"] / "labels/val" / f"{image_path.stem}.txt"
            gt_boxes = read_gt_boxes(label_path, *image.size, model["classes"])
            predicted_boxes = predict_boxes(model["weights"], image_path,
                                            model["classes"], conf, iou)
            draw_panel(axes[row, 0], image, [], "RGB image")
            draw_panel(axes[row, 1], image, gt_boxes, f"{model['name']} GT")
            draw_panel(axes[row, 2], image, predicted_boxes, f"{model['name']} prediction")
            print_stats(model["name"], image_path.name, gt_boxes, predicted_boxes)

        fig.suptitle("Box colors: crop = green    weed = red", fontsize=13)
        output = ROOT / f"outputs/yolo_prediction_comparison_sample{sample_index}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)
    print(f"\nSaved comparison: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    args = parser.parse_args()
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        parser.error("--conf and --iou must each be between 0 and 1")
    try:
        visualize(args.sample_index, args.conf, args.iou)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

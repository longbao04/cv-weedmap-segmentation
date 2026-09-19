"""Extract confidence operating points from a YOLO validation sweep."""

import argparse
import csv
from pathlib import Path

import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "runs/detect/runs/yolo_weedmap_compare/r010_5epochs/weights/best.pt",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/yolo_weedmap_detect_r010/data.yaml",
    )
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--candidate-confidences",
        type=float,
        nargs="+",
        default=(0.25, 0.30, 0.40, 0.50),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/yolo_validation_curve_summary.csv",
    )
    parser.add_argument("--plots", action="store_true")
    return parser.parse_args()


def curve(metrics, name):
    try:
        index = metrics.curves.index(name)
    except ValueError as error:
        raise RuntimeError(f"Ultralytics did not return the {name} curve") from error
    x_values, y_values, _, _ = metrics.curves_results[index]
    return np.asarray(x_values), np.asarray(y_values)


def main():
    args = parse_args()
    metrics = YOLO(str(args.model)).val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=args.plots,
        verbose=False,
    )
    confidence, f1 = curve(metrics, "F1-Confidence(B)")
    precision_confidence, precision = curve(metrics, "Precision-Confidence(B)")
    recall_confidence, recall = curve(metrics, "Recall-Confidence(B)")
    if not (np.array_equal(confidence, precision_confidence)
            and np.array_equal(confidence, recall_confidence)):
        raise RuntimeError("Confidence grids differ between validation curves")

    class_names = [metrics.names[index] for index in range(f1.shape[0])]
    rows = []
    mean_f1 = f1.mean(axis=0)
    best_index = int(mean_f1.argmax())
    rows.append({
        "scope": "all",
        "setting": "best_mean_f1",
        "confidence": confidence[best_index],
        "precision": precision[:, best_index].mean(),
        "recall": recall[:, best_index].mean(),
        "f1": mean_f1[best_index],
    })
    for class_index, class_name in enumerate(class_names):
        class_best_index = int(f1[class_index].argmax())
        rows.append({
            "scope": class_name,
            "setting": "best_class_f1",
            "confidence": confidence[class_best_index],
            "precision": precision[class_index, class_best_index],
            "recall": recall[class_index, class_best_index],
            "f1": f1[class_index, class_best_index],
        })
    for candidate in args.candidate_confidences:
        index = int(np.abs(confidence - candidate).argmin())
        rows.append({
            "scope": "all",
            "setting": "candidate",
            "confidence": confidence[index],
            "precision": precision[:, index].mean(),
            "recall": recall[:, index].mean(),
            "f1": f1[:, index].mean(),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=("scope", "setting", "confidence", "precision", "recall", "f1"),
        )
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['scope']:>5} | {row['setting']:<13} | "
            f"conf={row['confidence']:.3f} | P={row['precision']:.3f} | "
            f"R={row['recall']:.3f} | F1={row['f1']:.3f}"
        )
    print(f"inference speed: {metrics.speed['inference']:.1f} ms/image")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

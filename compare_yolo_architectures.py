"""Validate YOLOv8n and MobileNetV3-YOLOv8 under identical conditions."""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODELS = (
    (
        "YOLOv8n r010",
        ROOT / "runs/detect/runs/yolo_weedmap_compare/r010_5epochs/weights/best.pt",
        4.61,
    ),
    (
        "MobileNetV3-Small-YOLOv8 r010 (two-stage)",
        ROOT
        / "runs/yolo_mobilenetv3_compare/mobilenetv3_small_stage2_finetune_20epochs/weights/best.pt",
        2.55,
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/yolo_weedmap_detect_r010/data.yaml",
    )
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/yolo_architecture_comparison.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for name, model_path, expected_gflops in MODELS:
        model = YOLO(str(model_path))
        metrics = model.val(
            data=str(args.data),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            plots=False,
            verbose=False,
        )
        info = model.info(imgsz=args.imgsz)
        reported_gflops = info[3]
        rows.append({
            "model": name,
            "precision": metrics.box.mp,
            "recall": metrics.box.mr,
            "mAP50": metrics.box.map50,
            "mAP50_95": metrics.box.map,
            "params": sum(parameter.numel() for parameter in model.model.parameters()),
            # Ultralytics' profiler skips the raw grouped 1x1 input-normalization
            # layer in the custom graph and can consequently report zero FLOPs.
            "gflops": reported_gflops or expected_gflops,
            "model_size_mb": model_path.stat().st_size / 1_000_000,
            "preprocess_ms": metrics.speed["preprocess"],
            "inference_ms": metrics.speed["inference"],
            "postprocess_ms": metrics.speed["postprocess"],
            "model_path": str(model_path.relative_to(ROOT)),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['model']}: P={row['precision']:.4f}, R={row['recall']:.4f}, "
            f"mAP50={row['mAP50']:.4f}, mAP50-95={row['mAP50_95']:.4f}, "
            f"params={row['params']:,}, GFLOPs={row['gflops']:.2f}, "
            f"size={row['model_size_mb']:.2f} MB, "
            f"inference={row['inference_ms']:.1f} ms/image"
        )
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

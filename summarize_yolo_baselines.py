"""Summarize existing YOLOv8n runs without training or validation inference.

Detection metrics are the all-class values from the final row of results.csv.
Model statistics come from each run's best.pt checkpoint.
"""

import csv
import re
from pathlib import Path

import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "runs/detect/runs/yolo_weedmap_compare"
OUTPUT = ROOT / "outputs/yolo_baseline_summary.csv"
EXPERIMENTS = (
    ("r020_5epochs", "crop+weed", "crop,weed", 0.20),
    ("r010_5epochs", "crop+weed", "crop,weed", 0.10),
    ("weed_only_r010_5epochs", "weed-only", "weed", 0.10),
)
FIELDS = (
    "experiment", "dataset_type", "target_classes", "max_box_area_ratio",
    "epochs", "imgsz", "batch", "precision", "recall", "mAP50",
    "mAP50_95", "model_size_mb", "params", "gflops", "best_model_path",
    "metrics_epoch", "metrics_source",
)
METRIC_ALIASES = {
    "precision": ("metrics/precision(B)", "metrics/precision", "precision(B)", "precision", "P"),
    "recall": ("metrics/recall(B)", "metrics/recall", "recall(B)", "recall", "R"),
    "mAP50": ("metrics/mAP50(B)", "metrics/mAP50", "mAP50(B)", "mAP50", "map50"),
    "mAP50_95": (
        "metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95(B)",
        "mAP50-95", "mAP50_95", "map5095",
    ),
}


def normalized(name):
    """Ignore punctuation, spacing, and capitalization in CSV headings."""
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def read_metrics(path):
    if not path.is_file():
        print(f"WARNING: missing {path.relative_to(ROOT)}; metrics left blank")
        return {}, ""
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        headings = reader.fieldnames or []
        print(f"{path.relative_to(ROOT)} fields: {', '.join(headings)}")
        rows = [row for row in reader if any(value and value.strip() for value in row.values())]
    if not rows:
        print(f"WARNING: no data rows in {path.relative_to(ROOT)}")
        return {}, ""
    heading_map = {normalized(heading): heading for heading in headings}
    final_row = rows[-1]
    metrics = {}
    for metric, aliases in METRIC_ALIASES.items():
        heading = next((heading_map[normalized(alias)] for alias in aliases
                        if normalized(alias) in heading_map), None)
        if heading is None:
            print(f"WARNING: {path.relative_to(ROOT)} has no column for {metric}")
            metrics[metric] = ""
        else:
            try:
                metrics[metric] = float(final_row[heading])
            except (TypeError, ValueError):
                print(f"WARNING: invalid {metric} value in final row of {path.relative_to(ROOT)}")
                metrics[metric] = ""
    epoch_heading = heading_map.get("epoch")
    return metrics, final_row.get(epoch_heading, "") if epoch_heading else ""


def read_settings(path):
    if not path.is_file():
        print(f"WARNING: missing {path.relative_to(ROOT)}; settings left blank")
        return {}
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def read_model_stats(path, imgsz):
    if not path.is_file():
        print(f"WARNING: missing {path.relative_to(ROOT)}; model stats left blank")
        return "", "", ""
    size_mb = path.stat().st_size / 1_000_000
    try:
        model = YOLO(str(path))
        print(f"Model summary for {path.relative_to(ROOT)} (imgsz={imgsz}):")
        summary = model.info(imgsz=imgsz or 640)
        params = sum(parameter.numel() for parameter in model.model.parameters())
        gflops = summary[3] if isinstance(summary, (list, tuple)) and len(summary) > 3 else ""
        return size_mb, params, gflops
    except Exception as error:
        print(f"WARNING: could not inspect {path.relative_to(ROOT)}: {error}")
        return size_mb, "", ""


def format_cell(value, key):
    if value == "":
        return "—"
    if key in {"precision", "recall", "mAP50", "mAP50_95"}:
        return f"{value:.4f}"
    if key in {"model_size_mb", "gflops"}:
        return f"{value:.2f}"
    return str(value)


def print_table(rows):
    columns = ("experiment", "precision", "recall", "mAP50", "mAP50_95",
               "model_size_mb", "params", "gflops")
    labels = ("Experiment", "P", "R", "mAP50", "mAP50-95", "Size MB", "Params", "GFLOPs")
    formatted = [[format_cell(row[key], key) for key in columns] for row in rows]
    widths = [max(len(label), *(len(row[index]) for row in formatted))
              for index, label in enumerate(labels)]
    print("  ".join(label.ljust(width) for label, width in zip(labels, widths)))
    print("  ".join("-" * width for width in widths))
    for row in formatted:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))


def main():
    rows = []
    for experiment, dataset_type, target_classes, max_ratio in EXPERIMENTS:
        run_dir = RUN_ROOT / experiment
        settings = read_settings(run_dir / "args.yaml")
        metrics, metrics_epoch = read_metrics(run_dir / "results.csv")
        model_path = run_dir / "weights/best.pt"
        size_mb, params, gflops = read_model_stats(model_path, settings.get("imgsz"))
        rows.append({
            "experiment": experiment,
            "dataset_type": dataset_type,
            "target_classes": target_classes,
            "max_box_area_ratio": max_ratio,
            "epochs": settings.get("epochs", ""),
            "imgsz": settings.get("imgsz", ""),
            "batch": settings.get("batch", ""),
            **metrics,
            "model_size_mb": size_mb,
            "params": params,
            "gflops": gflops,
            "best_model_path": str(model_path.relative_to(ROOT)) if model_path.is_file() else "",
            "metrics_epoch": metrics_epoch,
            "metrics_source": "results.csv final row" if metrics_epoch or metrics else "",
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print("\nYOLO baseline summary (all-class final-epoch metrics; best.pt model statistics)")
    print_table(rows)
    print(f"\nSaved: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

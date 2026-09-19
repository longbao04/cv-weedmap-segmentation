"""Summarize best-checkpoint VCR router metrics across random seeds."""

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

import torch

from vcr_router_models import build_vcr_router


ROOT = Path(__file__).resolve().parent
MODELS = ("tiny", "se", "cbam")
METRICS = (
    "mae",
    "rmse",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "sparse_recall",
    "non_dense_recall",
    "non_dense_pr_auc",
    "mcc",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/vcr_router")
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument(
        "--runs-output",
        type=Path,
        default=ROOT / "outputs/vcr_router_run_results.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "outputs/vcr_router_model_summary.csv",
    )
    return parser.parse_args()


def load_rows(run_root, seeds):
    rows = []
    for model_name in MODELS:
        parameters = sum(
            parameter.numel()
            for parameter in build_vcr_router(model_name).parameters()
        )
        for seed in seeds:
            checkpoint_path = run_root / f"{model_name}_seed{seed}" / "best.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            metrics = checkpoint["metrics"]
            rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "parameters": parameters,
                    **{metric: metrics[metric] for metric in METRICS},
                    "confusion_matrix": str(metrics["confusion_matrix"]),
                    "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                }
            )
    return rows


def aggregate(rows):
    summary = []
    for model_name in MODELS:
        model_rows = [row for row in rows if row["model"] == model_name]
        result = {
            "model": model_name,
            "runs": len(model_rows),
            "parameters": model_rows[0]["parameters"],
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in model_rows]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary.append(result)
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows = load_rows(args.run_root, args.seeds)
    summary = aggregate(rows)
    write_csv(args.runs_output, rows)
    write_csv(args.summary_output, summary)
    for row in summary:
        print(
            f"{row['model']}: MAE={row['mae_mean']:.4f}±{row['mae_std']:.4f}, "
            f"RMSE={row['rmse_mean']:.4f}±{row['rmse_std']:.4f}, "
            f"balanced_acc={row['balanced_accuracy_mean']:.4f}±"
            f"{row['balanced_accuracy_std']:.4f}, "
            f"sparse_recall={row['sparse_recall_mean']:.4f}±"
            f"{row['sparse_recall_std']:.4f}, "
            f"PR-AUC={row['non_dense_pr_auc_mean']:.4f}±"
            f"{row['non_dense_pr_auc_std']:.4f}"
        )
    print(f"saved: {args.runs_output}")
    print(f"saved: {args.summary_output}")


if __name__ == "__main__":
    main()

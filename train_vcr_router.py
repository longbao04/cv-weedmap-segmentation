"""Train and evaluate a Tiny CNN VCR regression scene router."""

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from vcr_router_models import build_vcr_router


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "outputs/vcr_regression_samples.csv"
SCENES = ("sparse", "transition", "dense")


def default_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", choices=("tiny", "se", "cbam"), default="tiny")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--loss", choices=("huber", "mae"), default="huber")
    parser.add_argument(
        "--scene-weighting",
        choices=("none", "sqrt-inverse"),
        default="sqrt-inverse",
    )
    parser.add_argument("--spectral-jitter", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--val-subset", help="Optional subset-level holdout")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/vcr_router")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.height < 16 or args.width < 16:
        parser.error("epochs/batch-size must be positive and image size must be >= 16")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning-rate must be positive and weight-decay non-negative")
    if not 0 <= args.dropout < 1 or not 0 <= args.spectral_jitter <= 0.5:
        parser.error("dropout must be in [0,1); spectral-jitter must be in [0,0.5]")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_manifest(path, val_subset=None):
    with path.open(newline="", encoding="utf-8-sig") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    required = {
        "sample_id", "split", "subset", "frame_id", "G_path", "R_path",
        "RE_path", "NIR_path", "VCR", "scene_type",
    }
    if not rows:
        raise ValueError(f"Manifest contains no samples: {path}")
    if not required.issubset(rows[0]):
        raise ValueError("Manifest is missing required VCR router fields")

    for row in rows:
        row["VCR"] = float(row["VCR"])
        if val_subset:
            row["effective_split"] = (
                "val" if row["subset"] == val_subset else "train"
            )
        else:
            row["effective_split"] = row["split"]
    train_rows = [row for row in rows if row["effective_split"] == "train"]
    val_rows = [row for row in rows if row["effective_split"] == "val"]
    if not train_rows or not val_rows:
        raise ValueError("Both train and validation samples are required")
    return train_rows, val_rows


def load_band(path):
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected single-channel band: {path}")
    if np.issubdtype(array.dtype, np.integer):
        scale = np.iinfo(array.dtype).max
        array = array.astype(np.float32) / max(scale, 1)
    else:
        array = array.astype(np.float32)
        maximum = float(array.max()) if array.size else 0.0
        if maximum > 1.0:
            array /= maximum
    return np.clip(array, 0.0, 1.0)


class VCRRegressionDataset(Dataset):
    def __init__(self, rows, image_size, augment=False, spectral_jitter=0.0):
        self.rows = rows
        self.image_size = image_size
        self.augment = augment
        self.spectral_jitter = spectral_jitter

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        paths = [ROOT / row[f"{band}_path"] for band in ("G", "R", "RE", "NIR")]
        bands = [load_band(path) for path in paths]
        if len({band.shape for band in bands}) != 1:
            raise ValueError(f"Band shape mismatch for {row['sample_id']}")
        image = torch.from_numpy(np.stack(bands)).float()
        if self.augment:
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=(2,))
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=(1,))
            rotations = int(torch.randint(0, 4, ()).item())
            if rotations:
                image = torch.rot90(image, rotations, dims=(1, 2))
            if self.spectral_jitter:
                low = 1.0 - self.spectral_jitter
                high = 1.0 + self.spectral_jitter
                gains = torch.empty(4, 1, 1).uniform_(low, high)
                image = torch.clamp(image * gains, 0.0, 1.0)
        image = F.interpolate(
            image.unsqueeze(0), self.image_size, mode="bilinear", align_corners=False
        ).squeeze(0)
        return image, torch.tensor(row["VCR"], dtype=torch.float32), row["scene_type"], row["sample_id"]


def regression_loss(predictions, targets, loss_name):
    if loss_name == "huber":
        return F.huber_loss(predictions, targets, reduction="none", delta=0.1)
    return F.l1_loss(predictions, targets, reduction="none")


def build_scene_weights(rows, mode):
    if mode == "none":
        return {scene: 1.0 for scene in SCENES}
    counts = Counter(row["scene_type"] for row in rows)
    raw = {scene: 1.0 / math.sqrt(counts[scene]) for scene in SCENES}
    normalizer = sum(raw[row["scene_type"]] for row in rows) / len(rows)
    return {scene: value / normalizer for scene, value in raw.items()}


def scene_index(vcr):
    if vcr < 0.20:
        return 0
    if vcr > 0.30:
        return 2
    return 1


def average_precision(labels, scores):
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    precision = true_positives / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / positives)


def compute_metrics(targets, predictions):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    true_scenes = np.array([scene_index(value) for value in targets])
    predicted_scenes = np.array([scene_index(value) for value in predictions])
    confusion = np.zeros((3, 3), dtype=np.int64)
    for truth, predicted in zip(true_scenes, predicted_scenes):
        confusion[truth, predicted] += 1

    recalls = np.divide(
        np.diag(confusion), confusion.sum(axis=1),
        out=np.zeros(3, dtype=float), where=confusion.sum(axis=1) != 0,
    )
    precisions = np.divide(
        np.diag(confusion), confusion.sum(axis=0),
        out=np.zeros(3, dtype=float), where=confusion.sum(axis=0) != 0,
    )
    f1 = np.divide(
        2 * precisions * recalls, precisions + recalls,
        out=np.zeros(3, dtype=float), where=(precisions + recalls) != 0,
    )
    supported_scenes = confusion.sum(axis=1) > 0

    true_non_dense = (targets <= 0.30).astype(np.int64)
    predicted_non_dense = (predictions <= 0.30).astype(np.int64)
    tp = int(((true_non_dense == 1) & (predicted_non_dense == 1)).sum())
    tn = int(((true_non_dense == 0) & (predicted_non_dense == 0)).sum())
    fp = int(((true_non_dense == 0) & (predicted_non_dense == 1)).sum())
    fn = int(((true_non_dense == 1) & (predicted_non_dense == 0)).sum())
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return {
        "mae": float(np.mean(np.abs(predictions - targets))),
        "rmse": float(np.sqrt(np.mean((predictions - targets) ** 2))),
        "accuracy": float(np.mean(true_scenes == predicted_scenes)),
        "balanced_accuracy": float(recalls[supported_scenes].mean()),
        "macro_f1": float(f1[supported_scenes].mean()),
        "sparse_recall": float(recalls[0]) if supported_scenes[0] else float("nan"),
        "non_dense_recall": tp / (tp + fn) if tp + fn else float("nan"),
        "non_dense_pr_auc": average_precision(true_non_dense, -predictions),
        "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
        "confusion_matrix": confusion.tolist(),
    }


def run_epoch(model, loader, device, loss_name, scene_weights, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    targets_all = []
    predictions_all = []
    sample_ids = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets, scenes, ids in loader:
            images = images.to(device)
            targets = targets.to(device)
            predictions = model(images)
            losses = regression_loss(predictions, targets, loss_name)
            weights = torch.tensor(
                [scene_weights[scene] for scene in scenes],
                device=device,
                dtype=losses.dtype,
            )
            loss = (losses * weights).mean()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(targets)
            sample_count += len(targets)
            targets_all.extend(targets.detach().cpu().tolist())
            predictions_all.extend(predictions.detach().cpu().tolist())
            sample_ids.extend(ids)
    return total_loss / sample_count, compute_metrics(targets_all, predictions_all), list(
        zip(sample_ids, targets_all, predictions_all)
    )


def main():
    args = parse_args()
    seed_everything(args.seed)
    train_rows, val_rows = load_manifest(args.manifest, args.val_subset)
    scene_weights = build_scene_weights(train_rows, args.scene_weighting)
    train_dataset = VCRRegressionDataset(
        train_rows, (args.height, args.width), True, args.spectral_jitter
    )
    val_dataset = VCRRegressionDataset(val_rows, (args.height, args.width))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
    )

    device = torch.device(args.device)
    model = build_vcr_router(args.model, args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    run_name = f"{args.model}_seed{args.seed}"
    if args.val_subset:
        run_name += f"_holdout_{args.val_subset}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _ = run_epoch(
            model, train_loader, device, args.loss, scene_weights, optimizer
        )
        val_loss, val_metrics, val_predictions = run_epoch(
            model, val_loader, device, args.loss,
            {scene: 1.0 for scene in SCENES},
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_metrics["mae"],
                "val_loss": val_loss,
                **{f"val_{key}": value for key, value in val_metrics.items() if key != "confusion_matrix"},
            }
        )
        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            torch.save(
                {
                    "model_name": args.model,
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "metrics": val_metrics,
                },
                run_dir / "best.pt",
            )
            with (run_dir / "best_predictions.csv").open(
                "w", newline="", encoding="utf-8"
            ) as predictions_file:
                writer = csv.writer(predictions_file)
                writer.writerow(("sample_id", "target_vcr", "predicted_vcr"))
                writer.writerows(val_predictions)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_mae={val_metrics['mae']:.4f} val_rmse={val_metrics['rmse']:.4f} "
            f"balanced_acc={val_metrics['balanced_accuracy']:.4f} "
            f"sparse_recall={val_metrics['sparse_recall']:.4f}"
        )

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(
            {
                "model": args.model,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
                "scene_weights": scene_weights,
                "best_val_mae": best_mae,
                "best_checkpoint": str(run_dir / "best.pt"),
            },
            summary_file,
            indent=2,
        )
    print(f"saved: {run_dir}")


if __name__ == "__main__":
    main()

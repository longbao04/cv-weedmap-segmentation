"""Train the existing U-Net on local, real WeedMap tiles."""

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from unet import SmallUNet
from weedmap_dataset import WeedMapDataset


NUM_CLASSES = 3
IGNORE_INDEX = 255
CLASS_NAMES = ("background", "crop", "weed")
CLASS_WEIGHTS = (1.0, 4.0, 8.0)
LOSS_CHOICES = ("ce", "weighted_ce")
INPUT_CHOICES = ("rgb", "multispectral")
PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/weedmap")
    parser.add_argument(
        "--input-type", choices=INPUT_CHOICES, default="multispectral"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--loss", choices=LOSS_CHOICES, default="weighted_ce")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help=(
            "模型保存路径；默认 models/real_weedmap_<input_type>_<loss>.pth"
        ),
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs 必须大于 0")
    if args.batch_size < 1:
        parser.error("--batch-size 必须大于 0")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr 必须是有限正数")
    if args.num_workers < 0:
        parser.error("--num-workers 不能小于 0")
    return args


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """Seed NumPy and Python RNGs used inside DataLoader workers."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def update_confusion_matrix(matrix, predictions, labels):
    """Accumulate a confusion matrix after removing ignore_index pixels."""
    valid = labels != IGNORE_INDEX
    if not torch.any(valid):
        return False

    valid_labels = labels[valid].to(torch.int64)
    valid_predictions = predictions[valid].to(torch.int64)
    if torch.any((valid_labels < 0) | (valid_labels >= NUM_CLASSES)):
        values = torch.unique(valid_labels).cpu().tolist()
        raise ValueError(
            f"标签应只包含 0、1、2、255，发现有效标签值：{values}"
        )

    indices = valid_labels * NUM_CLASSES + valid_predictions
    matrix += torch.bincount(
        indices.cpu(), minlength=NUM_CLASSES * NUM_CLASSES
    ).reshape(NUM_CLASSES, NUM_CLASSES)
    return True


def metrics_from_confusion_matrix(matrix):
    matrix = matrix.to(torch.float64)
    total = matrix.sum()
    pixel_accuracy = (matrix.diag().sum() / total).item() if total > 0 else math.nan

    intersections = matrix.diag()
    unions = matrix.sum(dim=1) + matrix.sum(dim=0) - intersections
    ious = torch.full((NUM_CLASSES,), math.nan, dtype=torch.float64)
    present = unions > 0
    ious[present] = intersections[present] / unions[present]
    mean_iou = ious[present].mean().item() if torch.any(present) else math.nan
    return pixel_accuracy, mean_iou, ious.tolist()


def evaluate_real_data(model, data_loader, device):
    """Evaluate real WeedMap data while completely ignoring label 255."""
    model.eval()
    matrix = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            # An all-ignore batch contributes nothing and is deliberately skipped.
            update_confusion_matrix(matrix, predictions, labels)
    return metrics_from_confusion_matrix(matrix)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = select_device()

    dataset = WeedMapDataset(
        data_root=args.data_root,
        input_type=args.input_type,
        filter_empty=True,
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    if train_size == 0 or val_size == 0:
        raise ValueError(
            "过滤后的 WeedMapDataset 至少需要 2 个样本，"
            f"当前只有 {len(dataset)} 个。请检查 --data-root。"
        )

    split_generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=split_generator
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
    )

    in_channels = 3 if args.input_type == "rgb" else 5
    model = SmallUNet(in_channels=in_channels, num_classes=NUM_CLASSES).to(device)
    if args.loss == "weighted_ce":
        class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=IGNORE_INDEX
        )
    else:
        class_weights = None
        criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_dir = PROJECT_ROOT / "models"
    output_dir = PROJECT_ROOT / "outputs"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.save_path or (
        model_dir / f"real_weedmap_{args.input_type}_{args.loss}.pth"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = (
        output_dir / f"real_weedmap_history_{args.input_type}_{args.loss}.csv"
    )
    fieldnames = (
        "epoch",
        "train_loss",
        "pixel_accuracy",
        "mean_iou",
        "background_iou",
        "crop_iou",
        "weed_iou",
    )
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        csv.DictWriter(history_file, fieldnames=fieldnames).writeheader()

    print(f"device: {device}")
    print(f"input_type: {args.input_type}")
    print(f"train samples: {train_size}")
    print(f"val samples: {val_size}")
    print(f"loss type: {args.loss}")
    print(f"ignore_index={IGNORE_INDEX}")
    if class_weights is not None:
        print("class weights: background=1.0, crop=4.0, weed=8.0")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        trained_samples = 0
        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)
            # CrossEntropyLoss has no valid denominator for an all-ignore batch.
            if not torch.any(label != IGNORE_INDEX):
                continue

            optimizer.zero_grad()
            logits = model(image)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * image.size(0)
            trained_samples += image.size(0)

        if trained_samples == 0:
            raise RuntimeError("训练集没有可参与 loss 计算的有效像素。")
        train_loss = loss_sum / trained_samples
        pixel_accuracy, mean_iou, class_ious = evaluate_real_data(
            model, val_loader, device
        )
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "pixel_accuracy": pixel_accuracy,
            "mean_iou": mean_iou,
            "background_iou": class_ious[0],
            "crop_iou": class_ious[1],
            "weed_iou": class_ious[2],
        }
        with history_path.open("a", newline="", encoding="utf-8") as history_file:
            csv.DictWriter(history_file, fieldnames=fieldnames).writerow(history_row)

        class_details = " | ".join(
            f"{name} IoU: {iou:.4f}"
            for name, iou in zip(CLASS_NAMES, class_ious)
        )
        print(
            f"Epoch {epoch}/{args.epochs} | train loss: {train_loss:.4f} | "
            f"val pixel accuracy: {pixel_accuracy:.4f} | "
            f"val mean IoU: {mean_iou:.4f} | {class_details}"
        )

    # CPU tensors make the state dict portable across MPS, CUDA, and CPU hosts.
    torch.save(
        {name: value.detach().cpu() for name, value in model.state_dict().items()},
        model_path,
    )
    print(f"模型已保存到 {model_path}")
    print(f"训练 history 已保存到 {history_path}")


if __name__ == "__main__":
    main()

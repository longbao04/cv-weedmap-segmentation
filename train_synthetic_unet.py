"""仅使用模拟数据训练小型 U-Net，不下载数据。"""

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from losses import DiceLoss, FocalLoss
from metrics import CLASS_NAMES, confusion_matrix, mean_iou, per_class_iou, pixel_accuracy
from synthetic_dataset import SyntheticWeedDataset
from unet import SmallUNet

MODEL_PATH = Path(__file__).resolve().parent / "models" / "synthetic_unet_rgb_ce.pth"
LOSS_CHOICES = ("ce", "weighted_ce", "dice", "focal")


def get_model_path(loss, input_type="rgb"):
    # 每种输入类型和损失单独保存模型，便于比较实验。
    return MODEL_PATH.with_name(f"synthetic_unet_{input_type}_{loss}.pth")


def select_device():
    # Apple Silicon 优先使用 MPS；其他环境使用 CPU。
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--loss", choices=LOSS_CHOICES, default="ce",
                        help="损失函数，默认 ce")
    parser.add_argument("--input-type", choices=("rgb", "multispectral"), default="rgb",
                        help="输入类型：RGB 三通道或模拟多光谱六通道，默认 rgb")
    parser.add_argument("--use-class-weights", action="store_true",
                        help="使用类别权重：background=1.0、crop=2.0、weed=6.0")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.lr < float("inf"):
        parser.error("epochs、batch-size 和 lr 必须为正数，lr 必须有限")

    torch.manual_seed(42)
    device = select_device()
    # 独立随机种子使训练与测试样本不同；测试集只用于评估。
    train_loader = DataLoader(SyntheticWeedDataset(256, seed=42, input_type=args.input_type),
                              batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(SyntheticWeedDataset(64, seed=2026, input_type=args.input_type),
                             batch_size=args.batch_size)
    # 根据输入类型调整第一层通道数，其他网络结构和输出类别保持一致。
    in_channels = 3 if args.input_type == "rgb" else 6
    model = SmallUNet(in_channels=in_channels).to(device)
    # 旧开关优先，等价于 --loss weighted_ce，保持已有命令兼容。
    loss_name = "weighted_ce" if args.use_class_weights else args.loss
    if loss_name == "weighted_ce":
        # 顺序为 background/crop/weed；权重和模型必须位于同一设备。
        class_weights = torch.tensor([1.0, 2.0, 6.0], device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == "dice":
        criterion = DiceLoss(num_classes=3, ignore_index=None)
    elif loss_name == "focal":
        criterion = FocalLoss(num_classes=3, ignore_index=None)
    else:
        criterion = nn.CrossEntropyLoss()
    model_path = get_model_path(loss_name, args.input_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 按输入类型和损失分别保存；相同组合重新训练时覆盖旧记录。
    output_dir = MODEL_PATH.parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"synthetic_history_{args.input_type}_{loss_name}.csv"
    fieldnames = ["epoch", "average_train_loss", "pixel_accuracy", "mean_iou",
                  "background_iou", "crop_iou", "weed_iou"]
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        csv.DictWriter(history_file, fieldnames=fieldnames).writeheader()
    print(f"Device: {device}; train samples: 256; test samples: 64")
    print(f"当前 input_type：{args.input_type}；输入通道数：{in_channels}")
    print(f"当前 loss 类型：{loss_name}")
    print(f"是否使用 class weights：{'是（background=1.0、crop=2.0、weed=6.0）' if loss_name == 'weighted_ce' else '否'}")

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            # 按样本数量加权，最后一个不足 batch-size 的批次也能正确平均。
            loss_sum += loss.item() * images.size(0)

        model.eval()
        matrix = torch.zeros((3, 3), dtype=torch.long)
        with torch.no_grad():
            for images, masks in test_loader:
                # 通道维 argmax 将类别分数转换为 [B,H,W] 的类别编号。
                predicted = model(images.to(device)).argmax(dim=1).cpu()
                matrix += confusion_matrix(predicted, masks)
        ious = per_class_iou(matrix=matrix)
        # 保存原始指标数值，accuracy 和 IoU 使用 0～1，而不是百分数。
        history_row = {
            "epoch": epoch + 1,
            "average_train_loss": loss_sum / len(train_loader.dataset),
            "pixel_accuracy": pixel_accuracy(matrix=matrix),
            "mean_iou": mean_iou(matrix=matrix),
            "background_iou": ious[0].item(),
            "crop_iou": ious[1].item(),
            "weed_iou": ious[2].item(),
        }
        # 每个 epoch 结束后追加一行并关闭文件，及时保存已完成的记录。
        with history_path.open("a", newline="", encoding="utf-8") as history_file:
            csv.DictWriter(history_file, fieldnames=fieldnames).writerow(history_row)
        details = ", ".join(f"{name} IoU: {value.item():.4f}"
                            for name, value in zip(CLASS_NAMES, ious))
        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"average train loss: {history_row['average_train_loss']:.4f} | "
              f"pixel accuracy: {history_row['pixel_accuracy']:.4f} | "
              f"mean IoU: {history_row['mean_iou']:.4f} | {details}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    # 保存 CPU 权重，方便在不同设备上加载。
    torch.save({key: value.cpu() for key, value in model.state_dict().items()}, model_path)
    print(f"模型已保存到 {model_path}")
    print(f"训练过程 CSV 已保存到 {history_path}")


if __name__ == "__main__":
    main()

"""读取模拟 U-Net 训练记录，绘制 loss、准确率和各类 IoU 曲线。"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-class-weights", action="store_true",
                        help="读取使用类别权重训练的 history")
    args = parser.parse_args()
    mode = "weighted" if args.use_class_weights else "baseline"
    # 相对于脚本定位文件，从其他目录运行时也能找到项目 outputs。
    output_dir = Path(__file__).resolve().parent / "outputs"
    history_path = output_dir / f"synthetic_history_{mode}.csv"
    if not history_path.is_file():
        train_command = "python train_synthetic_unet.py --epochs 5"
        if args.use_class_weights:
            train_command += " --use-class-weights"
        print(f"CSV 文件不存在：{history_path}\n请先运行：{train_command}")
        return

    metrics = ["average_train_loss", "pixel_accuracy", "mean_iou",
               "background_iou", "crop_iou", "weed_iou"]
    # 使用标准库读取 CSV，不需要额外安装 pandas。
    with history_path.open(newline="", encoding="utf-8") as history_file:
        reader = csv.DictReader(history_file)
        required_columns = ["epoch", *metrics]
        if not set(required_columns).issubset(reader.fieldnames or []):
            parser.error(f"CSV 缺少必要列：{', '.join(required_columns)}")
        try:
            rows = [{column: float(row[column]) for column in required_columns}
                    for row in reader]
        except (ValueError, TypeError):
            parser.error("CSV 中的 epoch 和指标必须为数值")
    if not rows:
        print(f"CSV 尚无 epoch 记录：{history_path}\n请等待训练完成至少一个 epoch。")
        return

    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Synthetic U-Net training history ({mode})")
    # 六个指标各占一个子图，避免 loss 与 IoU 的数值范围互相影响。
    for ax, metric in zip(axes.flat, metrics):
        values = [row[metric] for row in rows]
        # 使用橙色强调 weed IoU，方便关注少数类的训练变化。
        color = "tab:orange" if metric == "weed_iou" else "tab:blue"
        ax.plot(epochs, values, marker="o", color=color)
        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss" if metric == "average_train_loss" else "Score (0-1)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if metric != "average_train_loss":
            ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    # 同时保存图片，便于整理实验记录或在无图形界面环境中查看。
    figure_path = output_dir / f"synthetic_history_{mode}.png"
    fig.savefig(figure_path, dpi=150)
    print(f"训练曲线已保存到 {figure_path}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Plot seed=0 validation IoU curves for three real WeedMap losses."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent
HISTORIES = {
    "Weighted CE": ROOT / "outputs/real_weedmap_history_common_multispectral_weighted_ce_20epochs.csv",
    "Focal Loss": ROOT / "outputs/real_weedmap_history_common_multispectral_focal_20epochs_seed0.csv",
    "Dice + CE": ROOT / "outputs/real_weedmap_history_common_multispectral_dice_ce_20epochs_seed0.csv",
}
OUTPUTS = (
    ROOT / "outputs/real_weedmap_loss_comparison_seed0.png",
    ROOT / "reports/assets/real_weedmap_loss_comparison_seed0.png",
)
REQUIRED_COLUMNS = ("epoch", "mean_iou", "weed_iou")


def read_history(path):
    if not path.is_file():
        raise FileNotFoundError(f"缺少 history CSV 文件：{path}")

    with path.open(newline="", encoding="utf-8-sig") as history_file:
        reader = csv.DictReader(history_file)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV 缺少必要列 {', '.join(sorted(missing))}：{path}")
        try:
            rows = [tuple(float(row[column]) for column in REQUIRED_COLUMNS) for row in reader]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CSV 的 epoch 和 IoU 必须是数值：{path}") from exc

    if not rows:
        raise ValueError(f"CSV 没有 epoch 记录：{path}")
    return rows


def main():
    histories = {name: read_history(path) for name, path in HISTORIES.items()}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    fig.suptitle("Real WeedMap validation IoU by loss (multispectral, seed=0)")
    for ax, (metric, title) in zip(
        axes, ((1, "Validation mean IoU"), (2, "Validation weed IoU"))
    ):
        for name, rows in histories.items():
            ax.plot([row[0] for row in rows], [row[metric] for row in rows],
                    marker="o", markersize=3, label=name)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("IoU")
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    for path in OUTPUTS:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        print(f"对比曲线已保存到 {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

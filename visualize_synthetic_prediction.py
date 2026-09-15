"""比较新的模拟图像、真实 mask、预测 mask 与错误位置。"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import torch

from metrics import CLASS_NAMES
from synthetic_dataset import SyntheticWeedDataset
from train_synthetic_unet import MODEL_PATH, select_device
from unet import SmallUNet


def main():
    if not MODEL_PATH.is_file():
        print("模型文件不存在，请先运行：python train_synthetic_unet.py")
        return
    device = select_device()
    model = SmallUNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    # 使用不同于训练和评估的种子，生成三张新的模拟样本。
    dataset = SyntheticWeedDataset(3, seed=9999)
    colors = ["#765233", "#298533", "#57b329"]
    cmap = ListedColormap(colors)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), squeeze=False)
    with torch.no_grad():
        for row in range(len(dataset)):
            image, mask = dataset[row]
            predicted = model(image.unsqueeze(0).to(device)).argmax(1)[0].cpu()
            # 错误图中黑色表示正确，红色表示预测类别与真值不同。
            error = (predicted != mask).numpy()
            axes[row, 0].imshow(image.permute(1, 2, 0).numpy())
            axes[row, 1].imshow(mask.numpy(), cmap=cmap, vmin=0, vmax=2,
                                interpolation="nearest")
            axes[row, 2].imshow(predicted.numpy(), cmap=cmap, vmin=0, vmax=2,
                                interpolation="nearest")
            axes[row, 3].imshow(error, cmap=ListedColormap(["black", "red"]),
                                vmin=0, vmax=1, interpolation="nearest")
            for col, title in enumerate(["RGB image", "True mask", "Predicted mask",
                                         "Error map (red = wrong)"]):
                axes[row, col].set_title(title)
                axes[row, col].axis("off")
    fig.legend(handles=[Patch(color=color, label=name)
                        for color, name in zip(colors, CLASS_NAMES)],
               loc="lower center", ncol=3)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    # 同时保存图片，便于无图形界面的环境查看结果。
    output_dir = MODEL_PATH.parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_prediction.png"
    fig.savefig(output_path, dpi=150)
    print(f"预测可视化已保存到 {output_path}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

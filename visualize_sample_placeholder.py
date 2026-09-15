"""用假数据演示 image / mask / overlay，不读取真实 WeedMap 数据。"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


def main():
    # mask 是二维整数数组，每个像素保存一个类别编号。
    # 这里约定 0=background、1=crop、2=weed，不代表真实数据编码。
    height, width = 128, 128
    mask = np.zeros((height, width), dtype=np.uint8)
    yy, xx = np.indices((height, width))
    for center_y in (30, 64, 98):
        for center_x in (35, 85):
            crop_area = ((xx - center_x) / 11) ** 2 + ((yy - center_y) / 8) ** 2 <= 1
            mask[crop_area] = 1
    for center_y, center_x, radius in ((15, 63, 6), (49, 112, 8), (80, 16, 7), (112, 62, 6)):
        weed_area = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2
        mask[weed_area] = 2

    # RGB image 是三维数组：高度 × 宽度 × 3，每个像素保存红、绿、蓝。
    # 通过类别生成棕色土壤和绿色植被，再加入少量可复现的纹理噪声。
    rng = np.random.default_rng(42)
    image_colors = np.array([[0.48, 0.32, 0.19], [0.18, 0.58, 0.22], [0.36, 0.65, 0.16]])
    rgb = np.clip(image_colors[mask] + rng.normal(0, 0.035, (height, width, 3)), 0, 1)

    # 标签颜色是人为指定的显示方式，不是原始影像颜色。
    label_colors = np.array([[0.55, 0.40, 0.26], [0.10, 0.75, 0.25], [0.95, 0.25, 0.20]])
    cmap = ListedColormap(label_colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    # overlay 用透明度混合 RGB 和标签颜色，方便观察标签所在的位置。
    alpha = 0.45
    overlay = (1 - alpha) * rgb + alpha * label_colors[mask]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Synthetic RGB image")
    axes[1].imshow(mask, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title("Segmentation mask")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (alpha=0.45)")
    for axis in axes:
        axis.axis("off")
    # 图中文字用英文，避免本地环境缺少中文字体导致显示异常。
    labels = ("0: background", "1: crop", "2: weed")
    fig.legend(handles=[Patch(facecolor=color, label=label)
                        for color, label in zip(label_colors, labels)],
               loc="lower center", ncol=3)
    fig.suptitle("Placeholder only - not real WeedMap data")
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    plt.show()


if __name__ == "__main__":
    main()

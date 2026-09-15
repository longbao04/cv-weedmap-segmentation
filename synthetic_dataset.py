"""用 NumPy 生成农田 RGB 图像和逐像素类别标签，不读取真实数据。"""

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticWeedDataset(Dataset):
    """同一个 seed 和 index 对应同一个样本，方便复现实验。"""

    def __init__(self, num_samples=256, seed=42):
        if num_samples < 1:
            raise ValueError("num_samples 必须大于 0")
        self.num_samples = num_samples
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        height = width = 128
        yy, xx = np.mgrid[:height, :width]
        mask = np.zeros((height, width), dtype=np.int64)

        # 作物沿田垄规则排列；位置和大小略有变化，避免每张图完全相同。
        for cy in range(int(rng.integers(10, 20)), height, 24):
            for cx in range(int(rng.integers(10, 20)), width, 28):
                x = cx + int(rng.integers(-2, 3))
                y = cy + int(rng.integers(-2, 3))
                rx, ry = rng.uniform(6, 9), rng.uniform(7, 10)
                region = ((xx - x) / rx) ** 2 + ((yy - y) / ry) ** 2 <= 1
                mask[region] = 1

        # 杂草是随机小块；覆盖到作物时，将可见的该区域标记为杂草。
        for _ in range(int(rng.integers(18, 36))):
            x, y = rng.integers(0, width), rng.integers(0, height)
            rx, ry = rng.uniform(2, 5), rng.uniform(2, 5)
            region = ((xx - x) / rx) ** 2 + ((yy - y) / ry) ** 2 <= 1
            mask[region] = 2

        # 土壤为棕色，作物为深绿色，杂草为较亮的绿色。
        palette = np.array([[0.46, 0.32, 0.20], [0.16, 0.52, 0.20],
                            [0.34, 0.70, 0.16]], dtype=np.float32)
        rgb = palette[mask]
        # 整体光照变化加上逐像素噪声，模拟简单的拍摄差异和纹理。
        rgb = rgb * rng.uniform(0.85, 1.15)
        rgb += rng.normal(0, 0.045, size=rgb.shape)
        rgb = np.clip(rgb, 0, 1).astype(np.float32)
        # NumPy 的 HWC 转为 PyTorch 的 CHW；标签不做归一化。
        image_tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
        mask_tensor = torch.from_numpy(mask).long()
        return image_tensor, mask_tensor

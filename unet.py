"""适用于 128×128 RGB 或多光谱图像的小型 U-Net。"""

import torch
from torch import nn


def double_conv(in_channels, out_channels):
    # padding=1 保持空间尺寸不变，两次卷积提取局部特征。
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        # Encoder：降低分辨率、增加通道，学习更大范围的图像信息。
        # RGB 输入为 3 通道，多光谱输入为 6 通道；输出仍为三个类别。
        self.enc1 = double_conv(in_channels, 16)
        self.enc2 = double_conv(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(32, 64)
        # Decoder：逐步上采样，恢复逐像素预测所需的空间分辨率。
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = double_conv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = double_conv(32, 16)
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        center = self.bottleneck(self.pool(e2))
        # Skip connection：拼接 encoder 特征，补回下采样丢失的细节。
        d2 = self.dec2(torch.cat([self.up2(center), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        # 返回 [B, 3, H, W] 原始分数；CrossEntropyLoss 内部处理 softmax。
        return self.head(d1)

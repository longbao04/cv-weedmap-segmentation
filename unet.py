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


class InvertedResidual(nn.Module):
    """MobileNetV2-style pointwise-depthwise-pointwise block."""

    def __init__(self, in_channels, out_channels, stride, expansion):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        if expansion not in (1, 6):
            raise ValueError("expansion must be 1 or 6")

        hidden_channels = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(
                hidden_channels, hidden_channels, 3,
                stride=stride, padding=1, groups=hidden_channels, bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        result = self.layers(x)
        return x + result if self.use_residual else result


class MobileNetV2ShallowUNet(nn.Module):
    """Keep C1/C2 and the SmallUNet decoder; use shallow B1-B6 blocks."""

    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.bridge = nn.Conv2d(16, 32, 1)
        self.b1 = InvertedResidual(32, 16, stride=1, expansion=1)
        self.b2 = InvertedResidual(16, 24, stride=2, expansion=6)
        self.b3 = InvertedResidual(24, 24, stride=1, expansion=6)
        self.e2_adapter = nn.Conv2d(24, 32, 1)
        self.b4 = InvertedResidual(24, 32, stride=2, expansion=6)
        self.b5 = InvertedResidual(32, 32, stride=1, expansion=6)
        self.b6 = InvertedResidual(32, 32, stride=1, expansion=6)
        self.center_adapter = nn.Conv2d(32, 64, 1)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = double_conv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = double_conv(32, 16)
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        e1 = self.c2(self.c1(x))
        b3 = self.b3(self.b2(self.b1(self.bridge(e1))))
        e2 = self.e2_adapter(b3)
        center = self.center_adapter(self.b6(self.b5(self.b4(b3))))
        d2 = self.dec2(torch.cat([self.up2(center), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)

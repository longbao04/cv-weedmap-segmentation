"""Tiny CNN regressors for the VCR-based scene density router."""

import torch
from torch import nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        return inputs * self.excitation(self.pool(inputs))


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        self.channel_gate = nn.Sigmoid()
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        average = self.channel_mlp(torch.mean(inputs, dim=(2, 3), keepdim=True))
        maximum = self.channel_mlp(torch.amax(inputs, dim=(2, 3), keepdim=True))
        outputs = inputs * self.channel_gate(average + maximum)
        spatial = torch.cat(
            (
                torch.mean(outputs, dim=1, keepdim=True),
                torch.amax(outputs, dim=1, keepdim=True),
            ),
            dim=1,
        )
        return outputs * self.spatial_gate(spatial)


class ConvStage(nn.Module):
    def __init__(self, in_channels, out_channels, attention="none"):
        super().__init__()
        if attention == "none":
            attention_layer = nn.Identity()
        elif attention == "se":
            attention_layer = SEBlock(out_channels)
        elif attention == "cbam":
            attention_layer = CBAMBlock(out_channels)
        else:
            raise ValueError(f"Unsupported attention type: {attention}")
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            attention_layer,
            nn.MaxPool2d(2),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class VCRTinyCNN(nn.Module):
    """Predict a continuous VCR in [0, 1] from G/R/RE/NIR inputs."""

    def __init__(self, attention="none", dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            ConvStage(4, 16, attention),
            ConvStage(16, 32, attention),
            ConvStage(32, 64, attention),
        )
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        return self.regressor(self.features(inputs)).squeeze(1)


def build_vcr_router(model_name, dropout=0.2):
    attention_by_model = {"tiny": "none", "se": "se", "cbam": "cbam"}
    if model_name not in attention_by_model:
        raise ValueError(
            f"Unknown model {model_name!r}; choose from {sorted(attention_by_model)}"
        )
    return VCRTinyCNN(attention=attention_by_model[model_name], dropout=dropout)


if __name__ == "__main__":
    example = torch.rand(2, 4, 180, 240)
    for name in ("tiny", "se", "cbam"):
        model = build_vcr_router(name)
        output = model(example)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"{name}: output={tuple(output.shape)}, params={parameters:,}")

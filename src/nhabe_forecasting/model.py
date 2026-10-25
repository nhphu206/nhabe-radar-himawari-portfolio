"""Small 2D U-Net used for controlled radar forecasting experiments."""

from __future__ import annotations

import torch
from torch import nn


MODEL_CONFIG = {
    "schema_version": "nhabe-small-unet-v1",
    "architecture": "small_unet_2d",
    "base_channels": 16,
    "encoder_depth": 3,
    "convolutions_per_block": 2,
    "kernel_size": 3,
    "activation": "ReLU",
    "normalization": "GroupNorm",
    "group_norm_groups": 8,
    "downsampling": "MaxPool2d",
    "upsampling": "ConvTranspose2d",
    "output_channels": 1,
    "output_activation": "none",
    "spatial_shape": [256, 256],
    "input_channels": {"radar": 2, "multimodal": 8},
}


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with GroupNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8) -> None:
        super().__init__()
        if out_channels % groups != 0:
            raise ValueError("GroupNorm groups must divide every block width")
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UpBlock(nn.Module):
    """Upsample, concatenate the same-scale encoder feature, then refine."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = self.up(inputs)
        if inputs.shape[-2:] != skip.shape[-2:]:
            raise ValueError("Encoder and decoder spatial shapes do not match")
        return self.conv(torch.cat((skip, inputs), dim=1))


class SmallUNet2D(nn.Module):
    """Three-level U-Net with one unrestricted regression output channel."""

    def __init__(self, in_channels: int, base_channels: int = 16, out_channels: int = 1) -> None:
        super().__init__()
        if in_channels < 1 or base_channels < 8 or out_channels < 1:
            raise ValueError("Invalid U-Net channel configuration")
        if any(width % 8 for width in (base_channels, base_channels * 2, base_channels * 4,
                                       base_channels * 8)):
            raise ValueError("All U-Net block widths must be divisible by 8")
        self.in_channels = in_channels
        self.out_channels = out_channels
        widths = [base_channels * (2 ** level) for level in range(4)]

        self.encoder1 = DoubleConv(in_channels, widths[0])
        self.encoder2 = DoubleConv(widths[0], widths[1])
        self.encoder3 = DoubleConv(widths[1], widths[2])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(widths[2], widths[3])
        self.decoder3 = UpBlock(widths[3], widths[2], widths[2])
        self.decoder2 = UpBlock(widths[2], widths[1], widths[1])
        self.decoder1 = UpBlock(widths[1], widths[0], widths[0])
        self.output = nn.Conv2d(widths[0], out_channels, kernel_size=1)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(f"Expected BCHW input with {self.in_channels} channels")
        if inputs.shape[-2] % 8 or inputs.shape[-1] % 8:
            raise ValueError("Input height and width must be divisible by 8")
        skip1 = self.encoder1(inputs)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        encoded = self.bottleneck(self.pool(skip3))
        decoded = self.decoder3(encoded, skip3)
        decoded = self.decoder2(decoded, skip2)
        decoded = self.decoder1(decoded, skip1)
        return self.output(decoded)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

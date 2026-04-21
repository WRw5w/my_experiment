"""
U-Net Architecture for Noise2Void Image Denoising
===================================================
Standard U-Net with residual learning (predicts noise residual).
Supports both grayscale (1-channel) and color (3-channel) images.

Architecture:
  Encoder: 4 down-sampling blocks (Conv-BN-ReLU × 2 + MaxPool)
  Bottleneck: Conv-BN-ReLU × 2
  Decoder: 4 up-sampling blocks (ConvTranspose + skip concat + Conv-BN-ReLU × 2)
  Output:  1×1 Conv → residual noise estimate

Channel progression: 64 → 128 → 256 → 512 → 1024 → 512 → 256 → 128 → 64

Note: The blind-spot masking strategy for Noise2Void is handled at the
data/training level, not in the network architecture itself. The U-Net
architecture remains identical to exp7/exp8.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive (Conv3×3 → BatchNorm → ReLU) blocks, optionally with Dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ])
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Down-sampling: MaxPool2×2 followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Up-sampling: ConvTranspose2d → concatenate skip → DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                     kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                                   diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for image denoising with residual learning and optional Dropout.
    """

    def __init__(self, in_channels: int = 1, base_features: int = 64, dropout: float = 0.0):
        super().__init__()
        f = base_features

        self.enc1 = DoubleConv(in_channels, f, dropout)
        self.enc2 = Down(f, f * 2, dropout)
        self.enc3 = Down(f * 2, f * 4, dropout)
        self.enc4 = Down(f * 4, f * 8, dropout)

        self.bottleneck = Down(f * 8, f * 16, dropout)

        self.dec4 = Up(f * 16, f * 8, dropout)
        self.dec3 = Up(f * 8, f * 4, dropout)
        self.dec2 = Up(f * 4, f * 2, dropout)
        self.dec1 = Up(f * 2, f, dropout)

        self.out_conv = nn.Conv2d(f, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        b = self.bottleneck(s4)

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        noise = self.out_conv(d1)
        return x - noise


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test: verify model builds and runs
    for ch in [1, 3]:
        model = UNet(in_channels=ch)
        x = torch.randn(1, ch, 128, 128)
        y = model(x)
        print(f"[{'Gray' if ch == 1 else 'Color'}] Input: {x.shape} → Output: {y.shape}, "
              f"Params: {count_parameters(model):,}")

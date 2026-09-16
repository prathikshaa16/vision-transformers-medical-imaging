"""
ConvStem: Lightweight Convolutional Stem for E2H-ViT
Preserves localized boundary textures, edges, and micro-lesions with strong inductive bias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    Lightweight Depthwise Separable Convolution block with residual skip connection.
    Reduces parameter and FLOP footprint relative to standard convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

        # Shortcut projection if dimensions or spatial resolution change
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.norm(out)
        out = self.act(out + res)
        return out


class ConvStem(nn.Module):
    """
    3-Stage Convolutional Stem with inductive bias for medical image analysis.
    Downsamples image from (B, C_in, H, W) to (B, out_channels, H/8, W/8)
    while capturing fine-grained lesion margins and tissue borders.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 96):
        super().__init__()
        c1 = max(16, out_channels // 4)
        c2 = max(32, out_channels // 2)

        # Stage 1: 224x224 -> 112x112
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
        )

        # Stage 2: 112x112 -> 56x56
        self.stage2 = DepthwiseSeparableConv(c1, c2, stride=2)

        # Stage 3: 56x56 -> 28x28
        self.stage3 = DepthwiseSeparableConv(c2, out_channels, stride=2)

        # Refinement block at 28x28 (Target layer for Grad-CAM)
        self.refinement = DepthwiseSeparableConv(out_channels, out_channels, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, in_channels, H, W)
        Returns:
            Local feature map of shape (B, out_channels, H/8, W/8)
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.refinement(x)
        return x

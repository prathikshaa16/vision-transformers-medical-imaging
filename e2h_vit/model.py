"""
E2HViT: Efficient Explainable Hybrid Vision Transformer
Complete unified architecture combining:
1. ConvStem (Inductive local feature preservation)
2. SwinBranch (Hierarchical shifted-window global context)
3. CrossAttentionFeatureFusion (CAFF bridging local and global)
4. WGAPHead (Weighted Global Average Pooling classification)
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from .stem import ConvStem
from .swin_block import SwinBranch
from .fusion import CrossAttentionFeatureFusion
from .head import WGAPHead


class E2HViT(nn.Module):
    """
    E2H-ViT (Efficient Explainable Hybrid Vision Transformer).
    
    Synthesizes the four foundational paradigms from recent literature:
    - Local inductive edge preservation from CNNs (Paper 1 & Paper 2)
    - Shifted-window global self-attention from Swin (Paper 1 & Paper 2)
    - Compact WGAP pooling for edge/IoMT efficiency (Paper 3)
    - Dual-lens explainability pathways for clinical trust (Paper 4)
    """
    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        num_classes: int = 2,
        stem_channels: int = 352,
        swin_embed_dim: int = 176,
        swin_out_dim: int = 352,
        window_size: int = 7,
        fusion_heads: int = 8,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        assert stem_channels == swin_out_dim, "stem_channels and swin_out_dim must match for fusion"

        self.img_size = img_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.spatial_res = (img_size // 8, img_size // 8)  # (28, 28) for 224x224

        # 1. Local Inductive CNN Stem
        self.stem = ConvStem(in_channels=in_chans, out_channels=stem_channels)

        # 2. Global Context Swin Transformer Branch
        self.swin = SwinBranch(
            img_size=img_size,
            patch_size=4,
            in_chans=in_chans,
            embed_dim=swin_embed_dim,
            out_dim=swin_out_dim,
            window_size=window_size,
        )

        # 3. Cross-Attention Feature Fusion (CAFF)
        self.fusion = CrossAttentionFeatureFusion(
            dim=stem_channels,
            num_heads=fusion_heads,
            proj_drop=drop_rate,
        )

        # 4. Weighted Global Average Pooling Classification Head
        self.head = WGAPHead(
            in_features=stem_channels,
            num_classes=num_classes,
            drop=drop_rate,
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass of E2H-ViT.
        Args:
            x: Input medical image tensor of shape (B, in_chans, H, W)
            return_features: If True, returns a dictionary containing intermediate feature maps
        Returns:
            logits: Diagnostic predictions of shape (B, num_classes)
            features: Optional dictionary of internal feature maps
        """
        # 1. Local CNN Stem
        cnn_feat = self.stem(x)  # (B, C, 28, 28)

        # 2. Global Swin Branch
        swin_tokens, swin_feat = self.swin(x)  # (B, N, C), (B, C, 28, 28)

        # 3. Cross-Attention Feature Fusion
        fused_tokens, fused_feat = self.fusion(cnn_feat, swin_feat)

        # 4. WGAP Classification Head
        logits, spatial_weights = self.head(fused_tokens, spatial_res=self.spatial_res)

        if return_features:
            features = {
                "cnn_feat": cnn_feat,
                "swin_feat": swin_feat,
                "fused_feat": fused_feat,
                "spatial_weights": spatial_weights,
                "swin_attns": self.swin.get_attention_maps(),
            }
            return logits, features

        return logits

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_breakdown(self) -> Dict[str, int]:
        """Returns parameter count per architectural module."""
        return {
            "CNN Stem": sum(p.numel() for p in self.stem.parameters()),
            "Swin Branch": sum(p.numel() for p in self.swin.parameters()),
            "Cross Fusion (CAFF)": sum(p.numel() for p in self.fusion.parameters()),
            "WGAP Head": sum(p.numel() for p in self.head.parameters()),
            "Total": self.count_parameters(),
        }


def build_e2h_vit_nano(num_classes: int = 2, in_chans: int = 3, img_size: int = 224) -> E2HViT:
    """
    Builds E2H-ViT-Nano (~0.44M parameters).
    Ultra-lightweight configuration for low-power microcontroller / wearable IoMT hardware.
    """
    return E2HViT(
        img_size=img_size,
        in_chans=in_chans,
        num_classes=num_classes,
        stem_channels=96,
        swin_embed_dim=48,
        swin_out_dim=96,
        window_size=7,
        fusion_heads=6,
        drop_rate=0.1,
    )


def build_e2h_vit_tiny(num_classes: int = 2, in_chans: int = 3, img_size: int = 224) -> E2HViT:
    """
    Builds E2H-ViT-Tiny (~5.78M parameters).
    Optimized for IoMT, mobile point-of-care clinics, and real-time medical screening.
    """
    return E2HViT(
        img_size=img_size,
        in_chans=in_chans,
        num_classes=num_classes,
        stem_channels=352,
        swin_embed_dim=176,
        swin_out_dim=352,
        window_size=7,
        fusion_heads=8,
        drop_rate=0.1,
    )


def build_e2h_vit_small(num_classes: int = 2, in_chans: int = 3, img_size: int = 224) -> E2HViT:
    """
    Builds E2H-ViT-Small (~12.2M parameters).
    High-capacity variant for multi-organ segmentation and gigapixel pathology.
    """
    return E2HViT(
        img_size=img_size,
        in_chans=in_chans,
        num_classes=num_classes,
        stem_channels=512,
        swin_embed_dim=256,
        swin_out_dim=512,
        window_size=7,
        fusion_heads=16,
        drop_rate=0.15,
    )

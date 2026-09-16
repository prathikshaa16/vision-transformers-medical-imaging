"""
Cross-Attention Feature Fusion (CAFF) for E2H-ViT
Bridges localized CNN boundary features with global Swin Transformer context.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFeatureFusion(nn.Module):
    """
    Cross-Attention Feature Fusion Module (CAFF).
    Combines local CNN inductive features with global Swin Transformer representations.
    
    Query: Global Swin Tokens (seeking localized spatial verification)
    Key & Value: Local CNN Feature Map (providing boundary texture & edge references)
    """
    def __init__(
        self,
        dim: int = 96,
        num_heads: int = 6,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim

        # Ensure num_heads evenly divides dim
        valid_heads = num_heads
        if dim % valid_heads != 0:
            for h in [8, 16, 12, 6, 4, 2, 1]:
                if dim % h == 0:
                    valid_heads = h
                    break
        self.num_heads = valid_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.norm_swin = nn.LayerNorm(dim)
        self.norm_cnn = nn.LayerNorm(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Adaptive Channel & Spatial Gating
        self.gate_fc = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        # Feed-Forward Network refinement
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(2 * dim, dim),
            nn.Dropout(proj_drop),
        )

    def forward(
        self, cnn_feat: torch.Tensor, swin_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cnn_feat: Local CNN feature map (B, C, H, W)
            swin_feat: Global Swin feature map (B, C, H, W) or (B, N, C)
        Returns:
            fused_tokens: (B, N, C) where N = H * W
            fused_map: (B, C, H, W)
        """
        B, C, H, W = cnn_feat.shape
        N = H * W

        # Flatten CNN spatial dimensions to sequence: (B, N, C)
        t_cnn = cnn_feat.flatten(2).transpose(1, 2)

        # Flatten Swin feature map if 4D
        if swin_feat.dim() == 4:
            t_swin = swin_feat.flatten(2).transpose(1, 2)
        else:
            t_swin = swin_feat

        # Layer norms
        q_in = self.norm_swin(t_swin)
        kv_in = self.norm_cnn(t_cnn)

        # Linear projections for Multi-Head Cross-Attention
        q = self.q(q_in).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(kv_in).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(kv_in).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Attention: (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Aggregated cross features
        cross = (attn @ v).transpose(1, 2).reshape(B, N, C)
        cross = self.proj(cross)
        cross = self.proj_drop(cross)

        # Adaptive Gating between Swin context and CNN local features
        gate = self.gate_fc(torch.cat([t_swin, t_cnn], dim=-1))
        fused = t_swin + cross + gate * t_cnn

        # FFN refinement with residual connection
        fused = fused + self.ffn(self.norm_ffn(fused))

        # Reconstruct spatial feature map
        fused_map = fused.transpose(1, 2).view(B, C, H, W)
        return fused, fused_map

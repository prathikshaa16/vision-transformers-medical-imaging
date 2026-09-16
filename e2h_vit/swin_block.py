"""
Swin Transformer Branch for E2H-ViT
Hierarchical shifted-window self-attention in pure PyTorch.
Computes long-range global context with O(M^2 * N * d) linear complexity.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_valid_heads(dim: int, preferred: int = 4) -> int:
    """Finds the largest valid divisor head count close to preferred."""
    for h in [preferred, 16, 12, 8, 6, 4, 2, 1]:
        if dim % h == 0 and h <= 16:
            return h
    return 1


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition input into non-overlapping windows.
    Args:
        x: Tensor of shape (B, H, W, C)
        window_size: Window dimension M
    Returns:
        Windows tensor of shape (num_windows * B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse windows back into full image feature map.
    Args:
        windows: (num_windows * B, window_size, window_size, C)
        window_size: Window dimension M
        H, W: Spatial height and width of feature map
    Returns:
        Tensor of shape (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Window-based Multi-Head Self-Attention (W-MSA) with relative position bias.
    Supports both standard windows and shifted windows (SW-MSA).
    """
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Hook to store attention weights for Attention Rollout / Explainability
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input features with shape of (num_windows * B, N, C)
            mask: Attention mask (0/-100) of shape (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        # Cache attention weights for explainability rollout
        self.last_attn_weights = attn.detach()

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """Multi-Layer Perceptron (MLP) block for Swin Transformer."""
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        out_features = in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Single Swin Transformer Block with LayerNorm, W-MSA or SW-MSA, and MLP.
    """
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        # Compute cyclic shift attention mask if shift_size > 0
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input feature size ({L}) does not match resolution ({H}*{W})"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window attention / Shifted window attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x

        # FFN
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging Layer: Downsamples spatial dimensions by 2 and linearly projects channels 2C.
    """
    def __init__(self, input_resolution: Tuple[int, int], dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"H ({H}) and W ({W}) must be even"

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch Embedding: Projects image (B, C_in, H, W) to (B, H/patch_size * W/patch_size, embed_dim).
    """
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 48):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


class SwinBranch(nn.Module):
    """
    Hierarchical 2-stage Swin Transformer Branch.
    Processes input image to 28x28 token map with long-range multi-head shifted-window attention.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 48,
        out_dim: int = 96,
        window_size: int = 7,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        res1 = self.patch_embed.patches_resolution  # (56, 56)

        heads1 = get_valid_heads(embed_dim, preferred=4)
        heads2 = get_valid_heads(out_dim, preferred=8)

        # Stage 1: 2 blocks at 56x56
        self.stage1_block1 = SwinTransformerBlock(
            dim=embed_dim, input_resolution=res1, num_heads=heads1, window_size=window_size, shift_size=0
        )
        self.stage1_block2 = SwinTransformerBlock(
            dim=embed_dim, input_resolution=res1, num_heads=heads1, window_size=window_size, shift_size=window_size // 2
        )

        # Patch Merging: 56x56 -> 28x28, dim embed_dim -> 2 * embed_dim
        self.merge = PatchMerging(input_resolution=res1, dim=embed_dim)
        res2 = (res1[0] // 2, res1[1] // 2)  # (28, 28)

        # Stage 2: 2 blocks at 28x28
        self.stage2_block1 = SwinTransformerBlock(
            dim=out_dim, input_resolution=res2, num_heads=heads2, window_size=window_size, shift_size=0
        )
        self.stage2_block2 = SwinTransformerBlock(
            dim=out_dim, input_resolution=res2, num_heads=heads2, window_size=window_size, shift_size=window_size // 2
        )
        self.norm = nn.LayerNorm(out_dim)
        self.out_res = res2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            token_seq: (B, N, out_dim) where N = 28 * 28 = 784
            feature_map: (B, out_dim, 28, 28)
        """
        x = self.patch_embed(x)
        x = self.stage1_block1(x)
        x = self.stage1_block2(x)
        x = self.merge(x)
        x = self.stage2_block1(x)
        x = self.stage2_block2(x)
        x = self.norm(x)

        B, N, C = x.shape
        H, W = self.out_res
        feature_map = x.transpose(1, 2).view(B, C, H, W)
        return x, feature_map

    def get_attention_maps(self) -> list:
        """Collects cached attention weight tensors from all Swin blocks."""
        attns = []
        for block in [self.stage1_block1, self.stage1_block2, self.stage2_block1, self.stage2_block2]:
            if block.attn.last_attn_weights is not None:
                attns.append(block.attn.last_attn_weights)
        return attns

"""
Weighted Global Average Pooling (WGAP) Head for E2H-ViT
Inspired by LightAMViT (Springer 2025).
Computes learned spatial token weighting to prioritize focal lesion regions while minimizing parameter footprint.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class WGAPHead(nn.Module):
    """
    Weighted Global Average Pooling (WGAP) Classification Head.
    
    Instead of uniform averaging (1/N * sum(x)), WGAP learns a spatial importance
    distribution w = softmax(Linear(x)), focusing the diagnostic representation
    on pathological lesion areas while remaining lightweight.
    """
    def __init__(self, in_features: int = 96, num_classes: int = 2, drop: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        # Spatial importance scoring
        self.weight_proj = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.GELU(),
            nn.Linear(in_features // 2, 1),
        )

        self.norm = nn.LayerNorm(in_features)
        self.drop = nn.Dropout(drop)
        self.classifier = nn.Linear(in_features, num_classes)

        # Cached spatial weight map for visualization
        self.last_spatial_weights: Optional[torch.Tensor] = None

    def forward(
        self, tokens: torch.Tensor, spatial_res: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: Fused sequence tokens (B, N, C)
            spatial_res: Optional (H, W) tuple to reshape weights into a 2D map
        Returns:
            logits: Classification output (B, num_classes)
            spatial_weights: Learned attention weights (B, N, 1) or (B, 1, H, W)
        """
        B, N, C = tokens.shape

        # Compute importance scores across tokens: (B, N, 1)
        raw_scores = self.weight_proj(tokens)
        weights = F.softmax(raw_scores, dim=1)
        self.last_spatial_weights = weights.detach()

        # Weighted spatial aggregation: sum(w_i * x_i)
        pooled = (tokens * weights).sum(dim=1)  # (B, C)

        # Classification projection
        pooled = self.norm(pooled)
        pooled = self.drop(pooled)
        logits = self.classifier(pooled)

        if spatial_res is not None:
            H, W = spatial_res
            weight_map = weights.view(B, H, W, 1).permute(0, 3, 1, 2).contiguous()
        else:
            weight_map = weights

        return logits, weight_map

"""
E2H-ViT: Efficient Explainable Hybrid Vision Transformer for Medical Image Analysis
Package Initialization
"""

from .model import E2HViT, build_e2h_vit_nano, build_e2h_vit_tiny, build_e2h_vit_small
from .stem import ConvStem
from .swin_block import SwinBranch, SwinTransformerBlock, WindowAttention
from .fusion import CrossAttentionFeatureFusion
from .head import WGAPHead
from .explainability import GradCAM, AttentionRollout, DualLensExplainer, evaluate_explanation_faithfulness

__all__ = [
    "E2HViT",
    "build_e2h_vit_nano",
    "build_e2h_vit_tiny",
    "build_e2h_vit_small",
    "ConvStem",
    "SwinBranch",
    "SwinTransformerBlock",
    "WindowAttention",
    "CrossAttentionFeatureFusion",
    "WGAPHead",
    "GradCAM",
    "AttentionRollout",
    "DualLensExplainer",
    "evaluate_explanation_faithfulness",
]

__version__ = "1.0.0"

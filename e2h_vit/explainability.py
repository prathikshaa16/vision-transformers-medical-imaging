"""
Dual-Lens Explainability Engine for E2H-ViT
Combines:
1. Grad-CAM on the CNN local stem (cellular boundary & texture saliency)
2. Attention Rollout on the Swin Transformer branch (long-range context importance)
3. Quantitative Explanation Faithfulness evaluation via perturbation testing (from XViT / Paper 4)
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for E2H-ViT.
    Hooks into the final convolutional layer of the local stem.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or model.stem.refinement
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
        """
        Generates Grad-CAM heatmap for a single image tensor.
        Args:
            x: Input tensor of shape (1, C, H, W)
            class_idx: Target diagnostic class index (if None, uses argmax)
        Returns:
            Normalized heatmap tensor of shape (H, W) in [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=-1).item()

        # Backward pass on target class score
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations/gradients")

        # Global average pooling of gradients: (1, C, 1, 1)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of forward activation maps: (1, 1, h, w)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))

        # Upsample to original image resolution (H, W)
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze()

        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


class AttentionRollout:
    """
    Attention Rollout for Swin Transformer Branch.
    Reconstructs patch importance flow across hierarchical shifted windows.
    """
    def __init__(self, model: nn.Module):
        self.model = model

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs spatial attention map across Swin layers.
        Args:
            x: Input tensor of shape (1, C, H, W)
        Returns:
            Attention heatmap tensor of shape (H, W) in [0, 1]
        """
        self.model.eval()
        with torch.no_grad():
            _, features = self.model(x, return_features=True)

        attns = features.get("swin_attns", [])
        H_out, W_out = self.model.spatial_res  # (28, 28)

        if not attns:
            # Fallback to WGAP spatial importance weights if raw attention is unavailable
            spatial_weights = features.get("spatial_weights", None)
            if spatial_weights is not None:
                attn_map = F.interpolate(
                    spatial_weights, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False
                ).squeeze()
                attn_min, attn_max = attn_map.min(), attn_map.max()
                return (attn_map - attn_min) / (attn_max - attn_min + 1e-8)

        # Process the last stage Swin attention maps (Stage 2 operates at 28x28 with 7x7 windows)
        # num_windows = (28/7) * (28/7) = 16 windows
        last_attn = attns[-1]  # Shape: (num_windows, num_heads, window_size*window_size, window_size*window_size)
        M = 7
        num_windows_side = H_out // M  # 4

        # Average across attention heads
        attn_avg = last_attn.mean(dim=1)  # (16, 49, 49)

        # Average attention received by each token in each window (column mean)
        token_importance = attn_avg.mean(dim=1)  # (16, 49)
        token_importance = token_importance.view(num_windows_side, num_windows_side, M, M)
        # Permute to reconstruct full (28, 28) spatial map
        full_map = token_importance.permute(0, 2, 1, 3).contiguous().view(1, 1, H_out, W_out)

        # Upsample to input image resolution (H, W)
        full_map = F.interpolate(full_map, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        full_map = full_map.squeeze()

        # Min-max normalization
        f_min, f_max = full_map.min(), full_map.max()
        if f_max > f_min:
            full_map = (full_map - f_min) / (f_max - f_min)
        else:
            full_map = torch.zeros_like(full_map)

        return full_map


class DualLensExplainer:
    """
    Dual-Lens Explainability Engine synthesizing CNN Grad-CAM and Transformer Attention.
    Provides clinicians with both cellular boundary justification and global anatomical context.
    """
    def __init__(self, model: nn.Module, beta: float = 0.5):
        """
        Args:
            model: E2HViT model instance
            beta: Weighting factor for CNN Grad-CAM vs Transformer Attention (0.0 to 1.0)
        """
        self.model = model
        self.beta = beta
        self.grad_cam = GradCAM(model)
        self.attn_rollout = AttentionRollout(model)

    def explain(
        self, x: torch.Tensor, class_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generates individual and fused saliency maps.
        Args:
            x: Input tensor of shape (1, C, H, W)
            class_idx: Target class index
        Returns:
            Dictionary with 'grad_cam', 'attention', and 'fused' heatmaps
        """
        cam_map = self.grad_cam.generate(x, class_idx=class_idx)
        attn_map = self.attn_rollout.generate(x)

        fused = self.beta * cam_map + (1.0 - self.beta) * attn_map
        f_min, f_max = fused.min(), fused.max()
        if f_max > f_min:
            fused = (fused - f_min) / (f_max - f_min)

        return {
            "grad_cam": cam_map,
            "attention": attn_map,
            "fused": fused,
        }

    def close(self):
        self.grad_cam.remove_hooks()


def evaluate_explanation_faithfulness(
    model: nn.Module,
    x: torch.Tensor,
    saliency_map: torch.Tensor,
    mask_ratio: float = 0.20,
    class_idx: Optional[int] = None,
) -> float:
    """
    Evaluates explanation faithfulness via perturbation testing (inspired by XViT / Paper 4).
    Masks the top-k% most salient pixels identified by the explanation.
    Faithfulness is measured by the relative drop in predicted target class probability:
        Faithfulness = (P(y|X) - P(y|X_masked)) / P(y|X)
    
    A high positive score indicates the explanation highlighted truly diagnostic features.
    
    Args:
        model: E2HViT model
        x: Input image tensor (1, C, H, W)
        saliency_map: 2D saliency heatmap (H, W) in [0, 1]
        mask_ratio: Fraction of top salient pixels to mask (e.g. 0.20 = 20%)
        class_idx: Target class index
    Returns:
        Faithfulness score in [0.0, 1.0]
    """
    model.eval()
    with torch.no_grad():
        logits_orig = model(x)
        probs_orig = F.softmax(logits_orig, dim=-1)

        if class_idx is None:
            class_idx = logits_orig.argmax(dim=-1).item()

        p_orig = probs_orig[0, class_idx].item()

        # Find threshold for top mask_ratio pixels
        flat_saliency = saliency_map.flatten()
        k = int(len(flat_saliency) * mask_ratio)
        if k == 0:
            return 0.0

        top_k_val, _ = torch.topk(flat_saliency, k)
        threshold = top_k_val[-1]

        # Mask top salient pixels (set to 0 / background)
        mask = (saliency_map < threshold).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        x_masked = x * mask

        logits_masked = model(x_masked)
        probs_masked = F.softmax(logits_masked, dim=-1)
        p_masked = probs_masked[0, class_idx].item()

        # Relative drop in probability
        if p_orig > 1e-7:
            faithfulness = max(0.0, (p_orig - p_masked) / p_orig)
        else:
            faithfulness = 0.0

    return float(faithfulness)

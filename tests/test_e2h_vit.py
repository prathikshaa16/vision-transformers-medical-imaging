"""
Unit & Integration Test Suite for E2H-ViT Framework
Verifies:
- ConvStem forward pass & shapes
- Swin Transformer shifted-window attention & patch merging
- Cross-Attention Feature Fusion (CAFF)
- Weighted Global Average Pooling (WGAP)
- Full model forward & backward passes
- Dual-Lens Explainability (Grad-CAM + Attention Rollout)
- Quantitative Perturbation Faithfulness metric
- Parameter footprints across Nano and Tiny configurations
"""

import os
import sys
import unittest
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from e2h_vit import (
    E2HViT,
    build_e2h_vit_nano,
    build_e2h_vit_tiny,
    build_e2h_vit_small,
    ConvStem,
    SwinBranch,
    CrossAttentionFeatureFusion,
    WGAPHead,
    GradCAM,
    AttentionRollout,
    DualLensExplainer,
    evaluate_explanation_faithfulness,
)


class TestE2HViTModules(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.B = 2
        self.C = 3
        self.H = 224
        self.W = 224
        self.x = torch.randn(self.B, self.C, self.H, self.W, device=self.device)

    def test_conv_stem(self):
        stem = ConvStem(in_channels=3, out_channels=96).to(self.device)
        out = stem(self.x)
        self.assertEqual(out.shape, (self.B, 96, 28, 28))

    def test_swin_branch(self):
        swin = SwinBranch(img_size=224, patch_size=4, in_chans=3, embed_dim=48, out_dim=96, window_size=7).to(self.device)
        tokens, feat_map = swin(self.x)
        self.assertEqual(tokens.shape, (self.B, 784, 96))
        self.assertEqual(feat_map.shape, (self.B, 96, 28, 28))
        attns = swin.get_attention_maps()
        self.assertGreater(len(attns), 0)

    def test_cross_attention_fusion(self):
        fusion = CrossAttentionFeatureFusion(dim=96, num_heads=6).to(self.device)
        cnn_feat = torch.randn(self.B, 96, 28, 28, device=self.device)
        swin_feat = torch.randn(self.B, 96, 28, 28, device=self.device)
        fused_tokens, fused_map = fusion(cnn_feat, swin_feat)
        self.assertEqual(fused_tokens.shape, (self.B, 784, 96))
        self.assertEqual(fused_map.shape, (self.B, 96, 28, 28))

    def test_wgap_head(self):
        head = WGAPHead(in_features=96, num_classes=3).to(self.device)
        tokens = torch.randn(self.B, 784, 96, device=self.device)
        logits, weight_map = head(tokens, spatial_res=(28, 28))
        self.assertEqual(logits.shape, (self.B, 3))
        self.assertEqual(weight_map.shape, (self.B, 1, 28, 28))
        # Weights should sum to 1 over spatial dimension
        self.assertTrue(torch.allclose(weight_map.sum(dim=(1, 2, 3)), torch.ones(self.B, device=self.device), atol=1e-5))


class TestE2HViTModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = build_e2h_vit_nano(num_classes=2)
        self.x = torch.randn(2, 3, 224, 224)

    def test_forward_shape(self):
        logits = self.model(self.x)
        self.assertEqual(logits.shape, (2, 2))

    def test_forward_with_features(self):
        logits, features = self.model(self.x, return_features=True)
        self.assertEqual(logits.shape, (2, 2))
        self.assertIn("cnn_feat", features)
        self.assertIn("swin_feat", features)
        self.assertIn("fused_feat", features)
        self.assertIn("spatial_weights", features)
        self.assertIn("swin_attns", features)

    def test_backward_pass(self):
        self.model.train()
        logits = self.model(self.x)
        loss = logits.sum()
        loss.backward()

        # Check that gradients flow to stem and swin parameters
        stem_grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.stem.parameters())
        swin_grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.swin.parameters())
        fusion_grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.fusion.parameters())
        head_grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.head.parameters())

        self.assertTrue(stem_grad_exists, "Gradients missing in CNN stem")
        self.assertTrue(swin_grad_exists, "Gradients missing in Swin branch")
        self.assertTrue(fusion_grad_exists, "Gradients missing in Cross-attention fusion")
        self.assertTrue(head_grad_exists, "Gradients missing in WGAP head")

    def test_parameter_counts(self):
        nano = build_e2h_vit_nano(num_classes=2)
        tiny = build_e2h_vit_tiny(num_classes=2)

        nano_params = nano.count_parameters()
        tiny_params = tiny.count_parameters()

        # Nano should be ~0.44M
        self.assertGreater(nano_params, 400_000)
        self.assertLess(nano_params, 500_000)

        # Tiny should be ~5.78M
        self.assertGreater(tiny_params, 5_000_000)
        self.assertLess(tiny_params, 6_500_000)


class TestExplainability(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = build_e2h_vit_nano(num_classes=2)
        self.x = torch.randn(1, 3, 224, 224)

    def test_grad_cam(self):
        cam = GradCAM(self.model)
        heatmap = cam.generate(self.x, class_idx=0)
        cam.remove_hooks()

        self.assertEqual(heatmap.shape, (224, 224))
        self.assertGreaterEqual(heatmap.min().item(), 0.0)
        self.assertLessEqual(heatmap.max().item(), 1.0)

    def test_attention_rollout(self):
        rollout = AttentionRollout(self.model)
        heatmap = rollout.generate(self.x)

        self.assertEqual(heatmap.shape, (224, 224))
        self.assertGreaterEqual(heatmap.min().item(), 0.0)
        self.assertLessEqual(heatmap.max().item(), 1.0)

    def test_dual_lens_explainer(self):
        explainer = DualLensExplainer(self.model, beta=0.5)
        maps = explainer.explain(self.x, class_idx=0)
        explainer.close()

        self.assertIn("grad_cam", maps)
        self.assertIn("attention", maps)
        self.assertIn("fused", maps)
        self.assertEqual(maps["fused"].shape, (224, 224))

    def test_faithfulness_metric(self):
        explainer = DualLensExplainer(self.model)
        maps = explainer.explain(self.x, class_idx=0)
        explainer.close()

        faithfulness = evaluate_explanation_faithfulness(
            self.model, self.x, maps["fused"], mask_ratio=0.20, class_idx=0
        )
        self.assertIsInstance(faithfulness, float)
        self.assertGreaterEqual(faithfulness, 0.0)
        self.assertLessEqual(faithfulness, 1.0)


if __name__ == "__main__":
    unittest.main()

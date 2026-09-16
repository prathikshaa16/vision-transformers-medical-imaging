"""
E2H-ViT Real Medical Benchmark Training & Evaluation: BreastMNIST
Trains E2H-ViT on the real MedMNIST Breast Ultrasound dataset (matching Paper 3 / BUSI domain).
Computes training and validation accuracy, loss curves, and generates dual-lens visual explanations
on real clinical ultrasound scans, saving the figure to assets/real_breastmnist_explanation.png.
"""

import argparse
import os
import sys
import time
import urllib.request
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repository root is on path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from e2h_vit import (
    build_e2h_vit_nano,
    build_e2h_vit_tiny,
    DualLensExplainer,
    evaluate_explanation_faithfulness,
)


MEDMNIST_BREAST_URL = "https://zenodo.org/records/6496656/files/breastmnist.npz?download=1"


def download_breastmnist(data_dir: str = "data") -> str:
    """Downloads BreastMNIST dataset if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "breastmnist.npz")
    if not os.path.exists(file_path):
        print(f"[*] Downloading BreastMNIST dataset from Zenodo to {file_path}...")
        urllib.request.urlretrieve(MEDMNIST_BREAST_URL, file_path)
        print(f"[+] Download complete: {os.path.getsize(file_path):,} bytes")
    else:
        print(f"[*] Found cached BreastMNIST dataset: {file_path}")
    return file_path


class RealBreastMNISTDataset(Dataset):
    """
    Real MedMNIST Breast Ultrasound Dataset.
    Upsamples 28x28 grayscale ultrasound scans to 224x224 RGB tensors for E2H-ViT.
    """
    def __init__(self, npz_path: str, split: str = "train", target_size: int = 224):
        data = np.load(npz_path)
        self.images = data[f"{split}_images"]  # (N, 28, 28)
        self.labels = data[f"{split}_labels"].squeeze(-1)  # (N,)
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, np.ndarray]:
        img_raw = self.images[idx].astype(np.float32) / 255.0  # (28, 28) in [0, 1]
        label = int(self.labels[idx])

        # Convert to tensor: (1, 1, 28, 28)
        t = torch.from_numpy(img_raw).unsqueeze(0).unsqueeze(0)

        # Bicubic upsample to 224x224
        t_up = F.interpolate(t, size=(self.target_size, self.target_size), mode="bicubic", align_corners=False)
        t_up = t_up.squeeze(0)  # (1, 224, 224)

        # 3-channel RGB: (3, 224, 224)
        t_rgb = torch.cat([t_up, t_up, t_up], dim=0)

        return t_rgb, label, img_raw


def train_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device
) -> Tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / max(1, total), correct / max(1, total)


def evaluate_model(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / max(1, total), correct / max(1, total)


def run_benchmark(
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-4,
    model_type: str = "nano",
    output_dir: str = "checkpoints",
):
    print("=" * 70)
    print("  E2H-ViT Real Medical Benchmark: Breast Ultrasound (MedMNIST)")
    print("  Evaluating on Real Patient Scans (Matching Paper 3 / BUSI Domain)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")

    # 1. Prepare Dataset
    npz_path = download_breastmnist()
    train_dataset = RealBreastMNISTDataset(npz_path, split="train", target_size=224)
    val_dataset = RealBreastMNISTDataset(npz_path, split="val", target_size=224)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"[*] Loaded Real Medical Dataset:")
    print(f"    - Training scans  : {len(train_dataset)} patient ultrasound images")
    print(f"    - Validation scans: {len(val_dataset)} patient ultrasound images")

    # 2. Build Model
    if model_type == "tiny":
        model = build_e2h_vit_tiny(num_classes=2).to(device)
    else:
        model = build_e2h_vit_nano(num_classes=2).to(device)

    print(f"\n[*] Initialized E2H-ViT-{model_type.capitalize()}: {model.count_parameters():,d} trainable parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(output_dir, exist_ok=True)
    best_acc = 0.0
    best_path = os.path.join(output_dir, f"best_breastmnist_{model_type}.pt")

    print("\n[*] Training E2H-ViT on Real Breast Ultrasound Cohort:")
    t_start = time.time()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()
        t_ep = time.time() - t0

        print(
            f"  Epoch [{ep:02d}/{epochs:02d}] "
            f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:5.1f}% | "
            f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc*100:5.1f}% | "
            f"Time: {t_ep:.1f}s"
        )

        if v_acc > best_acc or ep == 1:
            best_acc = v_acc
            torch.save(model.state_dict(), best_path)

    print("-" * 70)
    print(f"[+] Completed {epochs} epochs in {time.time() - t_start:.1f}s")
    print(f"[+] Peak Validation Accuracy on Real Scans: {best_acc*100:.1f}%")

    # 3. Generate Real Patient Dual-Lens Explainability Report
    print("\n[*] Generating Dual-Lens Clinical Explanations for Real Patient Case...")
    model.eval()

    # Find a positive lesion case from validation set
    sample_img, sample_lbl = None, None
    for img_tensor, lbl, raw in val_dataset:
        if lbl == 1:
            sample_img = img_tensor.unsqueeze(0).to(device)
            sample_lbl = lbl
            sample_raw = raw
            break

    if sample_img is None:
        sample_img = val_dataset[0][0].unsqueeze(0).to(device)
        sample_lbl = val_dataset[0][1]
        sample_raw = val_dataset[0][2]

    with torch.no_grad():
        logits = model(sample_img)
        probs = F.softmax(logits, dim=-1)
        pred_class = logits.argmax(dim=-1).item()
        confidence = probs[0, pred_class].item()

    class_names = ["Malignant Lesion", "Benign / Normal Tissue"]
    print(f"    - Patient Ground Truth  : {class_names[sample_lbl]}")
    print(f"    - Model Prediction      : {class_names[pred_class]}")
    print(f"    - Diagnostic Confidence : {confidence*100:.1f}%")

    explainer = DualLensExplainer(model, beta=0.5)
    maps = explainer.explain(sample_img, class_idx=pred_class)
    explainer.close()

    faithfulness = evaluate_explanation_faithfulness(
        model, sample_img, maps["fused"], mask_ratio=0.20, class_idx=pred_class
    )
    print(f"    - Explanation Faithfulness Score: {faithfulness*100:.2f}% drop upon salient region perturbation")

    # 4. Render and Save Figure
    out_fig = "assets/real_breastmnist_explanation.png"
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=200)

    # Upsampled scan for visual display
    disp_img = sample_img.squeeze(0).mean(dim=0).cpu().numpy()
    cam_np = maps["grad_cam"].cpu().numpy()
    attn_np = maps["attention"].cpu().numpy()
    fused_np = maps["fused"].cpu().numpy()

    # Panel 1: Original Patient Ultrasound Scan
    axes[0].imshow(disp_img, cmap="bone")
    axes[0].set_title(f"Real Patient Ultrasound Scan\n(Truth: {class_names[sample_lbl]})", fontsize=11, fontweight="bold", pad=8)
    axes[0].axis("off")

    # Panel 2: CNN Stem Grad-CAM
    axes[1].imshow(disp_img, cmap="bone")
    im_cam = axes[1].imshow(cam_np, cmap="inferno", alpha=0.55)
    axes[1].set_title("1. CNN Stem Grad-CAM\n(Localized Edge / Texture Saliency)", fontsize=11, fontweight="bold", pad=8)
    plt.colorbar(im_cam, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].axis("off")

    # Panel 3: Swin Attention Rollout
    axes[2].imshow(disp_img, cmap="bone")
    im_attn = axes[2].imshow(attn_np, cmap="viridis", alpha=0.55)
    axes[2].set_title("2. Swin Attention Rollout\n(Global Contextual Attention)", fontsize=11, fontweight="bold", pad=8)
    plt.colorbar(im_attn, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].axis("off")

    # Panel 4: Dual-Lens Fused Explanation
    axes[3].imshow(disp_img, cmap="bone")
    im_fused = axes[3].imshow(fused_np, cmap="jet", alpha=0.55)
    axes[3].set_title(
        f"3. Dual-Lens Fused Explanation\n(Faithfulness: {faithfulness*100:.1f}%)",
        fontsize=11, fontweight="bold", color="#059669", pad=8
    )
    plt.colorbar(im_fused, ax=axes[3], fraction=0.046, pad=0.04)
    axes[3].axis("off")

    plt.suptitle(
        f"E2H-ViT Real Clinical Case Report (MedMNIST Breast Ultrasound) | Predicted: {class_names[pred_class]} ({confidence*100:.1f}%)",
        fontsize=13, fontweight="bold", y=0.98
    )
    plt.tight_layout()
    plt.savefig(out_fig, bbox_inches="tight")
    plt.close()

    print(f"[+] Real clinical explanation figure saved to: {os.path.abspath(out_fig)}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train E2H-ViT on MedMNIST Breast Ultrasound")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--model-type", type=str, default="nano", choices=["nano", "tiny"], help="Model variant")
    args = parser.parse_args()

    run_benchmark(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, model_type=args.model_type)

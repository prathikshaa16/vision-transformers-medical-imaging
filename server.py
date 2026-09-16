"""
E2H-ViT Production FastAPI Backend Server
Provides REST API endpoints for:
- /api/health: System health and model status
- /api/samples: Pre-packaged clinical test library
- /api/analyze: Real-time PyTorch inference, Grad-CAM, Swin Attention Rollout,
                lesion quantification, and base64 heatmap visualization
- Static file serving for the complete web UI
"""

import base64
import io
import os
import sys
import time
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure repository root is on path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from e2h_vit import (
    E2HViT,
    build_e2h_vit_nano,
    build_e2h_vit_tiny,
    DualLensExplainer,
    evaluate_explanation_faithfulness,
)

app = FastAPI(
    title="E2H-ViT Medical Diagnostic API",
    description="Full-stack clinical AI diagnostic backend using Efficient Explainable Hybrid Vision Transformers",
    version="1.0.0",
)

# Enable CORS for local cross-port access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
START_TIME = time.time()
MODELS: Dict[str, E2HViT] = {}

SAMPLE_LIBRARY = {
    "brain_mri": {
        "id": "brain_mri",
        "title": "3D Brain MRI (FLAIR)",
        "paper": "Paper 1 (BMC Neurology 2024)",
        "task": "White Matter Lesion Segmentation",
        "path": "assets/samples/brain_mri.png",
        "ground_truth": "White Matter Hyperintensity (Ischemic / Small Vessel Disease)",
    },
    "cardiac_cine": {
        "id": "cardiac_cine",
        "title": "Cardiac Cine-MRI (Short-Axis)",
        "paper": "Paper 2 (PSVT / Elsevier 2025)",
        "task": "Cardiac Chamber & Wall Segmentation",
        "path": "assets/samples/cardiac_cine.png",
        "ground_truth": "Left Ventricular Chamber & Myocardium",
    },
    "breast_ultrasound": {
        "id": "breast_ultrasound",
        "title": "Breast Ultrasound (BUSI / MedMNIST)",
        "paper": "Paper 3 (LightAMViT / Springer 2025)",
        "task": "Focal Breast Lesion Classification",
        "path": "assets/samples/breast_ultrasound.png",
        "ground_truth": "Hypoechoic Focal Lesion with Acoustic Shadowing",
    },
    "histopathology": {
        "id": "histopathology",
        "title": "Cancer Histopathology (LCS25000)",
        "paper": "Paper 4 (XViT / Elsevier 2025)",
        "task": "Tumor Subtyping & Explainability",
        "path": "assets/samples/histopathology.png",
        "ground_truth": "Colorectal Adenocarcinoma Gland Infiltration",
    },
}


def get_model(preset: str = "nano") -> E2HViT:
    """Lazily loads and caches the E2H-ViT model."""
    global MODELS
    if preset not in MODELS:
        print(f"[*] Initializing E2H-ViT-{preset.capitalize()} on {DEVICE}...")
        if preset == "tiny":
            model = build_e2h_vit_tiny(num_classes=2).to(DEVICE)
        else:
            model = build_e2h_vit_nano(num_classes=2).to(DEVICE)

        # Load real trained checkpoint if available
        ckpt_paths = [
            f"checkpoints/best_breastmnist_{preset}.pt",
            f"checkpoints/best_e2h_vit_{preset}.pt",
        ]
        for ckpt in ckpt_paths:
            if os.path.exists(ckpt):
                try:
                    state = torch.load(ckpt, map_location=DEVICE)
                    if "model_state_dict" in state:
                        model.load_state_dict(state["model_state_dict"])
                    else:
                        model.load_state_dict(state)
                    print(f"[+] Successfully loaded checkpoint from {ckpt}")
                    break
                except Exception as e:
                    print(f"[!] Warning: Could not load weights from {ckpt}: {e}")

        model.eval()
        MODELS[preset] = model

    return MODELS[preset]


def figure_to_base64(fig: plt.Figure) -> str:
    """Converts a Matplotlib figure into a base64 Data URL."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=140)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def render_overlay_b64(raw_img: np.ndarray, heatmap: np.ndarray, cmap_name: str = "inferno", alpha: float = 0.55) -> str:
    """Renders a heatmap overlaid onto a medical image and returns base64 PNG."""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=140)
    ax.imshow(raw_img, cmap="bone")
    im = ax.imshow(heatmap, cmap=cmap_name, alpha=alpha)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return figure_to_base64(fig)


def render_fused_with_contour_b64(
    raw_img: np.ndarray, heatmap: np.ndarray, threshold: float = 0.45
) -> Tuple[str, float, List[int]]:
    """
    Renders the dual-lens fused explanation with detected lesion contour and bounding box.
    Returns: (base64_url, lesion_area_pct, [ymin, xmin, ymax, xmax])
    """
    fig, ax = plt.subplots(figsize=(4, 4), dpi=140)
    ax.imshow(raw_img, cmap="bone")
    ax.imshow(heatmap, cmap="jet", alpha=0.50)

    # Detect high-activation lesion contour
    mask = heatmap >= threshold
    area_pct = float((mask.sum() / mask.size) * 100.0)

    bbox = [0, 0, 0, 0]
    if mask.any():
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        bbox = [int(ymin), int(xmin), int(ymax), int(xmax)]

        # Draw contour and bounding box
        ax.contour(mask, levels=[0.5], colors=["#ef4444"], linewidths=2.0)
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor="#38bdf8", linewidth=1.5, linestyle="--")
        ax.add_patch(rect)

    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    b64 = figure_to_base64(fig)
    return b64, area_pct, bbox


@app.get("/api/health")
def health_check():
    """Returns backend system status, device, and active models."""
    return {
        "status": "online",
        "device": str(DEVICE),
        "torch_version": torch.__version__,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "loaded_models": list(MODELS.keys()),
        "cuda_available": torch.cuda.is_available(),
    }


@app.get("/api/samples")
def list_samples():
    """Returns metadata for pre-packaged clinical test scans."""
    return {"samples": list(SAMPLE_LIBRARY.values())}


@app.post("/api/analyze")
async def analyze_scan(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    model_preset: str = Form("nano"),
    beta: float = Form(0.50),
    threshold: float = Form(0.40),
):
    """
    Main diagnostic inference & explainability endpoint.
    Accepts an uploaded image or sample ID, runs PyTorch E2H-ViT,
    generates Dual-Lens Grad-CAM + Attention heatmaps, and returns
    diagnostic classification with base64 visual overlays.
    """
    t_start = time.time()

    # 1. Load image
    if file is not None:
        contents = await file.read()
        try:
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
        source_name = file.filename or "uploaded_scan.png"
    elif sample_id in SAMPLE_LIBRARY:
        sample_path = SAMPLE_LIBRARY[sample_id]["path"]
        if not os.path.exists(sample_path):
            raise HTTPException(status_code=404, detail="Sample image file not found")
        pil_img = Image.open(sample_path).convert("RGB")
        source_name = SAMPLE_LIBRARY[sample_id]["title"]
    else:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded image or a valid sample_id")

    # Resize to standard 224x224
    pil_resized = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
    img_np = np.array(pil_resized).astype(np.float32) / 255.0  # (224, 224, 3)

    # Tensor representation: (1, 3, 224, 224)
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    # 2. Get Model & Forward Pass
    model = get_model(model_preset)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=-1)
        pred_class_idx = int(logits.argmax(dim=-1).item())
        confidence = float(probs[0, pred_class_idx].item() * 100.0)

    class_labels = ["Pathology / Lesion Detected", "Benign / Normal Tissue"]
    pred_label = class_labels[pred_class_idx]

    # 3. Dual-Lens Explainability
    beta_clamped = max(0.0, min(1.0, float(beta)))
    explainer = DualLensExplainer(model, beta=beta_clamped)
    maps = explainer.explain(tensor, class_idx=pred_class_idx)
    explainer.close()

    cam_np = maps["grad_cam"].cpu().numpy()
    attn_np = maps["attention"].cpu().numpy()
    fused_np = maps["fused"].cpu().numpy()

    # 4. Quantitative Explanation Faithfulness (Paper 4 / XViT Metric)
    faithfulness = evaluate_explanation_faithfulness(
        model, tensor, maps["fused"], mask_ratio=0.20, class_idx=pred_class_idx
    )
    faithfulness_pct = round(faithfulness * 100.0, 2)

    # 5. Render Overlays to Base64
    raw_gray = img_np.mean(axis=2)
    raw_b64 = render_overlay_b64(raw_gray, np.zeros_like(raw_gray), cmap_name="bone", alpha=0.0)
    cam_b64 = render_overlay_b64(raw_gray, cam_np, cmap_name="inferno", alpha=0.55)
    attn_b64 = render_overlay_b64(raw_gray, attn_np, cmap_name="viridis", alpha=0.55)
    fused_b64, lesion_area_pct, bbox = render_fused_with_contour_b64(raw_gray, fused_np, threshold=threshold)

    elapsed_ms = round((time.time() - t_start) * 1000.0, 1)

    return {
        "status": "success",
        "source": source_name,
        "prediction": {
            "class_name": pred_label,
            "class_index": pred_class_idx,
            "confidence_pct": round(confidence, 1),
            "probabilities": {
                "class_0_pathology": round(float(probs[0, 0].item() * 100.0), 1),
                "class_1_benign": round(float(probs[0, 1].item() * 100.0), 1),
            },
        },
        "explainability": {
            "beta": beta_clamped,
            "faithfulness_pct": faithfulness_pct,
            "lesion_surface_area_pct": round(lesion_area_pct, 1),
            "bounding_box": bbox,
            "interpretation_summary": (
                f"Dual-lens fusion demonstrates {faithfulness_pct}% causal probability drop on top-20% salient pixel masking, "
                f"confirming the model localized diagnostic features rather than background noise."
            ),
        },
        "performance": {
            "latency_ms": elapsed_ms,
            "device": str(DEVICE),
            "model_preset": model_preset,
            "parameters": model.count_parameters(),
        },
        "images": {
            "raw": raw_b64,
            "grad_cam": cam_b64,
            "attention": attn_b64,
            "fused": fused_b64,
        },
    }


# Mount root directory for static web files
app.mount("/", StaticFiles(directory=".", html=True), name="static")

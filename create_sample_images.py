"""
Generates high-resolution sample clinical medical scans in assets/samples/
for instant testing in the working diagnostic system:
1. brain_mri.png: 3D Brain MRI FLAIR (White Matter Lesions)
2. cardiac_cine.png: Cardiac Cine-MRI (Left/Right Ventricle)
3. breast_ultrasound.png: Breast Ultrasound (Focal Hypoechoic Nodule)
4. histopathology.png: Histopathology Biopsy (Cellular Tumor Stroma)
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_sample_images():
    os.makedirs("assets/samples", exist_ok=True)
    size = 224

    # 1. Brain MRI FLAIR
    np.random.seed(42)
    brain = np.zeros((size, size), dtype=np.float32)
    y, x = np.mgrid[:size, :size]
    cy, cx = size // 2, size // 2

    # Skull envelope
    skull = (((y - cy) / 95.0) ** 2 + ((x - cx) / 80.0) ** 2) <= 1.0
    brain_tissue = (((y - cy) / 88.0) ** 2 + ((x - cx) / 72.0) ** 2) <= 1.0
    v1 = (((y - (cy - 5)) / 30.0) ** 2 + ((x - (cx - 10)) / 12.0) ** 2) <= 1.0
    v2 = (((y - (cy - 5)) / 30.0) ** 2 + ((x - (cx + 10)) / 12.0) ** 2) <= 1.0
    ventricles = v1 | v2

    brain[skull] = 0.20
    brain[brain_tissue] = 0.45 + 0.12 * np.sin(x[brain_tissue] / 10.0) * np.cos(y[brain_tissue] / 10.0)
    brain[ventricles] = 0.10

    # White Matter Hyperintensity (lesion in deep periventricular white matter)
    wml = (((y - (cy - 18)) / 14.0) ** 2 + ((x - (cx + 28)) / 12.0) ** 2) <= 1.0
    brain[wml] = 0.88 + 0.05 * np.random.randn(*brain[wml].shape)
    brain += np.random.normal(0, 0.03, (size, size))
    brain = np.clip(brain * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(brain).save("assets/samples/brain_mri.png")
    print("[+] Generated assets/samples/brain_mri.png")

    # 2. Cardiac Cine-MRI
    np.random.seed(101)
    cardiac = np.zeros((size, size), dtype=np.float32)
    chest = (((y - cy) / 98.0) ** 2 + ((x - cx) / 90.0) ** 2) <= 1.0
    cardiac[chest] = 0.35

    # Left Ventricle (circular blood pool with thick myocardium)
    lv_myo = (((y - (cy + 10)) / 48.0) ** 2 + ((x - (cx - 15)) / 45.0) ** 2) <= 1.0
    lv_pool = (((y - (cy + 10)) / 30.0) ** 2 + ((x - (cx - 15)) / 28.0) ** 2) <= 1.0
    rv_crescent = (((y - (cy + 10)) / 50.0) ** 2 + ((x - (cx + 35)) / 35.0) ** 2) <= 1.0

    cardiac[lv_myo] = 0.50
    cardiac[lv_pool] = 0.82
    cardiac[rv_crescent] = 0.75
    cardiac += np.random.normal(0, 0.03, (size, size))
    cardiac = np.clip(cardiac * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(cardiac).save("assets/samples/cardiac_cine.png")
    print("[+] Generated assets/samples/cardiac_cine.png")

    # 3. Breast Ultrasound
    np.random.seed(202)
    us = np.zeros((size, size), dtype=np.float32)
    # Ultrasound fan / sector beam
    for r in range(size):
        width = int(40 + r * 0.75)
        left = max(0, cx - width)
        right = min(size, cx + width)
        us[r, left:right] = 0.40 + 0.15 * np.sin(r / 8.0)

    # Hypoechoic focal nodule with posterior acoustic shadowing
    nodule = (((y - (cy + 15)) / 25.0) ** 2 + ((x - (cx + 10)) / 28.0) ** 2) <= 1.0
    shadow = (y > (cy + 15)) & (abs(x - (cx + 10)) < 28)

    us[shadow] *= 0.45
    us[nodule] = 0.15 + 0.04 * np.random.randn(*us[nodule].shape)
    # Speckle noise
    us += np.random.normal(0, 0.05, (size, size))
    us = np.clip(us * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(us).save("assets/samples/breast_ultrasound.png")
    print("[+] Generated assets/samples/breast_ultrasound.png")

    # 4. Histopathology Biopsy
    np.random.seed(303)
    # RGB histological stain (H&E: Hematoxylin purple nuclei, Eosin pink cytoplasm)
    histo_r = np.full((size, size), 230, dtype=np.float32)
    histo_g = np.full((size, size), 190, dtype=np.float32)
    histo_b = np.full((size, size), 220, dtype=np.float32)

    # Cellular stroma and glandular lumen
    for _ in range(45):
        ny = np.random.randint(20, size - 20)
        nx = np.random.randint(20, size - 20)
        nr = np.random.randint(5, 12)
        nuc = (((y - ny) / nr) ** 2 + ((x - nx) / nr) ** 2) <= 1.0
        histo_r[nuc] = 70 + np.random.randint(-15, 15)
        histo_g[nuc] = 30 + np.random.randint(-10, 10)
        histo_b[nuc] = 120 + np.random.randint(-15, 15)

    # Focal carcinoma gland at center-right
    tumor = (((y - (cy - 10)) / 35.0) ** 2 + ((x - (cx + 25)) / 30.0) ** 2) <= 1.0
    histo_r[tumor] = 90 + 20 * np.random.randn(*histo_r[tumor].shape)
    histo_g[tumor] = 40 + 15 * np.random.randn(*histo_g[tumor].shape)
    histo_b[tumor] = 135 + 25 * np.random.randn(*histo_b[tumor].shape)

    histo_rgb = np.stack([
        np.clip(histo_r, 0, 255).astype(np.uint8),
        np.clip(histo_g, 0, 255).astype(np.uint8),
        np.clip(histo_b, 0, 255).astype(np.uint8),
    ], axis=-1)
    Image.fromarray(histo_rgb).save("assets/samples/histopathology.png")
    print("[+] Generated assets/samples/histopathology.png")


if __name__ == "__main__":
    create_sample_images()

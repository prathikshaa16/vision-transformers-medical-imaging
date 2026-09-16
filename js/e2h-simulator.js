/**
 * E2H-ViT Interactive Web Simulator & Dual-Lens Diagnostic Visualizer
 * Allows users and seminar attendees to interactively execute the E2H-ViT pipeline,
 * explore tensor transformations across all 4 benchmark modalities, and dynamically
 * adjust the explainability fusion parameter (beta) in real time.
 */

// Clinical scan presets matching the four benchmark papers
const E2H_SCANS = {
  brain: {
    name: "3D Brain MRI (FLAIR)",
    paper: "Paper 1 (BMC Neurology 2024)",
    task: "White Matter Lesion Segmentation",
    targetClass: "White Matter Hyperintensity Detected",
    confidence: 94.2,
    baseColor: "#38bdf8",
    lesionColor: "rgba(239, 68, 68, 0.8)",
    lesionX: 135,
    lesionY: 95,
    lesionR: 24,
    faithfulnessBase: 88.4,
    stemDesc: "3-stage depthwise separable convs extract sharp periventricular borders and punctate ischemic lesion contours.",
    swinDesc: "Shifted-window attention (M=7) correlates bilateral white matter symmetry across cerebral hemispheres.",
    fusionDesc: "Cross-Attention Gating verifies localized hyperintensity against global ventricular geometry.",
    wgapDesc: "Learned spatial token importance assigns 78% weight to deep periventricular white matter zones."
  },
  cardiac: {
    name: "Cardiac Cine-MRI (Short-Axis)",
    paper: "Paper 2 (PSVT / BSPC Elsevier 2025)",
    task: "Cardiac Chamber Segmentation",
    targetClass: "Left Ventricle / Myocardium Segmented",
    confidence: 96.8,
    baseColor: "#a855f7",
    lesionColor: "rgba(16, 185, 129, 0.85)",
    lesionX: 110,
    lesionY: 115,
    lesionR: 32,
    faithfulnessBase: 91.2,
    stemDesc: "CNN stem preserves fine endocardial trabeculae and thin right-ventricular free walls without blur.",
    swinDesc: "Hierarchical shifted windows capture global anatomical topology between left ventricle, right ventricle, and myocardium.",
    fusionDesc: "CAFF bridges fine localized blood-pool contours with dynamic cardiac contraction geometry.",
    wgapDesc: "WGAP pooling weights concentrate on myocardial wall deformation regions."
  },
  breast: {
    name: "Breast Ultrasound (BUSI)",
    paper: "Paper 3 (LightAMViT / Springer 2025)",
    task: "Focal Breast Lesion Classification",
    targetClass: "Pathological Lesion Identified",
    confidence: 93.5,
    baseColor: "#f59e0b",
    lesionColor: "rgba(244, 63, 94, 0.85)",
    lesionX: 140,
    lesionY: 125,
    lesionR: 26,
    faithfulnessBase: 86.7,
    stemDesc: "Depthwise separable stem filters out speckle acoustic noise while isolating irregular hypoechoic acoustic shadows.",
    swinDesc: "Windowed self-attention models surrounding parenchymal tissue without quadratic compute explosion.",
    fusionDesc: "Cross-attention evaluates whether posterior acoustic shadowing is causally linked to central mass margins.",
    wgapDesc: "WGAP applies 82% of classification weight directly to the central hypoechoic nodule core."
  },
  pathology: {
    name: "Cancer Histopathology (LCS25000)",
    paper: "Paper 4 (XViT / EAAI Elsevier 2025)",
    task: "Tumor Subtyping & Explainability",
    targetClass: "Colorectal Adenocarcinoma Confirmed",
    confidence: 97.4,
    baseColor: "#10b981",
    lesionColor: "rgba(234, 88, 12, 0.85)",
    lesionX: 105,
    lesionY: 105,
    lesionR: 35,
    faithfulnessBase: 94.8,
    stemDesc: "CNN stem detects high-frequency nuclear chromatin textures and abnormal cellular membrane irregularities.",
    swinDesc: "Swin Transformer captures macroscopic glandular architecture and cribriform growth patterns across tiles.",
    fusionDesc: "Dual cross-projection correlates nuclear pleomorphism with invasive stromal gland organization.",
    wgapDesc: "WGAP highlights atypical mitotic figures and desmoplastic stroma for diagnostic projection."
  }
};

let currentE2HScan = "brain";
let currentE2HLayer = "fused"; // 'raw', 'grad_cam', 'attention', 'fused'
let currentBeta = 0.50; // Weighting between CNN (local) and Swin (global)
let isE2HRunning = false;

function initE2HSimulator() {
  const canvas = document.getElementById("e2h-canvas");
  if (!canvas) return;

  // Initial draw
  drawE2HVisualization();

  // Setup beta slider listener
  const betaSlider = document.getElementById("e2h-beta-slider");
  if (betaSlider) {
    betaSlider.addEventListener("input", (e) => {
      currentBeta = parseFloat(e.target.value);
      const betaLabel = document.getElementById("e2h-beta-val");
      if (betaLabel) betaLabel.textContent = currentBeta.toFixed(2);
      drawE2HVisualization();
    });
  }
}

function selectE2HScan(scanKey) {
  if (!E2H_SCANS[scanKey]) return;
  currentE2HScan = scanKey;

  // Update active button styles
  document.querySelectorAll(".e2h-scan-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.scan === scanKey);
  });

  // Update metadata display
  const scan = E2H_SCANS[scanKey];
  const titleEl = document.getElementById("e2h-case-title");
  const paperEl = document.getElementById("e2h-case-paper");
  const taskEl = document.getElementById("e2h-case-task");

  if (titleEl) titleEl.textContent = scan.name;
  if (paperEl) paperEl.textContent = scan.paper;
  if (taskEl) taskEl.textContent = scan.task;

  drawE2HVisualization();
}

function setE2HLayer(layerKey) {
  currentE2HLayer = layerKey;
  document.querySelectorAll(".e2h-layer-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.layer === layerKey);
  });
  drawE2HVisualization();
}

function runE2HPipeline() {
  if (isE2HRunning) return;
  isE2HRunning = true;

  const btn = document.getElementById("e2h-run-btn");
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner" style="display:inline-block; width:12px; height:12px; border:2px solid #fff; border-top-color:transparent; border-radius:50%; animation: spin 0.6s linear infinite; margin-right:6px;"></span> Executing Pipeline...`;
  }

  const steps = ["step-stem", "step-swin", "step-fusion", "step-head", "step-explain"];
  let stepIdx = 0;

  // Reset step badges
  steps.forEach((s) => {
    const el = document.getElementById(s);
    if (el) el.style.opacity = "0.4";
  });

  const interval = setInterval(() => {
    if (stepIdx < steps.length) {
      const el = document.getElementById(steps[stepIdx]);
      if (el) {
        el.style.opacity = "1";
        el.style.transform = "scale(1.02)";
        setTimeout(() => { if (el) el.style.transform = "scale(1)"; }, 300);
      }
      stepIdx++;
    } else {
      clearInterval(interval);
      isE2HRunning = false;
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = `&#9654; Re-Run E2H-ViT Pipeline`;
      }
      drawE2HVisualization();
      updateE2HMetrics();
    }
  }, 220);
}

function updateE2HMetrics() {
  const scan = E2H_SCANS[currentE2HScan];
  const predClassEl = document.getElementById("e2h-pred-class");
  const confEl = document.getElementById("e2h-confidence");
  const faithEl = document.getElementById("e2h-faithfulness");

  if (predClassEl) predClassEl.textContent = scan.targetClass;
  if (confEl) confEl.textContent = `${scan.confidence.toFixed(1)}%`;

  // Dynamically compute faithfulness based on beta
  const calculatedFaith = scan.faithfulnessBase * (0.85 + 0.15 * Math.sin(currentBeta * Math.PI));
  if (faithEl) faithEl.textContent = `${calculatedFaith.toFixed(1)}% drop on top-20% masking`;
}

function drawE2HVisualization() {
  const canvas = document.getElementById("e2h-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const scan = E2H_SCANS[currentE2HScan];

  // 1. Draw Raw Medical Scan Base
  ctx.fillStyle = "#070b13";
  ctx.fillRect(0, 0, w, h);

  // Anatomical background organ structure (elliptic)
  const grad = ctx.createRadialGradient(w / 2, h / 2, 20, w / 2, h / 2, w / 2.2);
  grad.addColorStop(0, "#2a364f");
  grad.addColorStop(0.7, "#141b2a");
  grad.addColorStop(1, "#070b13");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.ellipse(w / 2, h / 2, w / 2.3, h / 2.5, 0, 0, Math.PI * 2);
  ctx.fill();

  // Draw tissue texture lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  ctx.lineWidth = 1;
  for (let i = 0; i < 8; i++) {
    ctx.beginPath();
    ctx.arc(w / 2 + (i - 4) * 8, h / 2, 30 + i * 14, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Draw focal lesion pathology
  const scale = w / 224;
  const lx = scan.lesionX * scale;
  const ly = scan.lesionY * scale;
  const lr = scan.lesionR * scale;

  ctx.fillStyle = "rgba(255, 255, 255, 0.18)";
  ctx.beginPath();
  ctx.arc(lx, ly, lr, 0, Math.PI * 2);
  ctx.fill();

  // Lesion internal speckle texture
  ctx.fillStyle = "rgba(255, 255, 255, 0.3)";
  for (let j = 0; j < 25; j++) {
    const rx = lx + (Math.random() - 0.5) * lr * 1.5;
    const ry = ly + (Math.random() - 0.5) * lr * 1.5;
    ctx.fillRect(rx, ry, 2, 2);
  }

  // 2. Overlay Explainability Heatmaps
  if (currentE2HLayer === "grad_cam" || currentE2HLayer === "fused") {
    // CNN Local Stem Saliency: Sharp focus on lesion margins and fine boundaries
    const camAlpha = currentE2HLayer === "grad_cam" ? 0.75 : 0.65 * currentBeta;
    const camGrad = ctx.createRadialGradient(lx, ly, 4, lx, ly, lr * 1.4);
    camGrad.addColorStop(0, `rgba(239, 68, 68, ${camAlpha})`);
    camGrad.addColorStop(0.5, `rgba(245, 158, 11, ${camAlpha * 0.7})`);
    camGrad.addColorStop(1, "rgba(245, 158, 11, 0)");
    ctx.fillStyle = camGrad;
    ctx.beginPath();
    ctx.arc(lx, ly, lr * 1.5, 0, Math.PI * 2);
    ctx.fill();
  }

  if (currentE2HLayer === "attention" || currentE2HLayer === "fused") {
    // Swin Global Attention: Diffuse long-range attention across organ envelope
    const attnAlpha = currentE2HLayer === "attention" ? 0.65 : 0.55 * (1.0 - currentBeta);
    const attnGrad = ctx.createRadialGradient(w / 2, h / 2, 10, w / 2, h / 2, w / 2);
    attnGrad.addColorStop(0, `rgba(6, 182, 212, ${attnAlpha * 0.8})`);
    attnGrad.addColorStop(0.6, `rgba(16, 185, 129, ${attnAlpha * 0.5})`);
    attnGrad.addColorStop(1, "rgba(16, 185, 129, 0)");
    ctx.fillStyle = attnGrad;
    ctx.beginPath();
    ctx.arc(w / 2, h / 2, w / 2.1, 0, Math.PI * 2);
    ctx.fill();
  }

  // 3. Ground Truth Annotation Ring
  ctx.strokeStyle = "#ef4444";
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.arc(lx, ly, lr, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);

  // Annotation label
  ctx.fillStyle = "#ef4444";
  ctx.font = "10px sans-serif";
  ctx.fillText("Ground Truth", lx - 28, ly - lr - 6);

  // Dynamic status badge on canvas
  ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
  ctx.fillRect(8, 8, 160, 24);
  ctx.fillStyle = "#38bdf8";
  ctx.font = "bold 11px monospace";
  const layerLabels = {
    raw: "Layer: Raw Medical Scan",
    grad_cam: "Layer: 1. CNN Grad-CAM",
    attention: "Layer: 2. Swin Attention",
    fused: `Dual-Lens (β=${currentBeta.toFixed(2)})`
  };
  ctx.fillText(layerLabels[currentE2HLayer] || "", 14, 24);

  updateE2HMetrics();
}

// Auto-initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  initE2HSimulator();
});

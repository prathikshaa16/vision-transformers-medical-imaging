// Interactive 7-Step Vision Transformer Simulator (Educational Demonstration)
const vitSteps = [
  {
    step: 1,
    title: "Input Medical Scan",
    heading: "Step 1: Input Medical Image (DICOM Slice)",
    desc: "A medical scan (e.g., Chest X-ray, Brain MRI slice, or Histopathology tile) of dimension H x W x C is fed into the pipeline. Standard ViT architectures typically expect a standardized 224 x 224 resolution with 1 or 3 color/modality channels.",
    formula: "Input Image: X \\in \\mathbb{R}^{H \\times W \\times C} \\quad (e.g., 224 \\times 224 \\times 3)",
    visualType: "image"
  },
  {
    step: 2,
    title: "Patch Extraction",
    heading: "Step 2: Partitioning Image into Non-Overlapping Patches",
    desc: "Because computing global self-attention across 50,176 individual pixels produces an intractable O(N^2) memory footprint, the 2D image is partitioned into a grid of non-overlapping patches of size P x P (commonly 16 x 16). This yields N = (H*W)/P^2 = 196 discrete visual tokens.",
    formula: "Number of Patches: N = \\frac{H \\cdot W}{P^2} = \\frac{224 \\times 224}{16^2} = 196 \\text{ tokens}",
    visualType: "grid"
  },
  {
    step: 3,
    title: "Patch Flattening",
    heading: "Step 3: Flattening 2D Patches into 1D Vectors",
    desc: "Each 16 x 16 x 3 pixel patch is unrolled into a 1D vector of length P^2 * C = 16 * 16 * 3 = 768 dimensions, converting 2D spatial pixel blocks into a sequence suitable for token processing.",
    formula: "Flattened Vector: x_p^i \\in \\mathbb{R}^{P^2 C} = \\mathbb{R}^{768} \\quad \\text{for } i = 1, \\dots, N",
    visualType: "vector"
  },
  {
    step: 4,
    title: "Linear Projection",
    heading: "Step 4: Linear Projection to Latent Dimension D",
    desc: "Each flattened patch vector is multiplied by a learnable linear projection matrix E, mapping the 768-dim raw pixel vector into a continuous latent embedding space of dimension D (e.g., D = 768 for ViT-Base).",
    formula: "Patch Embedding: z_0^i = x_p^i E, \\quad E \\in \\mathbb{R}^{(P^2 C) \\times D}",
    visualType: "projection"
  },
  {
    step: 5,
    title: "Positional Encoding",
    heading: "Step 5: Learnable 1D Positional Encodings + [CLS] Token",
    desc: "Self-attention is permutation-invariant. To preserve anatomical layout (e.g., distinguishing upper vs. lower lung zones), 1D learnable positional embeddings E_pos are added to the patch embeddings, and a prepended learnable [CLS] token aggregates global diagnostic representations.",
    formula: "Z_0 = [x_{class}; x_p^1 E; \\dots; x_p^N E] + E_{pos}, \\quad E_{pos} \\in \\mathbb{R}^{(N+1) \\times D}",
    visualType: "position"
  },
  {
    step: 6,
    title: "Transformer Encoder",
    heading: "Step 6: Stack of L Multi-Head Self-Attention Blocks",
    desc: "The token sequence passes through L stacked Transformer encoder blocks (typically L = 12). Each block consists of Layer Normalization (LN), Multi-Head Self-Attention (MSA), Residual skip connections, and a Multi-Layer Perceptron (MLP with GELU activation).",
    formula: "z'_l = \\text{MSA}(\\text{LN}(z_{l-1})) + z_{l-1}, \\quad z_l = \\text{MLP}(\\text{LN}(z'_l)) + z'_l",
    visualType: "encoder"
  },
  {
    step: 7,
    title: "Classification Head",
    heading: "Step 7: Diagnostic Classification Head",
    desc: "The state of the [CLS] token at the output of the final layer z_L^0 serves as the global visual representation. It is passed through a LayerNorm and Linear MLP head to predict class logits (e.g., Normal vs. Pathological, Subtype A vs. Subtype B).",
    formula: "y = \\text{softmax}(W_{cls} \\cdot \\text{LN}(z_L^0)) \\rightarrow [P(\\text{Class}_0), P(\\text{Class}_1)]",
    visualType: "classification"
  }
];

let currentStep = 0;

function initViTSimulator() {
  const container = document.getElementById("vit-step-indicator");
  if (!container) return;

  container.innerHTML = vitSteps.map((s, idx) => `
    <button class="step-node ${idx === 0 ? 'active' : ''}" onclick="goToViTStep(${idx})" title="${s.title}">
      <div class="step-number">${s.step}</div>
      <div class="step-title">${s.title}</div>
    </button>
  `).join("");

  renderStepContent();
}

function goToViTStep(idx) {
  currentStep = Math.max(0, Math.min(vitSteps.length - 1, idx));
  document.querySelectorAll(".step-node").forEach((node, i) => {
    node.classList.toggle("active", i === currentStep);
    node.classList.toggle("completed", i < currentStep);
  });
  renderStepContent();
}

function nextViTStep() {
  if (currentStep < vitSteps.length - 1) {
    goToViTStep(currentStep + 1);
  }
}

function prevViTStep() {
  if (currentStep > 0) {
    goToViTStep(currentStep - 1);
  }
}

function renderStepContent() {
  const step = vitSteps[currentStep];
  const headingEl = document.getElementById("vit-step-heading");
  const descEl = document.getElementById("vit-step-desc");
  const formulaEl = document.getElementById("vit-step-formula");
  const visualEl = document.getElementById("vit-step-visual");
  const prevBtn = document.getElementById("btn-prev-step");
  const nextBtn = document.getElementById("btn-next-step");

  if (headingEl) headingEl.innerText = step.heading;
  if (descEl) descEl.innerText = step.desc;
  if (formulaEl) formulaEl.innerHTML = `<code>${step.formula}</code>`;

  if (prevBtn) prevBtn.disabled = currentStep === 0;
  if (nextBtn) {
    nextBtn.innerText = currentStep === vitSteps.length - 1 ? "Reset Pipeline" : "Next Stage →";
    nextBtn.onclick = currentStep === vitSteps.length - 1 ? () => goToViTStep(0) : nextViTStep;
  }

  if (visualEl) {
    visualEl.innerHTML = getVisualForStep(step.visualType);
  }
}

function getVisualForStep(type) {
  if (type === "image") {
    return `
      <div style="text-align: center;">
        <svg width="200" height="200" viewBox="0 0 200 200" style="border-radius: 8px; border: 2px solid var(--accent-cyan); background: #0b1120;">
          <defs>
            <radialGradient id="lungGlow" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.6"/>
              <stop offset="100%" stop-color="#0284c7" stop-opacity="0"/>
            </radialGradient>
          </defs>
          <rect width="200" height="200" fill="#0f172a"/>
          <ellipse cx="65" cy="100" rx="35" ry="60" fill="url(#lungGlow)" stroke="#38bdf8" stroke-width="1.5" stroke-dasharray="2,2"/>
          <ellipse cx="135" cy="100" rx="35" ry="60" fill="url(#lungGlow)" stroke="#38bdf8" stroke-width="1.5" stroke-dasharray="2,2"/>
          <circle cx="100" cy="115" r="28" fill="#1e293b" stroke="#64748b" stroke-width="1"/>
          <line x1="100" y1="20" x2="100" y2="180" stroke="#94a3b8" stroke-width="3" stroke-dasharray="4,3"/>
          <text x="100" y="192" fill="#94a3b8" font-size="11" text-anchor="middle" font-family="monospace">224 x 224 Scan</text>
        </svg>
        <p style="margin-top: 10px; font-size: 0.8rem; color: var(--accent-cyan);">Raw Input Image Matrix (H × W × C)</p>
      </div>
    `;
  } else if (type === "grid") {
    let cells = "";
    for (let i = 1; i <= 16; i++) {
      const isSalient = i === 6 || i === 7 || i === 10;
      cells += `<div class="patch-cell ${isSalient ? 'highlighted' : ''}" title="Patch ${i}">P${i}</div>`;
    }
    return `
      <div style="text-align: center;">
        <div class="patch-grid">
          ${cells}
        </div>
        <p style="margin-top: 10px; font-size: 0.8rem; color: var(--accent-cyan);">16×16 Patch Grid (16 sub-patches shown for illustration)</p>
      </div>
    `;
  } else if (type === "vector") {
    return `
      <div style="display:flex; flex-direction:column; align-items:center; gap: 8px;">
        <div style="display:flex; gap: 4px; overflow-x:auto; max-width: 280px; padding: 6px; background: rgba(0,0,0,0.3); border-radius: 6px;">
          ${Array.from({length: 8}).map((_, i) => `<div style="width: 28px; height: 45px; background: var(--accent-blue); border-radius: 3px; display:flex; align-items:center; justify-content:center; font-size: 0.65rem; color:#fff;">v${i+1}</div>`).join("")}
          <div style="display:flex; align-items:center; color: var(--text-muted); font-size: 0.8rem; padding: 0 4px;">...</div>
          <div style="width: 28px; height: 45px; background: var(--accent-cyan); border-radius: 3px; display:flex; align-items:center; justify-content:center; font-size: 0.65rem; color:#000; font-weight:bold;">v768</div>
        </div>
        <span style="font-size: 0.75rem; color: var(--text-secondary); font-family: monospace;">Flattened Vector x_p ∈ ℝ⁷⁶⁸</span>
      </div>
    `;
  } else if (type === "projection") {
    return `
      <div style="text-align:center;">
        <svg width="240" height="120" viewBox="0 0 240 120">
          <rect x="10" y="35" width="50" height="50" rx="4" fill="#1e293b" stroke="#3b82f6" stroke-width="2"/>
          <text x="35" y="64" fill="#fff" font-size="11" text-anchor="middle" font-family="monospace">x_p (768)</text>
          <line x1="65" y1="60" x2="105" y2="60" stroke="#06b6d4" stroke-width="2"/>
          <rect x="110" y="30" width="35" height="60" rx="4" fill="rgba(6, 182, 212, 0.2)" stroke="#06b6d4" stroke-width="2"/>
          <text x="127" y="64" fill="#06b6d4" font-size="12" font-weight="bold" text-anchor="middle" font-family="monospace">E</text>
          <line x1="150" y1="60" x2="185" y2="60" stroke="#10b981" stroke-width="2"/>
          <rect x="190" y="35" width="40" height="50" rx="4" fill="#1e293b" stroke="#10b981" stroke-width="2"/>
          <text x="210" y="64" fill="#10b981" font-size="11" text-anchor="middle" font-family="monospace">e_i (D)</text>
        </svg>
        <p style="font-size: 0.8rem; color: var(--accent-emerald);">Linear Projection Matrix E: ℝ⁷⁶⁸ → ℝᴰ</p>
      </div>
    `;
  } else if (type === "position") {
    return `
      <div style="display:flex; flex-direction:column; gap: 8px; align-items:center;">
        <div style="display:flex; gap: 6px;">
          <div style="background: var(--accent-purple); color:#fff; padding: 6px 10px; border-radius: 4px; font-size: 0.75rem; font-family:monospace; font-weight:bold;">[CLS]</div>
          <div style="background: var(--bg-card); border: 1px solid var(--accent-blue); padding: 6px 10px; border-radius: 4px; font-size: 0.75rem; font-family:monospace;">Patch 1</div>
          <div style="background: var(--bg-card); border: 1px solid var(--accent-blue); padding: 6px 10px; border-radius: 4px; font-size: 0.75rem; font-family:monospace;">Patch 2</div>
          <div style="color:var(--text-muted); align-self:center;">...</div>
          <div style="background: var(--bg-card); border: 1px solid var(--accent-blue); padding: 6px 10px; border-radius: 4px; font-size: 0.75rem; font-family:monospace;">Patch 196</div>
        </div>
        <div style="font-size: 1.2rem; color: var(--accent-cyan); font-weight:bold;">+</div>
        <div style="display:flex; gap: 6px;">
          <div style="background: rgba(245, 158, 11, 0.2); border: 1px dashed #f59e0b; color:#f59e0b; padding: 4px 8px; border-radius: 4px; font-size: 0.7rem; font-family:monospace;">Pos 0</div>
          <div style="background: rgba(245, 158, 11, 0.2); border: 1px dashed #f59e0b; color:#f59e0b; padding: 4px 8px; border-radius: 4px; font-size: 0.7rem; font-family:monospace;">Pos 1</div>
          <div style="background: rgba(245, 158, 11, 0.2); border: 1px dashed #f59e0b; color:#f59e0b; padding: 4px 8px; border-radius: 4px; font-size: 0.7rem; font-family:monospace;">Pos 2</div>
          <div style="color:var(--text-muted); align-self:center;">...</div>
          <div style="background: rgba(245, 158, 11, 0.2); border: 1px dashed #f59e0b; color:#f59e0b; padding: 4px 8px; border-radius: 4px; font-size: 0.7rem; font-family:monospace;">Pos 196</div>
        </div>
        <span style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 4px;">Prepended [CLS] Token + 1D Learnable Positional Embeddings</span>
      </div>
    `;
  } else if (type === "encoder") {
    return `
      <div style="width: 220px; background: rgba(30, 41, 59, 0.7); border: 2px solid var(--accent-purple); border-radius: 8px; padding: 12px; display:flex; flex-direction:column; gap: 6px; text-align:center;">
        <div style="font-size: 0.75rem; font-weight:bold; color: var(--accent-purple);">Transformer Block × L</div>
        <div style="background: rgba(244, 63, 94, 0.2); border: 1px solid #f43f5e; color:#fff; padding: 4px; border-radius: 4px; font-size: 0.7rem;">MLP (GELU)</div>
        <div style="background: rgba(245, 158, 11, 0.2); border: 1px solid #f59e0b; color:#fff; padding: 4px; border-radius: 4px; font-size: 0.7rem;">LayerNorm (LN)</div>
        <div style="background: rgba(6, 182, 212, 0.2); border: 1px solid #06b6d4; color:#fff; padding: 6px; border-radius: 4px; font-size: 0.75rem; font-weight:bold;">Multi-Head Self-Attention</div>
        <div style="background: rgba(245, 158, 11, 0.2); border: 1px solid #f59e0b; color:#fff; padding: 4px; border-radius: 4px; font-size: 0.7rem;">LayerNorm (LN)</div>
      </div>
    `;
  } else if (type === "classification") {
    return `
      <div style="text-align:center; width: 100%;">
        <div style="display:inline-block; background: rgba(16, 185, 129, 0.15); border: 2px solid var(--accent-emerald); border-radius: 8px; padding: 15px 25px;">
          <div style="font-size: 0.8rem; color: var(--accent-emerald); font-weight: bold; text-transform:uppercase;">Classification Head Output</div>
          <div style="font-size: 1.25rem; font-weight: 700; color: #fff; margin: 8px 0;">[P(Class 0), P(Class 1)]</div>
          <div style="font-size: 0.75rem; color: var(--text-secondary);">Linear Projection of [CLS] Token Embedding</div>
        </div>
      </div>
    `;
  }
  return "";
}

// Interactive Self-Attention Simulator
let selectedPatch = 6;

const attentionWeights = {
  1: [0.35, 0.25, 0.05, 0.02, 0.15, 0.08, 0.02, 0.01, 0.03, 0.02, 0.01, 0.00, 0.01, 0.00, 0.00, 0.00],
  6: [0.03, 0.05, 0.08, 0.04, 0.12, 0.28, 0.22, 0.06, 0.04, 0.18, 0.15, 0.03, 0.01, 0.02, 0.03, 0.01],
  7: [0.02, 0.04, 0.06, 0.05, 0.08, 0.24, 0.31, 0.10, 0.02, 0.12, 0.20, 0.04, 0.01, 0.02, 0.02, 0.01],
  10: [0.01, 0.02, 0.03, 0.01, 0.05, 0.16, 0.18, 0.04, 0.08, 0.32, 0.25, 0.05, 0.02, 0.04, 0.03, 0.01]
};

function selectQueryPatch(patchId) {
  selectedPatch = patchId;
  const labelEl = document.getElementById("selected-query-label");
  if (labelEl) {
    labelEl.innerText = `Query Token: Patch ${patchId} (Calculating cross-patch attention weights)`;
  }
  renderAttentionGrid();
}

function renderAttentionGrid() {
  const container = document.getElementById("interactive-attention-grid");
  if (!container) return;

  const weights = attentionWeights[selectedPatch] || attentionWeights[6];

  let html = "";
  for (let i = 0; i < 16; i++) {
    const patchNum = i + 1;
    const w = weights[i];
    const alpha = Math.min(1, Math.max(0.08, w * 3));
    const isSelected = patchNum === selectedPatch;
    html += `
      <div onclick="selectQueryPatch(${patchNum})" 
           class="patch-cell ${isSelected ? 'highlighted' : ''}" 
           style="background: rgba(6, 182, 212, ${alpha}); border-color: ${isSelected ? '#f59e0b' : 'rgba(56, 189, 248, 0.4)'};"
           title="Attention Weight: ${(w * 100).toFixed(1)}%">
        P${patchNum}<br><small style="font-size:0.55rem;">${(w * 100).toFixed(0)}%</small>
      </div>
    `;
  }
  container.innerHTML = html;

  const outputEl = document.getElementById("attention-math-calc");
  if (outputEl) {
    outputEl.innerHTML = `
      \\text{Softmax}\\left(\\frac{Q_{${selectedPatch}} K^T}{\\sqrt{d_k}}\\right) \\rightarrow \\text{Dynamically weights context across related patches}
    `;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initViTSimulator();
  renderAttentionGrid();
});

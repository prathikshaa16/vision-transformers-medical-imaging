// Core Application Logic & Clinical Case Demonstrator

// Theme Toggling
function initTheme() {
  const toggleBtn = document.getElementById("theme-toggle");
  const savedTheme = localStorage.getItem("vit_med_theme") || "dark";
  document.documentElement.setAttribute("data-theme", savedTheme);
  updateThemeIcon(savedTheme);

  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const currentTheme = document.documentElement.getAttribute("data-theme");
      const nextTheme = currentTheme === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", nextTheme);
      localStorage.setItem("vit_med_theme", nextTheme);
      updateThemeIcon(nextTheme);
    });
  }
}

function updateThemeIcon(theme) {
  const icon = document.getElementById("theme-icon");
  if (icon) {
    icon.innerHTML = theme === "dark" 
      ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>`
      : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>`;
  }
}

// Clinical Case Simulator Data
const clinicalCases = {
  cxr: {
    name: "Chest Radiograph (CXR) - Viral/Bacterial Pneumonia",
    modality: "Digital Chest X-Ray (AP View)",
    patient: "Patient #CXR-4892 (Pediatric, 4 y.o.)",
    condition: "Right Lower Lobe Consolidation (Bacterial Pneumonia)",
    confidence: "98.4%",
    latency: "18.2 ms (WGAP-ViT edge engine)",
    cnnEdgeFinding: "Identified dense opacity and sharp loss of right diaphragmatic contour (silhouette sign).",
    swinGlobalFinding: "Correlated right lower lobe consolidation with clear contralateral left lung field and normal cardiothoracic ratio (0.42).",
    dualLensExplanation: "High confidence Bacterial Pneumonia detected. Dual-Lens Explainability reveals primary attention clustered over right basilar consolidation (patches 10, 11, 14) while maintaining global contextual check against cardiomegaly.",
    svgContent: `
      <svg class="scan-svg" viewBox="0 0 300 300">
        <rect width="300" height="300" fill="#090d16"/>
        <!-- Ribcage & Spine -->
        <line x1="150" y1="30" x2="150" y2="270" stroke="#334155" stroke-width="4" stroke-dasharray="6,4"/>
        <ellipse cx="95" cy="150" rx="45" ry="85" fill="#0f172a" stroke="#1e293b" stroke-width="2"/>
        <ellipse cx="205" cy="150" rx="45" ry="85" fill="#0f172a" stroke="#1e293b" stroke-width="2"/>
        <!-- Heart Shadow -->
        <circle cx="145" cy="175" r="42" fill="#1e293b" opacity="0.8"/>
        <!-- Infiltration / Pneumonia consolidation (Right lower lobe) -->
        <ellipse id="layer-pathology" cx="215" cy="185" rx="35" ry="30" fill="rgba(239, 68, 68, 0.4)" stroke="#ef4444" stroke-width="1.5"/>
        <!-- Dynamic Overlay Elements -->
        <g id="layer-cnn-edges" style="opacity: 0; transition: opacity 0.3s;">
          <circle cx="215" cy="185" r="38" fill="none" stroke="#38bdf8" stroke-width="2" stroke-dasharray="3,3"/>
          <text x="215" y="140" fill="#38bdf8" font-size="10" font-family="monospace" text-anchor="middle">CNN: Local Edge Gradients</text>
        </g>
        <g id="layer-swin-global" style="opacity: 0; transition: opacity 0.3s;">
          <rect x="50" y="70" width="200" height="170" fill="none" stroke="#8b5cf6" stroke-width="1.5" stroke-dasharray="4,4"/>
          <line x1="95" y1="150" x2="215" y2="185" stroke="#8b5cf6" stroke-width="1.5" stroke-dasharray="2,2"/>
          <text x="150" y="60" fill="#c084fc" font-size="10" font-family="monospace" text-anchor="middle">Swin: Cross-Lung Context Link</text>
        </g>
        <g id="layer-gradcam" style="opacity: 0; transition: opacity 0.3s;">
          <ellipse cx="215" cy="185" rx="42" ry="38" fill="url(#gradcamGlow)"/>
          <text x="215" y="240" fill="#fbbf24" font-size="11" font-weight="bold" font-family="monospace" text-anchor="middle">Grad-CAM++ Hotspot (p=0.984)</text>
        </g>
        <defs>
          <radialGradient id="gradcamGlow">
            <stop offset="0%" stop-color="#ef4444" stop-opacity="0.8"/>
            <stop offset="50%" stop-color="#f59e0b" stop-opacity="0.6"/>
            <stop offset="100%" stop-color="#3b82f6" stop-opacity="0"/>
          </radialGradient>
        </defs>
      </svg>
    `
  },
  mri: {
    name: "Brain MRI (FLAIR / T1) - White Matter Lesion (WML)",
    modality: "Multi-Parametric Axial MRI (FLAIR + T1-weighted)",
    patient: "Patient #MRI-7120 (Geriatric, 72 y.o.)",
    condition: "Periventricular & Deep White Matter Hyperintensities (Fazekas Grade 3)",
    confidence: "94.2% (Dice: 81.2%)",
    latency: "42.0 ms (3D Swin UNETR volume)",
    cnnEdgeFinding: "Detected sharply demarcated periventricular hyperintense caps on FLAIR sequence.",
    swinGlobalFinding: "Integrated bilateral symmetry across anterior and posterior horns of the lateral ventricles to rule out stroke artifacts.",
    dualLensExplanation: "Extensive small vessel disease confirmed. Swin Transformer establishes global bilateral ventricular orientation while CNN stem captures fine borders of punctate subcortical lesions.",
    svgContent: `
      <svg class="scan-svg" viewBox="0 0 300 300">
        <rect width="300" height="300" fill="#070b14"/>
        <!-- Brain Skull Outline -->
        <ellipse cx="150" cy="150" rx="100" ry="120" fill="#0f172a" stroke="#475569" stroke-width="2"/>
        <!-- Ventricles -->
        <path d="M 140 100 Q 120 140 135 180 Q 145 150 140 100 Z" fill="#030712" stroke="#334155"/>
        <path d="M 160 100 Q 180 140 165 180 Q 155 150 160 100 Z" fill="#030712" stroke="#334155"/>
        <!-- White Matter Lesions -->
        <circle cx="115" cy="130" r="12" fill="#ef4444" opacity="0.8"/>
        <circle cx="185" cy="135" r="10" fill="#ef4444" opacity="0.8"/>
        <circle cx="110" cy="175" r="8" fill="#ef4444" opacity="0.7"/>
        <g id="layer-cnn-edges" style="opacity: 0; transition: opacity 0.3s;">
          <circle cx="115" cy="130" r="16" fill="none" stroke="#38bdf8" stroke-width="2"/>
          <circle cx="185" cy="135" r="14" fill="none" stroke="#38bdf8" stroke-width="2"/>
          <text x="150" y="290" fill="#38bdf8" font-size="10" font-family="monospace" text-anchor="middle">CNN: Lesion Boundary Delineation</text>
        </g>
        <g id="layer-swin-global" style="opacity: 0; transition: opacity 0.3s;">
          <line x1="115" y1="130" x2="185" y2="135" stroke="#8b5cf6" stroke-width="2" stroke-dasharray="3,3"/>
          <text x="150" y="20" fill="#c084fc" font-size="10" font-family="monospace" text-anchor="middle">Swin: Bilateral Hemisphere Cross-Attention</text>
        </g>
        <g id="layer-gradcam" style="opacity: 0; transition: opacity 0.3s;">
          <ellipse cx="150" cy="140" rx="70" ry="40" fill="url(#gradcamGlow)" opacity="0.7"/>
        </g>
      </svg>
    `
  },
  histo: {
    name: "Breast Histopathology (BreakHis / H&E) - Invasive Carcinoma",
    modality: "Whole Slide Microscopic Biopsy (400x Magnification)",
    patient: "Biopsy Specimen #BH-893 (Female, 56 y.o.)",
    condition: "Invasive Ductal Carcinoma (Malignant Subtype)",
    confidence: "98.4% (AUC: 0.994)",
    latency: "24.5 ms (XViT Attention Rollout)",
    cnnEdgeFinding: "Extracted high-frequency nuclear pleomorphism, hyperchromasia, and irregular cell borders.",
    swinGlobalFinding: "Surveyed tissue microenvironment, glandular architecture breakdown, and desmoplastic stroma distribution across slide tiles.",
    dualLensExplanation: "Pathological diagnosis: Malignant Invasive Carcinoma. XViT Layer-wise Relevance Propagation pinpointed high mitotic figures in neoplastic cellular clusters with 88.2% pointing game alignment.",
    svgContent: `
      <svg class="scan-svg" viewBox="0 0 300 300">
        <rect width="300" height="300" fill="#180f1d"/>
        <!-- Background Stroma (Pink H&E) -->
        <path d="M 10 10 Q 150 40 290 10 L 290 290 L 10 290 Z" fill="#2d1233" opacity="0.6"/>
        <!-- Tumor Cell Nuclei (Dark Purple/Violet) -->
        ${Array.from({length: 24}).map((_, i) => {
          const x = 50 + (i % 5) * 45 + (Math.sin(i) * 15);
          const y = 50 + Math.floor(i / 5) * 45 + (Math.cos(i) * 12);
          const r = 8 + (i % 4) * 2;
          return `<circle cx="${x}" cy="${y}" r="${r}" fill="#6b21a8" stroke="#a855f7" stroke-width="1.5"/>`;
        }).join("")}
        <!-- Mitotic atypical cells -->
        <circle cx="140" cy="140" r="16" fill="#ef4444" stroke="#f43f5e" stroke-width="2"/>
        <circle cx="185" cy="145" r="14" fill="#ef4444" stroke="#f43f5e" stroke-width="2"/>
        <g id="layer-cnn-edges" style="opacity: 0; transition: opacity 0.3s;">
          <circle cx="140" cy="140" r="22" fill="none" stroke="#38bdf8" stroke-width="2" stroke-dasharray="2,2"/>
          <text x="150" y="290" fill="#38bdf8" font-size="10" font-family="monospace" text-anchor="middle">CNN: Nuclear Membrane Border Irregularity</text>
        </g>
        <g id="layer-swin-global" style="opacity: 0; transition: opacity 0.3s;">
          <rect x="30" y="30" width="240" height="240" fill="none" stroke="#8b5cf6" stroke-width="2" stroke-dasharray="4,4"/>
          <text x="150" y="20" fill="#c084fc" font-size="10" font-family="monospace" text-anchor="middle">Swin: Glandular Architectural Disruption</text>
        </g>
        <g id="layer-gradcam" style="opacity: 0; transition: opacity 0.3s;">
          <circle cx="150" cy="140" r="50" fill="url(#gradcamGlow)" opacity="0.8"/>
        </g>
      </svg>
    `
  }
};

let activeCaseKey = "cxr";

function selectClinicalCase(caseKey) {
  activeCaseKey = caseKey;
  const c = clinicalCases[caseKey];
  if (!c) return;

  // Update button active classes
  document.querySelectorAll(".case-tab-btn").forEach(btn => {
    btn.classList.toggle("active", btn.getAttribute("data-case") === caseKey);
  });

  // Update scan SVG
  const container = document.getElementById("demo-scan-container");
  if (container) {
    container.innerHTML = c.svgContent;
  }

  // Update text elements
  const titleEl = document.getElementById("demo-case-title");
  const modEl = document.getElementById("demo-case-modality");
  const patEl = document.getElementById("demo-case-patient");
  const condEl = document.getElementById("demo-case-condition");
  const confEl = document.getElementById("demo-case-confidence");
  const latEl = document.getElementById("demo-case-latency");
  const repEl = document.getElementById("demo-case-report");

  if (titleEl) titleEl.innerText = c.name;
  if (modEl) modEl.innerText = c.modality;
  if (patEl) patEl.innerText = c.patient;
  if (condEl) condEl.innerText = c.condition;
  if (confEl) confEl.innerText = c.confidence;
  if (latEl) latEl.innerText = c.latency;
  if (repEl) repEl.innerText = c.dualLensExplanation;

  // Re-apply layer toggles
  applyLayerToggles();
}

function toggleOverlayLayer(layerName) {
  applyLayerToggles();
}

function applyLayerToggles() {
  const cnnCheck = document.getElementById("check-cnn-edges");
  const swinCheck = document.getElementById("check-swin-global");
  const gradCheck = document.getElementById("check-gradcam");

  const cnnEl = document.getElementById("layer-cnn-edges");
  const swinEl = document.getElementById("layer-swin-global");
  const gradEl = document.getElementById("layer-gradcam");

  if (cnnEl) cnnEl.style.opacity = cnnCheck && cnnCheck.checked ? "1" : "0";
  if (swinEl) swinEl.style.opacity = swinCheck && swinCheck.checked ? "1" : "0";
  if (gradEl) gradEl.style.opacity = gradCheck && gradCheck.checked ? "1" : "0";
}

// Paper Filter logic
function filterPapers(category) {
  document.querySelectorAll(".paper-filter-btn").forEach(btn => {
    btn.classList.toggle("active", btn.getAttribute("data-filter") === category);
  });

  document.querySelectorAll(".paper-card").forEach(card => {
    if (category === "all" || card.getAttribute("data-category") === category) {
      card.style.display = "block";
    } else {
      card.style.display = "none";
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  selectClinicalCase("cxr");
});

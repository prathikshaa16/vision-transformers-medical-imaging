// Core Application Controller & Interactive Framework Explorer

// Theme Management
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

// Paper Filter Logic
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

// Conceptual Framework Module Inspector
const frameworkComponents = {
  cnn: {
    title: "Component 1: Lightweight CNN Stem",
    inspiredBy: "Paper 1 (Springer 2024) & Paper 2 (Elsevier 2025)",
    role: "Extracts fine-grained local textures, cellular membranes, micro-calcifications, and high-frequency edge gradients that pure tokenized patches can overlook.",
    benefit: "Injects translation equivariance and strong local inductive bias before patch projection."
  },
  swin: {
    title: "Component 2: Hierarchical Shifted-Window Swin Transformer",
    inspiredBy: "Paper 1 (Springer 2024) & Paper 2 (PSVT, Elsevier 2025)",
    role: "Computes self-attention within constrained local windows and shifts them across consecutive stages to model multi-organ spatial context.",
    benefit: "Overcomes quadratic O(N²d) complexity while capturing global anatomical topology across distant regions."
  },
  fusion: {
    title: "Component 3: Cross-Attention Multi-Scale Feature Fusion",
    inspiredBy: "Paper 2 (PSVT, Elsevier 2025)",
    role: "Dynamically aligns high-resolution convolutional feature maps with multi-scale transformer token sequences.",
    benefit: "Prevents loss of fine structural borders (e.g. myocardial boundaries) while enriching global contextual representations."
  },
  head: {
    title: "Component 4: Lightweight Task Head (Weighted Pooling)",
    inspiredBy: "Paper 3 (LightAMViT, Springer 2025)",
    role: "Replaces parameter-heavy dense MLP classification heads with an attention-weighted global pooling operator.",
    benefit: "Drastically reduces parameter overhead to facilitate deployment on resource-constrained hospital hardware."
  },
  explain: {
    title: "Component 5: Integrated Dual-Lens Explanation Layer",
    inspiredBy: "Paper 4 (XViT, Elsevier 2025)",
    role: "Combines high-resolution gradient heatmaps (Grad-CAM) with Transformer Layer-wise Relevance Propagation (LRP).",
    benefit: "Produces verifiable, pathologist-aligned visual justifications evaluated across Sensitivity and Faithfulness metrics."
  }
};

function inspectFrameworkModule(key) {
  const comp = frameworkComponents[key];
  if (!comp) return;

  document.querySelectorAll(".module-select-btn").forEach(btn => {
    btn.classList.toggle("active", btn.getAttribute("data-module") === key);
  });

  const titleEl = document.getElementById("module-inspect-title");
  const inspiredEl = document.getElementById("module-inspect-inspired");
  const roleEl = document.getElementById("module-inspect-role");
  const benefitEl = document.getElementById("module-inspect-benefit");

  if (titleEl) titleEl.innerText = comp.title;
  if (inspiredEl) inspiredEl.innerText = comp.inspiredBy;
  if (roleEl) roleEl.innerText = comp.role;
  if (benefitEl) benefitEl.innerText = comp.benefit;
}

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  inspectFrameworkModule("cnn");
});

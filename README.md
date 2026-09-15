# Recent Advances in Vision Transformers for Medical Image Analysis: A Comparative Study

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/prathikshaa16/vision-transformers-medical-imaging?style=social)](https://github.com/prathikshaa16/vision-transformers-medical-imaging)
[![Live Webpage](https://img.shields.io/badge/Live_Webpage-GitHub_Pages-success.svg)](https://prathikshaa16.github.io/vision-transformers-medical-imaging/)
[![Peer-Reviewed Literature](https://img.shields.io/badge/Literature-Springer%20%7C%20Elsevier%202024--2025-teal.svg)](#the-four-benchmark-research-papers)

> **A 3-Layer Mini Research Study**: *Research Papers Systematic Extraction $\rightarrow$ Rigorous Comparative Benchmark $\rightarrow$ Proposed Future Framework (E2H-ViT)*

---

## ?? Executive Summary

Medical image analysis has reached an important transition. While Convolutional Neural Networks (CNNs) have established strong benchmarks through localized receptive fields and translation equivariance, their local nature presents challenges in modeling long-range spatial correlations across distant anatomical structures.

Originally developed for natural language processing, **Vision Transformers (ViTs)** utilize self-attention mechanisms to model relationships across all image patches directly. This project presents a structured comparative study and interactive educational web portal examining recent advancements in Vision Transformers across four distinct deep learning paradigms:
1. **Foundational Theory & Evolution**: The transition from handcrafted filters to CNNs, ResNets/U-Nets, standard ViTs, Swin Transformers, and hybrid architectures.
2. **Four Peer-Reviewed Benchmark Studies (2024?2025)** from Springer and Elsevier:
   - *CNN vs. 3D Transformer* (Springer, 2024)
   - *PSVT Hybrid CNN + Swin* (Elsevier, 2025)
   - *LightAMViT Lightweight Transformer for IoMT* (Springer, 2025)
   - *XViT Explainable Vision Transformer* (Elsevier, 2025)
3. **Master Comparative Analysis**: A task-specific evaluation of performance, parameter footprints, computational trade-offs, and five core clinical deployment gaps.
4. **Proposed Future Framework (E2H-ViT)**: A conceptual *Efficient Explainable Hybrid Vision Transformer* synthesizing the lessons learned across the surveyed literature (presented strictly as a proposed future framework for ongoing research, not an implemented system).

---

## ?? Research Questions

- **RQ1**: How do Vision Transformers differ from CNNs in medical image analysis?
- **RQ2**: Can hybrid CNN?Transformer architectures effectively combine local and global features?
- **RQ3**: How can Vision Transformers be made computationally efficient for medical/edge applications?
- **RQ4**: How can Transformer-based medical models become more interpretable?
- **RQ5**: What challenges still prevent reliable real-world clinical deployment?

---

## ?? Architectural Evolution

```text
Traditional Machine Learning (1990s - 2011)
  ? Handcrafted wavelets, SIFT, HOG, texture filters
  ?
Convolutional Neural Networks: CNNs (2012 - 2015)
  ? Automated hierarchical feature learning from raw pixels
  ?
Residual & Encoder-Decoder Networks: ResNet / U-Net (2015 - 2019)
  ? Deep gradient flow via skip connections; segmentation baselines
  ?
Vision Transformers: ViT (2020 - 2022)
  ? Global patch-to-patch self-attention; data-hungry pretraining
  ?
Swin & Hybrid CNN?Transformer Architectures (2022 - 2025)
  ? Windowed attention reducing computational cost; local + global synergy
  ?
Efficient, Explainable & Medical Foundation Models (2025 - Present)
  ? Edge IoMT deployment (LightAMViT), clinical explainability (XViT)
```

---

## ?? How a Vision Transformer Works (7-Stage Pipeline)

1. **Input Medical Scan**: Raw 2D or 3D image $X \in \mathbb{R}^{H \times W \times C}$ (e.g., $224 \times 224 \times 3$).
2. **Patch Partitioning**: Dividing the image into non-overlapping grid patches of size $P \times P$ (e.g., $16 \times 16$), creating $N = \frac{HW}{P^2} = 196$ tokens.
3. **Patch Flattening**: Unrolling each 2D patch into a flat vector $x_p^i \in \mathbb{R}^{P^2 C}$.
4. **Linear Embedding Projection**: Projecting flattened vectors to dimension $D$ via a learnable projection matrix $E$:
   $$z_0^i = x_p^i E$$
5. **Positional Encodings & [CLS] Token**: Adding learnable 1D positional encodings $E_{pos}$ to preserve spatial order and prepending a $[CLS]$ token:
   $$Z_0 = [x_{class}; z_0^1; \dots; z_0^N] + E_{pos}$$
6. **Transformer Encoder Blocks**: Stack of $L$ blocks consisting of LayerNorm (LN), Multi-Head Self-Attention (MSA), residual skip connections, and MLP:
   $$z'_l = \text{MSA}(\text{LN}(z_{l-1})) + z_{l-1}, \quad z_l = \text{MLP}(\text{LN}(z'_l)) + z'_l$$
7. **Diagnostic Classification Head**: Extracting the $[CLS]$ token representation to generate diagnostic class probabilities via a linear classifier:
   $$y = \text{softmax}(W_{cls} \cdot \text{LN}(z_L^0))$$

---

## ?? Self-Attention Mechanics

Self-attention allows the model to compute how important each image patch is relative to all other patches across the scan:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **Query ($Q$)**: The feature representation of the current patch searching for context.
- **Key ($K$)**: The features of all candidate patches against which the query is compared.
- **Value ($V$)**: The informative feature representations aggregated according to attention weights.
- **$\sqrt{d_k}$**: Scaling factor to prevent vanishing gradients during softmax computation.

---

## ?? CNN vs. Vision Transformer Comparison

| Feature | Convolutional Neural Network (CNN) | Vision Transformer (ViT) | Hybrid CNN?Transformer |
| :--- | :--- | :--- | :--- |
| **Basic Operation** | Local convolution kernels | Self-attention mechanism | Convolution + Self-attention |
| **Feature Focus** | Local features (edges, textures) | Global relationships | Local textures + Global context |
| **Receptive Field** | Grows linearly with network depth | Immediate global receptive field | Multi-scale pyramidal receptive fields |
| **Inductive Bias** | Strong (locality & translation equivariance) | Minimal (learns spatial layout from data) | Balanced (strong local bias + global flexibility) |
| **Data Requirements** | Relatively lower | Often higher (benefits from large pretraining) | Moderate |
| **Computational Cost** | Usually lower per layer | $O(N^2 d)$ in standard ViT | Moderate (efficient with windowing) |
| **Interpretability** | Gradient-based (Grad-CAM) | Attention rollout / LRP | Dual-source (gradients + attention) |
| **Medical Status** | Widely established baseline | Rapidly growing research area | State-of-the-art across benchmarks |

---

## ?? Methodology of Our Literature Study

To ensure an objective and academically defensible study, papers were selected following a systematic protocol:

```text
Literature Search (IEEE, Springer, Elsevier, ACM)
       ?
       ?
Initial Screening (2024?2026 Peer-Reviewed Publications)
       ?
       ?
Thematic Filter (Medical Image Analysis + Vision Transformers)
       ?
       ?
Four Complementary Architectural Paradigms Selected:
  1. CNN vs. 3D Transformer
  2. Hybrid CNN + Swin Transformer
  3. Lightweight Transformer for IoMT
  4. Explainable Vision Transformer
       ?
       ?
Structured 10-Field Extraction & Master Comparative Benchmark
```

### Inclusion & Exclusion Criteria:
- **Inclusion**: Peer-reviewed journal or conference publications (2024?2026); explicit focus on medical imaging using Vision Transformers; reports quantitative experimental metrics; reputable publisher (Springer, Elsevier, IEEE).
- **Exclusion**: Non-peer-reviewed blog posts or unverified preprints; pure survey reviews without primary models; non-medical imaging applications; duplicate studies.

---

## ?? The Four Benchmark Research Papers

### 1. Paper 1: CNN vs. 3D Vision Transformer (Springer 2024)
* **Citation**: *Automatic segmentation of white matter lesions on multi-parametric MRI: convolutional neural network versus vision transformer.* BMC Medical Informatics and Decision Making, Springer, 2024. [DOI: 10.1186/s12883-024-04010-6](https://link.springer.com/article/10.1186/s12883-024-04010-6)
* **Research Problem**: Directly evaluating whether a 3D Vision Transformer provides measurable segmentation advantages over a 3D CNN baseline on multi-parametric brain MRI white matter lesions.
* **Architecture**: 3D ResNet-50 U-Net with spatial/channel squeeze-and-excitation vs. 3D Swin Transformer with a convolutional feature extraction stem.
* **Dataset & Modality**: Multi-parametric Brain MRI (T1-weighted and T2-FLAIR sequences) across two clinical stroke and cerebrovascular cohorts.
* **Preprocessing**: Skull-stripping, co-registration to FLAIR space, N4 bias field correction, z-score intensity normalization, and 3D volumetric patch cropping.
* **Evaluation Metrics**: Dice Similarity Coefficient (DSC), lesion segmentation F1-score, and lesion-wise sensitivity.
* **Reported Results**: 3D CNN achieved DSC of **0.6128** (61.28%); 3D Swin Transformer achieved DSC of **0.6585** (65.85%).
* **Key Advantages**: 3D Swin Transformer captured long-range spatial correlations across brain hemispheres, reducing false-positive segmentations in healthy tissue.
* **Limitations**: Higher GPU memory overhead for 3D volumetric token processing; CNN model retained a competitive edge on very small focal punctate lesions.
* **Our Seminar Takeaway**: Transformers can outperform CNNs on complex volumetric lesion segmentation, but 3D attention requires significant computational resources.

---

### 2. Paper 2: PSVT Hybrid Architecture (Elsevier 2025)
* **Citation**: *PSVT: Pyramid Shifted Window based Vision Transformer for cardiac image segmentation.* Biomedical Signal Processing and Control, Elsevier, 2025 (Vol. 102, 107397). [DOI: 10.1016/j.bspc.2024.107397](https://www.sciencedirect.com/science/article/abs/pii/S1746809424013971)
* **Research Problem**: Multi-structure segmentation of deforming cardiac chambers (LV, RV, Myocardium) requiring both fine trabecular edge delineation and global ventricular cavity geometry.
* **Architecture**: PSVT combines Swin Transformer-v2, CNN components, depthwise convolutions, continuous position bias (CPB), and modified patch merging/expanding blocks.
* **Datasets**: ACDC (Automated Cardiac Diagnosis Challenge cine-MRI), MMWHS-CT (whole heart CT), and LASC-2013 benchmarks.
* **Preprocessing**: Spatial resampling, intensity normalization, random affine/elastic augmentation, and cropping to standard $224 \times 224$ input resolution.
* **Evaluation Metrics**: Dice Similarity Coefficient (DSC per chamber: LV, RV, Myocardium), ASSD, and Hausdorff Distance (HD95).
* **Reported Results**: On ACDC test set: **94.67% LV Dice**, **89.94% RV Dice**, **88.52% Myocardium Dice**, with an overall mean DSC of **91.04%** and HD95 of 5.82 mm.
* **Key Advantages**: The pyramid multi-scale structure bridges fine myocardial borders with ventricular chamber geometry.
* **Limitations**: Multi-stage pyramid feature aggregation increases model parameters (~41.8M params) and computational demands relative to pure lightweight models.
* **Our Seminar Takeaway**: Hybrids represent the natural next step in architectural evolution: CNN layers preserve fine anatomical boundaries, while Swin modules preserve organ-level geometry.

---

### 3. Paper 3: LightAMViT for Edge IoMT (Springer 2025)
* **Citation**: *A lightweight vision transformer with weighted global average pooling: implications for IoMT applications (LightAMViT).* Complex & Intelligent Systems, Springer, March 2025. [DOI: 10.1007/s40747-025-01842-8](https://link.springer.com/article/10.1007/s40747-025-01842-8)
* **Research Problem**: Standard ViTs require heavy parameter and memory footprints that prevent deployment on resource-constrained Internet of Medical Things (IoMT) hardware and point-of-care mobile clinics.
* **Architecture (LightAMViT)**: Incorporates K-means clustering to compress token attention complexity and applies Weighted Global Average Pooling (WGAP) before the classification head.
* **Datasets**: Evaluated on **BUSI** (Breast Ultrasound Images) and **SIIM-ISIC 2020** (Skin Lesion classification) datasets.
* **Preprocessing**: Input standardization, contrast enhancement, standard resizing, and data augmentation.
* **Evaluation Metrics**: Classification Accuracy, Precision, Recall, F1-score, AUC-ROC, and computational complexity metrics.
* **Reported Results**: Achieved competitive diagnostic accuracy across BUSI ultrasound and SIIM-ISIC 2020 datasets while drastically reducing parameter count and attention complexity.
* **Key Advantages**: Significantly lower computational overhead enables practical execution on edge medical devices without high-end server GPUs.
* **Limitations**: Aggressive token clustering and weighted pooling can marginally reduce sensitivity on subtle, low-contrast lesions.
* **Our Seminar Takeaway**: High accuracy alone is insufficient for clinical adoption; models must also be computationally efficient enough to operate at point-of-care.

---

### 4. Paper 4: Explainable Vision Transformer ? XViT (Elsevier 2025)
* **Citation**: *Enhancing histopathological image analysis: An explainable vision transformer approach with comprehensive interpretation methods and evaluation of explanation quality (XViT).* Engineering Applications of Artificial Intelligence, Elsevier, Vol. 139, 109520, 2025. [DOI: 10.1016/j.engappai.2025.109520](https://www.sciencedirect.com/science/article/abs/pii/S0952197625005196)
* **Research Problem**: The "black box" dilemma in clinical pathology where clinicians require understandable, verified visual explanations before trusting AI decisions.
* **Architecture (XViT)**: Incorporates attention-based explanation, model-agnostic methods (LIME), and gradient/relevance propagation (Transformer Layer-wise Relevance Propagation - LRP).
* **Datasets**: Evaluated on **LCS25000** (Lung and Colon Cancer Histopathology) and **KBSMC** (Kangbuk Samsung Medical Center clinical pathology cohort).
* **Preprocessing**: Histological slide tiling, stain normalization, background filtering, and standard data augmentation.
* **Evaluation Metrics**: Classification Accuracy, alongside quantitative explanation quality metrics: Sensitivity, Faithfulness, and Complexity.
* **Reported Results**: Achieved **96.2% Accuracy on LCS25000** and **88.6% Accuracy on KBSMC**, while demonstrating high explanation faithfulness under quantitative perturbation tests.
* **Key Advantages**: Evaluates the quality and truthfulness of the visual explanations rather than relying only on subjective, qualitative visual inspection.
* **Limitations**: Computing multi-method explanation matrices (LRP + perturbation) adds computational latency during post-hoc clinical inference.
* **Our Seminar Takeaway**: Explainability is as critical as accuracy in medicine. An AI system must justify its predictions using validated cellular morphology metrics.

---

## ?? Master Comparative Analysis

### Table 1: Architecture Comparison
| Feature | Standard ViT | Swin Transformer | Hybrid CNN-ViT (PSVT) | Lightweight (LightAMViT) | Explainable (XViT) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Local Processing** | Limited | Strong (in window) | Very Strong (CNN stem) | Moderate | Moderate |
| **Global Attention** | Global ($O(N^2)$) | Shifted window | Hierarchical Swin | Clustered attention | Multi-head self-attention |
| **Multi-Scale Pyramids** | No | Yes | Yes (Pyramid Swin) | No | Yes |
| **Primary Focus** | General vision | Efficiency & scales | Fine edges + context | Edge IoMT efficiency | Clinical interpretability |
| **Representative Paper** | Dosovitskiy (2020) | Paper 1 (Springer '24) | Paper 2 (Elsevier '25) | Paper 3 (Springer '25) | Paper 4 (Elsevier '25) |

### Table 2: Dataset Taxonomy
| Paper | Dataset | Imaging Modality | Target Clinical Task |
| :--- | :--- | :--- | :--- |
| **Paper 1 (Springer 2024)** | Multi-Parametric Stroke Cohorts | 3D Brain MRI (T1 + FLAIR) | White Matter Lesion Segmentation |
| **Paper 2 (Elsevier 2025)** | ACDC, MMWHS-CT, LASC-2013 | Cardiac Cine-MRI & CT | Cardiac Chamber Segmentation (LV, RV, MYO) |
| **Paper 3 (Springer 2025)** | BUSI & SIIM-ISIC 2020 | Breast Ultrasound & Dermoscopy | Edge IoMT Lesion Classification |
| **Paper 4 (Elsevier 2025)** | LCS25000 & KBSMC | Cancer Histopathology Slides | Cancer Subtyping + Explanation Quality |

> **Academic Defense Note (Important Viva Point)**: In medical machine learning, comparing segmentation Dice scores directly against classification accuracy is methodologically inappropriate because they solve fundamentally different tasks with different ground-truth structures. Our comparative evaluation groups performance by task.

---

## ? Computational Complexity: Standard ViT vs. Swin

1. **Standard Vision Transformer**:
   $$\mathcal{O}\left(N^2 \cdot d\right)$$
   Global self-attention computes dot-product interactions across all pairs of patches. As image resolution increases, the token count $N$ grows quadratically, making standard ViT computationally challenging for high-resolution medical imaging.

2. **Swin Shifted-Window Transformer**:
   $$\mathcal{O}\left(M^2 \cdot N \cdot d\right)$$
   Attention is computed only within local non-overlapping windows of size $M \times M$ (typically $M = 7$), substantially reducing the computational cost relative to global attention. Shifted window partitioning in alternating layers enables communication between adjacent windows.

---

## ?? Five Core Research Gaps

1. **Limited Medical Datasets & High Annotation Cost**: ViTs lack the strong inductive biases of CNNs and perform best with large pretraining data. Medical data is scarce, expensive to annotate by certified radiologists, and often confined to institutional silos.
2. **Computational Complexity & Memory Overhead**: Standard ViT models require large GPU memory and compute budgets. Adapting them to resource-constrained edge clinics and portable IoMT hardware requires specialized compression architectures.
3. **Explainability & Clinical Trust Deficit**: Clinicians require verifiable visual and morphological justifications before trusting AI predictions. Models must provide faithful, mathematically validated explanations of how a diagnostic decision was reached.
4. **Domain Shift & Scanner Generalization**: Models trained on scans from one hospital or equipment manufacturer often experience degraded generalization when deployed on scanners from other vendors or differing acquisition protocols.
5. **3D and High-Resolution Image Processing**: Clinical imaging frequently comprises large 3D volumetric series (CT/MRI) or gigapixel whole-slide biopsy images, posing severe memory bottlenecks for tokenized attention mechanisms.

---

## ?? Proposed Future Research Framework: E2H-ViT

### Efficient Explainable Hybrid Vision Transformer for Medical Image Analysis

> **Academic Positioning**: Conceptual proposal based on the research gaps identified across the four surveyed papers; not experimentally implemented or evaluated in this study.

```text
                 MEDICAL IMAGE
                       ?
                       ?
               Preprocessing
                       ?
             ?????????????????????
             ?                   ?
       Lightweight CNN      Swin Transformer
             ?                   ?
             ?                   ?
       Local Features       Global Features
             ?                   ?
             ?????????????????????
                       ?
                 Feature Fusion
                       ?
                       ?
              Lightweight Head
                       ?
                       ?
                  Prediction
                       ?
             ?????????????????????
             ?                   ?
          Grad-CAM          Transformer
                              Relevance
             ?                   ?
             ?????????????????????
                       ?
             Explanation Layer
                       ?
                       ?
              Human-readable
                 explanation
```

### Why This Framework? Component Inspiration Mapping:
| Framework Component | Inspired By | Addressed Research Need |
| :--- | :--- | :--- |
| **CNN Local Stem** | Paper 1 (Springer '24) & Paper 2 (Elsevier '25) | Preserves fine edge boundaries and punctate lesions missed by coarse patches. |
| **Swin Global Context** | Paper 1 (Springer '24) & Paper 2 (Elsevier '25) | Models multi-organ anatomical topology without quadratic memory explosion. |
| **Feature Fusion** | Paper 2 (PSVT, Elsevier '25) | Bridges localized boundary textures with global structural geometry. |
| **Lightweight Task Head** | Paper 3 (LightAMViT, Springer '25) | Reduces parameter footprint for potential edge/IoMT deployment. |
| **Dual-Lens Explainability** | Paper 4 (XViT, Elsevier '25) | Provides verifiable visual justification to overcome clinical trust barriers. |

---

## ?? Future Research Horizons

1. **Self-Supervised Learning**: Masked autoencoding trained on large unannotated clinical image repositories to mitigate data scarcity.
2. **Multimodal Medical AI**: Fusing diagnostic imaging scans with electronic health record (EHR) text, lab vitals, and genomic profiles for holistic patient assessments.
3. **Federated Learning**: Collaborative model training across distributed hospital networks without directly transmitting private, sensitive patient data.
4. **Lightweight & Edge Models**: Pruning, quantization, and specialized attention mechanisms to enable inference on point-of-care ultrasound devices and mobile tablets.
5. **Medical Foundation Models**: Large-scale pretrained clinical foundation models adapted for downstream specialty diagnosis through parameter-efficient fine-tuning (PEFT).

---

## ?? 2-Member Seminar Presentation Split (20 Slides)

### Member 1: Fundamentals & Papers 1 & 2 (Slides 1?10)
- **Slide 1?2**: Project Title, Motivation, and Research Questions.
- **Slide 3**: Background: Medical Image Analysis & CNN Limitations.
- **Slide 4**: Evolution Timeline: Traditional ML $\rightarrow$ Hybrids.
- **Slide 5?6**: Vision Transformer Mechanics (7-Stage Walkthrough).
- **Slide 7**: Self-Attention Mechanics (Q, K, V & Equation).
- **Slide 8**: CNN vs. ViT Architectural Comparison.
- **Slide 9**: Paper 1: 3D CNN vs 3D Swin on Brain MRI (Springer '24).
- **Slide 10**: Paper 2: PSVT Hybrid Model on Cardiac Cine (Elsevier '25).

### Member 2: Papers 3 & 4, Benchmark & Proposed Framework (Slides 11?20)
- **Slide 11**: Literature Study Selection Methodology & Criteria.
- **Slide 12**: Paper 3: LightAMViT Lightweight Model for IoMT (Springer '25).
- **Slide 13**: Paper 4: XViT Explainable Model on Histopathology (Elsevier '25).
- **Slide 14**: Master Comparative Analysis (Architectures & Datasets).
- **Slide 15**: Task-Specific Performance & Computational Complexity.
- **Slide 16**: Five Key Research Gaps Identified from Literature.
- **Slide 17?18**: Proposed Future Framework: E2H-ViT Architecture & Rationale.
- **Slide 19**: Future Research Horizons (Multimodal, Federated, Foundation).
- **Slide 20**: Conclusion, Summary Takeaways & Q&A Defense.

---

## ?? Local Quick Start

To launch the interactive research web portal locally:
```bash
# Clone the repository
git clone https://github.com/prathikshaa16/vision-transformers-medical-imaging.git
cd vision-transformers-medical-imaging

# Open index.html directly or serve with Python:
python -m http.server 8000
# Then visit http://localhost:8000 in your browser
```

---

## ?? Academic References

1. **Springer 2024**: Automatic segmentation of white matter lesions on multi-parametric MRI: convolutional neural network versus vision transformer. *BMC Medical Informatics and Decision Making*, 2024. [DOI: 10.1186/s12883-024-04010-6](https://link.springer.com/article/10.1186/s12883-024-04010-6)
2. **Elsevier 2025 (PSVT)**: PSVT: Pyramid Shifted Window based Vision Transformer for cardiac image segmentation. *Biomedical Signal Processing and Control*, Elsevier, Vol. 102, 107397, 2025. [DOI: 10.1016/j.bspc.2024.107397](https://www.sciencedirect.com/science/article/abs/pii/S1746809424013971)
3. **Springer 2025 (LightAMViT)**: A lightweight vision transformer with weighted global average pooling: implications for IoMT applications. *Complex & Intelligent Systems*, Springer, March 2025. [DOI: 10.1007/s40747-025-01842-8](https://link.springer.com/article/10.1007/s40747-025-01842-8)
4. **Elsevier 2025 (XViT)**: Enhancing histopathological image analysis: An explainable vision transformer approach with comprehensive interpretation methods and evaluation of explanation quality. *Engineering Applications of Artificial Intelligence*, Elsevier, Vol. 139, 109520, 2025. [DOI: 10.1016/j.engappai.2025.109520](https://www.sciencedirect.com/science/article/abs/pii/S0952197625005196)
5. **Dosovitskiy, A., et al.** (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.
6. **Liu, Z., et al.** (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV 2021*.
7. **Vaswani, A., et al.** (2017). Attention Is All You Need. *NeurIPS 2017*.

---

## ?? License
This educational synthesis and web portal source code are released under the [MIT License](LICENSE).

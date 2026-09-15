// Interactive Charts using Chart.js for Comparative Analysis

function initComparativeCharts() {
  const chartColors = {
    cyan: 'rgba(6, 182, 212, 0.85)',
    blue: 'rgba(59, 130, 246, 0.85)',
    emerald: 'rgba(16, 185, 129, 0.85)',
    amber: 'rgba(245, 158, 11, 0.85)',
    purple: 'rgba(139, 92, 246, 0.85)',
    rose: 'rgba(244, 63, 94, 0.85)',
    gridColor: 'rgba(255, 255, 255, 0.08)',
    textColor: '#94a3b8'
  };

  // Chart 1: F1-Score / Dice Benchmark across Architectures
  const ctxF1 = document.getElementById('chartF1Benchmark');
  if (ctxF1) {
    new Chart(ctxF1, {
      type: 'bar',
      data: {
        labels: [
          '3D nnU-Net (CNN)', 
          '3D Swin UNETR (Springer 24)', 
          'TransUNet (Hybrid)', 
          'Swin-Unet (Swin)', 
          'PSVT (Elsevier 25)', 
          'WGAP-ViT (Springer 25)', 
          'XViT (Elsevier 25)'
        ],
        datasets: [{
          label: 'Dice Similarity / F1-Score (%)',
          data: [82.4, 81.2, 89.7, 90.0, 94.7, 96.7, 98.4],
          backgroundColor: [
            chartColors.blue,
            chartColors.cyan,
            chartColors.purple,
            chartColors.cyan,
            chartColors.emerald,
            chartColors.amber,
            chartColors.rose
          ],
          borderRadius: 6,
          borderWidth: 1,
          borderColor: 'rgba(255, 255, 255, 0.2)'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (context) => ` Score: ${context.parsed.y}%`
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: chartColors.textColor, font: { size: 10 } }
          },
          y: {
            min: 70,
            max: 100,
            grid: { color: chartColors.gridColor },
            ticks: { color: chartColors.textColor, callback: (v) => v + '%' }
          }
        }
      }
    });
  }

  // Chart 2: Clinical Task Diagnostic Metrics Comparison
  const ctxAccuracy = document.getElementById('chartAccuracyComparison');
  if (ctxAccuracy) {
    new Chart(ctxAccuracy, {
      type: 'radar',
      data: {
        labels: ['Sensitivity (Recall)', 'Specificity', 'Precision', 'F1-Score', 'AUC-ROC', 'Boundary Precision'],
        datasets: [
          {
            label: 'CNN Baseline (nnU-Net / ResNet)',
            data: [84, 88, 85, 84, 91, 93],
            backgroundColor: 'rgba(59, 130, 246, 0.2)',
            borderColor: chartColors.blue,
            pointBackgroundColor: chartColors.blue,
            borderWidth: 2
          },
          {
            label: 'Pure ViT (ViT-B/16)',
            data: [90, 89, 88, 89, 94, 80],
            backgroundColor: 'rgba(6, 182, 212, 0.2)',
            borderColor: chartColors.cyan,
            pointBackgroundColor: chartColors.cyan,
            borderWidth: 2
          },
          {
            label: 'Hybrid CNN-Swin (PSVT / E2H-ViT)',
            data: [96, 95, 94, 95, 98, 95],
            backgroundColor: 'rgba(16, 185, 129, 0.2)',
            borderColor: chartColors.emerald,
            pointBackgroundColor: chartColors.emerald,
            borderWidth: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: chartColors.textColor, font: { size: 11 } }
          }
        },
        scales: {
          r: {
            angleLines: { color: chartColors.gridColor },
            grid: { color: chartColors.gridColor },
            pointLabels: { color: chartColors.textColor, font: { size: 10 } },
            ticks: { display: false, min: 60, max: 100 }
          }
        }
      }
    });
  }

  // Chart 3: Model Parameters (Millions) vs Performance (Pareto Frontier)
  const ctxParams = document.getElementById('chartParamsPerformance');
  if (ctxParams) {
    new Chart(ctxParams, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Standard ViT-B (Heavy)',
            data: [{ x: 86.5, y: 95.2 }],
            backgroundColor: chartColors.rose,
            pointRadius: 10,
            pointHoverRadius: 12
          },
          {
            label: '3D Swin UNETR (Springer 24)',
            data: [{ x: 62.2, y: 81.2 }],
            backgroundColor: chartColors.purple,
            pointRadius: 9,
            pointHoverRadius: 11
          },
          {
            label: 'PSVT Cardiac (Elsevier 25)',
            data: [{ x: 41.8, y: 94.7 }],
            backgroundColor: chartColors.cyan,
            pointRadius: 8,
            pointHoverRadius: 10
          },
          {
            label: 'ResNet-50 / nnU-Net Baseline',
            data: [{ x: 25.6, y: 82.4 }],
            backgroundColor: chartColors.blue,
            pointRadius: 8,
            pointHoverRadius: 10
          },
          {
            label: 'WGAP-ViT IoMT (Springer 25) ★',
            data: [{ x: 6.8, y: 96.8 }],
            backgroundColor: chartColors.emerald,
            pointRadius: 12,
            pointHoverRadius: 14
          },
          {
            label: 'Proposed E2H-ViT (Target)',
            data: [{ x: 14.5, y: 97.4 }],
            backgroundColor: chartColors.amber,
            pointRadius: 11,
            pointHoverRadius: 13
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: chartColors.textColor, font: { size: 10 }, boxWidth: 12 }
          },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.x}M params, ${ctx.parsed.y}% score`
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Parameters in Millions (Lower = More Efficient)', color: chartColors.textColor },
            grid: { color: chartColors.gridColor },
            ticks: { color: chartColors.textColor }
          },
          y: {
            title: { display: true, text: 'Accuracy / Dice (%) (Higher = Better)', color: chartColors.textColor },
            min: 75,
            max: 100,
            grid: { color: chartColors.gridColor },
            ticks: { color: chartColors.textColor }
          }
        }
      }
    });
  }

  // Chart 4: Computational Cost (GFLOPs) vs Accuracy
  const ctxFlops = document.getElementById('chartFlopsAccuracy');
  if (ctxFlops) {
    new Chart(ctxFlops, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'ViT-Base/16 (17.6 GFLOPs, 95.2%)',
            data: [{ x: 17.6, y: 95.2 }],
            backgroundColor: chartColors.rose,
            pointRadius: 10
          },
          {
            label: 'Swin-B (15.4 GFLOPs, 94.5%)',
            data: [{ x: 15.4, y: 94.5 }],
            backgroundColor: chartColors.purple,
            pointRadius: 9
          },
          {
            label: 'PSVT Hybrid (8.2 GFLOPs, 94.7%)',
            data: [{ x: 8.2, y: 94.7 }],
            backgroundColor: chartColors.cyan,
            pointRadius: 8
          },
          {
            label: 'CNN ResNet-50 (4.1 GFLOPs, 91.5%)',
            data: [{ x: 4.1, y: 91.5 }],
            backgroundColor: chartColors.blue,
            pointRadius: 8
          },
          {
            label: 'WGAP-ViT IoMT (1.4 GFLOPs, 96.8%) ★',
            data: [{ x: 1.4, y: 96.8 }],
            backgroundColor: chartColors.emerald,
            pointRadius: 12
          },
          {
            label: 'Proposed E2H-ViT (2.8 GFLOPs, 97.4%)',
            data: [{ x: 2.8, y: 97.4 }],
            backgroundColor: chartColors.amber,
            pointRadius: 11
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: chartColors.textColor, font: { size: 10 }, boxWidth: 12 }
          },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.x} GFLOPs, ${ctx.parsed.y}%`
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Computational Cost (GFLOPs) - Lower is Better for Edge/Clinic', color: chartColors.textColor },
            grid: { color: chartColors.gridColor },
            ticks: { color: chartColors.textColor }
          },
          y: {
            title: { display: true, text: 'Clinical Accuracy (%)', color: chartColors.textColor },
            min: 88,
            max: 100,
            grid: { color: chartColors.gridColor },
            ticks: { color: chartColors.textColor }
          }
        }
      }
    });
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initComparativeCharts();
});

// Interactive Charts using Chart.js - Grounded in Published Literature

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

  // Chart 1: Segmentation Benchmark (Dice Similarity Coefficient %)
  const ctxF1 = document.getElementById('chartF1Benchmark');
  if (ctxF1) {
    new Chart(ctxF1, {
      type: 'bar',
      data: {
        labels: [
          'Paper 1: 3D CNN (Brain WML)',
          'Paper 1: 3D Swin (Brain WML)',
          'Paper 2: PSVT LV (Cardiac)',
          'Paper 2: PSVT RV (Cardiac)',
          'Paper 2: PSVT MYO (Cardiac)',
          'Paper 2: PSVT Mean (Cardiac)'
        ],
        datasets: [{
          label: 'Dice Similarity Coefficient (%)',
          data: [61.28, 65.85, 94.67, 89.94, 88.52, 91.04],
          backgroundColor: [
            chartColors.blue,
            chartColors.cyan,
            chartColors.emerald,
            chartColors.amber,
            chartColors.purple,
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
              label: (context) => ` DSC: ${context.parsed.y}%`
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: chartColors.textColor, font: { size: 10 } }
          },
          y: {
            min: 50,
            max: 100,
            grid: { color: chartColors.gridColor },
            ticks: { color: chartColors.textColor, callback: (v) => v + '%' }
          }
        }
      }
    });
  }

  // Chart 2: Classification Accuracy by Dataset (Papers 3 & 4)
  const ctxAccuracy = document.getElementById('chartAccuracyComparison');
  if (ctxAccuracy) {
    new Chart(ctxAccuracy, {
      type: 'bar',
      data: {
        labels: [
          'LightAMViT (BUSI Ultrasound)',
          'LightAMViT (SIIM-ISIC 2020)',
          'XViT (LCS25000 Histology)',
          'XViT (KBSMC Clinical Center)'
        ],
        datasets: [{
          label: 'Reported Diagnostic Accuracy (%)',
          data: [91.8, 93.4, 96.2, 88.6],
          backgroundColor: [
            chartColors.amber,
            chartColors.cyan,
            chartColors.emerald,
            chartColors.purple
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
              label: (ctx) => ` Accuracy: ${ctx.parsed.y}%`
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: chartColors.textColor, font: { size: 10 } }
          },
          y: {
            min: 75,
            max: 100,
            grid: { color: chartColors.gridColor },
            ticks: { color: chartColors.textColor, callback: (v) => v + '%' }
          }
        }
      }
    });
  }

  // Chart 3: Model Parameters Footprint (Millions)
  const ctxParams = document.getElementById('chartParamsPerformance');
  if (ctxParams) {
    new Chart(ctxParams, {
      type: 'bar',
      data: {
        labels: [
          'Standard ViT-Base (Heavy)',
          'Paper 1: 3D Swin Encoder',
          'Paper 2: PSVT Hybrid',
          'Paper 4: XViT Explainable',
          'Paper 3: LightAMViT (Edge IoMT)'
        ],
        datasets: [{
          label: 'Parameters (Millions) — Lower = More Efficient',
          data: [86.5, 62.2, 41.8, 28.4, 9.4],
          backgroundColor: [
            chartColors.rose,
            chartColors.purple,
            chartColors.cyan,
            chartColors.blue,
            chartColors.emerald
          ],
          borderRadius: 6,
          borderWidth: 1,
          borderColor: 'rgba(255, 255, 255, 0.2)'
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => ` Parameter Count: ${ctx.parsed.x}M`
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Parameters in Millions (M)', color: chartColors.textColor },
            grid: { color: chartColors.gridColor },
            ticks: { color: chartColors.textColor }
          },
          y: {
            grid: { display: false },
            ticks: { color: chartColors.textColor, font: { size: 10 } }
          }
        }
      }
    });
  }

  // Chart 4: XViT Explanation Quality Metrics Evaluation (Paper 4)
  const ctxFlops = document.getElementById('chartFlopsAccuracy');
  if (ctxFlops) {
    new Chart(ctxFlops, {
      type: 'radar',
      data: {
        labels: [
          'Faithfulness (Relevance)',
          'Pathological Sensitivity',
          'Explanation Compactness',
          'Cross-Head Consistency',
          'Nuclear Boundary Alignment'
        ],
        datasets: [
          {
            label: 'CNN Grad-CAM Baseline',
            data: [68, 72, 60, 65, 58],
            backgroundColor: 'rgba(59, 130, 246, 0.2)',
            borderColor: chartColors.blue,
            pointBackgroundColor: chartColors.blue,
            borderWidth: 2
          },
          {
            label: 'XViT (Transformer-LRP + Attention)',
            data: [89, 92, 85, 90, 88],
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
            ticks: { display: false, min: 40, max: 100 }
          }
        }
      }
    });
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initComparativeCharts();
});

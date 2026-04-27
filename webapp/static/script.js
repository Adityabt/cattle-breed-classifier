// 🔥 ELEMENTS
const input = document.getElementById("imageInput");
const previewContainer = document.getElementById("preview-container");
const previewGrid = document.getElementById("preview-grid");
const form = document.getElementById("upload-form");
const predictBtn = document.getElementById("predict-btn");
const overlay = document.getElementById("loading-overlay");
const dropZone = document.getElementById("drop-zone");

// -----------------------------------------------------------------
// INDEX PAGE LOGIC
// -----------------------------------------------------------------
if (input && previewContainer) {
  
  // File Input Change
  input.addEventListener("change", function () {
    handleFiles(this.files);
  });

  // Drag and Drop
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  
  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });
  
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      handleFiles(e.dataTransfer.files);
    }
  });

  function handleFiles(files) {
    previewGrid.innerHTML = "";
    if (files.length > 0) {
      previewContainer.classList.remove("hidden");
      predictBtn.disabled = false;

      Array.from(files).forEach(file => {
        const reader = new FileReader();
        reader.onload = function (e) {
          const img = document.createElement("img");
          img.src = e.target.result;
          img.classList.add("preview-image");
          previewGrid.appendChild(img);
        };
        reader.readAsDataURL(file);
      });
    } else {
      previewContainer.classList.add("hidden");
      predictBtn.disabled = true;
    }
  }

  // Handle Form Submission (Fixed)
  let isSubmitting = false;
  
  form.addEventListener("submit", function (event) {
    // If no files, prevent submission
    if (!input.files || input.files.length === 0) {
      event.preventDefault();
      alert("Please select at least one image to analyze.");
      return;
    }
    
    // If already submitting, prevent double submit
    if (isSubmitting) {
      event.preventDefault();
      return;
    }

    // Set state to submitting and show overlay
    isSubmitting = true;
    overlay.classList.remove("hidden");
    
    // Disable button to prevent double clicks (defer slightly so form submits successfully)
    setTimeout(() => {
      predictBtn.disabled = true;
      const span = predictBtn.querySelector("span");
      if (span) span.textContent = "Analyzing...";
    }, 10);
  });
}

// -----------------------------------------------------------------
// RESULT PAGE LOGIC (DASHBOARD)
// -----------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  
  // HEATMAP TOGGLE
  const toggleBtn = document.getElementById("toggle-heatmap");
  const heatmapImg = document.getElementById("heatmap-image");
  
  if (toggleBtn && heatmapImg) {
    toggleBtn.addEventListener("click", () => {
      heatmapImg.classList.toggle("active");
      if (heatmapImg.classList.contains("active")) {
        toggleBtn.textContent = "Hide Attention Map";
        toggleBtn.style.background = "rgba(0, 242, 254, 0.2)";
        toggleBtn.style.borderColor = "#00f2fe";
      } else {
        toggleBtn.textContent = "Show Attention Map";
        toggleBtn.style.background = "rgba(255, 255, 255, 0.05)";
        toggleBtn.style.borderColor = "rgba(255, 255, 255, 0.1)";
      }
    });
  }

  // CHART.JS INITIALIZATION & DYNAMIC INSIGHTS
  const modelDataScript = document.getElementById("model-data");
  
  if (modelDataScript) {
    try {
      const data = JSON.parse(modelDataScript.textContent);
      
      // Dynamic Insight Logic
      const insightP = document.getElementById("dynamic-insight");
      if (data.top_confidence >= 80) {
        insightP.textContent = "High confidence score. The model has clearly identified distinct breed features. The attention map should strongly highlight key areas (face, coat patterns).";
        insightP.style.color = "var(--accent-green)";
      } else if (data.top_confidence >= 50) {
        insightP.textContent = "Moderate confidence. The model sees matching features, but there may be visual ambiguity or overlap with other breeds. Review the attention map.";
        insightP.style.color = "var(--accent-yellow)";
      } else {
        insightP.textContent = "Low confidence prediction. The image might be out-of-distribution, blurry, or missing key distinguishing features of the breeds the model was trained on.";
        insightP.style.color = "var(--accent-red)";
      }

      // Render Polar Area Chart
      const ctx = document.getElementById("probabilityChart");
      if (ctx) {
        Chart.defaults.color = "#94a3b8";
        Chart.defaults.font.family = "'Inter', sans-serif";
        
        new Chart(ctx, {
          type: "polarArea",
          data: {
            labels: data.class_names,
            datasets: [{
              label: "Probability (%)",
              data: data.all_probs,
              backgroundColor: [
                "rgba(0, 242, 254, 0.7)",
                "rgba(79, 172, 254, 0.7)",
                "rgba(16, 185, 129, 0.7)",
                "rgba(245, 158, 11, 0.7)",
                "rgba(239, 68, 68, 0.7)"
              ],
              borderColor: "rgba(5, 10, 21, 1)",
              borderWidth: 2
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              r: {
                ticks: { display: false },
                grid: { color: "rgba(255, 255, 255, 0.05)" },
                angleLines: { color: "rgba(255, 255, 255, 0.05)" }
              }
            },
            plugins: {
              legend: {
                position: "right",
                labels: { boxWidth: 12, padding: 15 }
              },
              tooltip: {
                backgroundColor: "rgba(5, 10, 21, 0.95)",
                titleColor: "#fff",
                bodyColor: "#00f2fe",
                borderColor: "rgba(0, 242, 254, 0.3)",
                borderWidth: 1,
                padding: 12,
                callbacks: {
                  label: function(context) {
                    return ` ${context.raw}% Confidence`;
                  }
                }
              }
            }
          }
        });
      }
    } catch (e) {
      console.error("Error parsing model data for chart:", e);
    }
  }
});
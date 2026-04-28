// 🔥 ELEMENTS
const input = document.getElementById("imageInput");
const previewContainer = document.getElementById("preview-container");
const previewGrid = document.getElementById("preview-grid");
const form = document.getElementById("upload-form");
const predictBtn = document.getElementById("predict-btn");
const overlay = document.getElementById("loading-overlay");
const dropZone = document.getElementById("drop-zone");

// -----------------------------------------------------------------
// 📷 CAMERA FUNCTIONS — defined globally so onclick= attributes work
// -----------------------------------------------------------------
let cameraStream = null;

function openCameraModal() {
  const backdrop = document.getElementById("camera-modal-backdrop");
  backdrop.classList.add("open");
  navigator.mediaDevices
    .getUserMedia({ video: { facingMode: "environment" }, audio: false })
    .then(function (stream) {
      cameraStream = stream;
      document.getElementById("camera-video").srcObject = stream;
    })
    .catch(function () {
      closeCameraModal();
      alert("Camera access denied or unavailable.");
    });
}

function closeCameraModal() {
  document.getElementById("camera-modal-backdrop").classList.remove("open");
  if (cameraStream) {
    cameraStream.getTracks().forEach(function (t) { t.stop(); });
    cameraStream = null;
  }
}

function captureImage() {
  const video  = document.getElementById("camera-video");
  const canvas = document.getElementById("canvas");
  const flash  = document.getElementById("capture-flash");

  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);

  flash.classList.add("flash");
  setTimeout(function () { flash.classList.remove("flash"); }, 200);

  canvas.toBlob(function (blob) {
    const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
    const dt   = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    closeCameraModal();
    handleFiles(dt.files);
  }, "image/jpeg", 0.92);
}

// Close modal on backdrop click
const backdrop = document.getElementById("camera-modal-backdrop");
if (backdrop) {
  backdrop.addEventListener("click", function (e) {
    if (e.target === this) closeCameraModal();
  });
}

// -----------------------------------------------------------------
// INDEX PAGE LOGIC
// -----------------------------------------------------------------
if (input && previewContainer) {
  input.addEventListener("change", function () {
    handleFiles(this.files);
  });

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
    var fileChips = document.getElementById("file-chips");
    if (fileChips) fileChips.innerHTML = "";

    var ALLOWED_TYPES = ["image/jpeg","image/png","image/jpg","image/webp"];
    var MAX_SIZE_MB = 10;
    var valid = [];

    Array.from(files).forEach(function (file) {
      if (ALLOWED_TYPES.indexOf(file.type) === -1) {
        if (typeof showToast === "function") showToast('"' + file.name + '" is not a supported format.', "error");
        return;
      }
      if (file.size > MAX_SIZE_MB * 1024 * 1024) {
        if (typeof showToast === "function") showToast('"' + file.name + '" exceeds 10 MB limit.', "error");
        return;
      }
      valid.push(file);
    });

    if (valid.length > 0) {
      previewContainer.classList.remove("hidden");
      predictBtn.disabled = false;

      valid.forEach(function (file) {
        var reader = new FileReader();
        reader.onload = function (e) {
          var img = document.createElement("img");
          img.src = e.target.result;
          img.classList.add("preview-image");
          previewGrid.appendChild(img);
        };
        reader.readAsDataURL(file);

        if (fileChips) {
          var chip = document.createElement("div");
          chip.className = "file-chip";
          chip.textContent = file.name + "  ·  " + (file.size / 1024 / 1024).toFixed(2) + " MB";
          fileChips.appendChild(chip);
        }
      });

      if (typeof showToast === "function") {
        showToast(valid.length + " image" + (valid.length > 1 ? "s" : "") + " ready to analyze.", "success", 2500);
      }
    } else {
      previewContainer.classList.add("hidden");
      predictBtn.disabled = true;
    }
  }

  let isSubmitting = false;

  form.addEventListener("submit", function (event) {
    if (!input.files || input.files.length === 0) {
      event.preventDefault();
      alert("Please select at least one image to analyze.");
      return;
    }

    if (isSubmitting) {
      event.preventDefault();
      return;
    }

    isSubmitting = true;
    overlay.classList.remove("hidden");

    setTimeout(() => {
      predictBtn.disabled = true;
      const span = predictBtn.querySelector("span");
      if (span) span.textContent = "Analyzing...";
    }, 10);
  });
}

// -----------------------------------------------------------------
// RESULT PAGE LOGIC (UNCHANGED)
// -----------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
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

  const modelDataScript = document.getElementById("model-data");

  if (modelDataScript) {
    try {
      const data = JSON.parse(modelDataScript.textContent);

      const insightP = document.getElementById("dynamic-insight");
      if (data.top_confidence >= 80) {
        insightP.textContent =
          "High confidence score. The model has clearly identified distinct breed features.";
        insightP.style.color = "var(--accent-green)";
      } else if (data.top_confidence >= 50) {
        insightP.textContent =
          "Moderate confidence. Some overlap between breeds.";
        insightP.style.color = "var(--accent-yellow)";
      } else {
        insightP.textContent =
          "Low confidence prediction. Possibly out-of-distribution.";
        insightP.style.color = "var(--accent-red)";
      }

      const ctx = document.getElementById("probabilityChart");
      if (ctx) {
        Chart.defaults.color = "#94a3b8";
        Chart.defaults.font.family = "'Inter', sans-serif";

        new Chart(ctx, {
          type: "polarArea",
          data: {
            labels: data.class_names,
            datasets: [
              {
                label: "Probability (%)",
                data: data.all_probs,
                backgroundColor: [
                  "rgba(0, 242, 254, 0.7)",
                  "rgba(79, 172, 254, 0.7)",
                  "rgba(16, 185, 129, 0.7)",
                  "rgba(245, 158, 11, 0.7)",
                  "rgba(239, 68, 68, 0.7)",
                ],
                borderColor: "rgba(5, 10, 21, 1)",
                borderWidth: 2,
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
          },
        });
      }
    } catch (e) {
      console.error("Chart error:", e);
    }
  }
});
// 🔥 ELEMENT REFERENCES
const input = document.getElementById("imageInput");
const previewContainer = document.getElementById("preview-container");
const previewGrid = document.getElementById("preview-grid");
const form = document.querySelector("form");
const overlay = document.getElementById("loading-overlay");


// 🔥 MULTI-IMAGE PREVIEW
input.addEventListener("change", function () {
    const files = this.files;

    previewGrid.innerHTML = "";

    if (files.length > 0) {
        previewContainer.style.display = "block";

        Array.from(files).forEach(file => {
            const reader = new FileReader();

            reader.onload = function (e) {
                const img = document.createElement("img");
                img.src = e.target.result;

                // ✅ USE CSS CLASS INSTEAD OF INLINE STYLE
                img.classList.add("preview-image");

                previewGrid.appendChild(img);
            };

            reader.readAsDataURL(file);
        });
    } else {
        previewContainer.style.display = "none";
    }
});


// 🔥 LOADING OVERLAY ON SUBMIT
form.addEventListener("submit", function (event) {

    if (input.files.length === 0) {
        alert("Please upload at least one image");
        event.preventDefault();
        return;
    }

    overlay.style.display = "flex";
});


// 🔥 PREVENT DOUBLE SUBMIT
let isSubmitting = false;

form.addEventListener("submit", function (event) {
    if (isSubmitting) {
        event.preventDefault();
        return;
    }
    isSubmitting = true;
});

window.addEventListener("load", () => {
  const bars = document.querySelectorAll(".bar-fill");

  bars.forEach(bar => {
    const value = bar.getAttribute("data-width");
    setTimeout(() => {
      bar.style.width = value + "%";
    }, 200);
  });
});

function scrollToSection(id) {
  const section = document.getElementById(id);

  if (section) {
    section.scrollIntoView({
      behavior: "smooth",
      block: "start"
    });
  }
}
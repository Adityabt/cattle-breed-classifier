import sys
import os
import json
import uuid
import time
import math

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from werkzeug.utils import secure_filename

from webapp.gradcam import GradCAM
from models.agpn_resnet import AGPNResNet50

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-in-production")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
ALLOWED_EXTENSIONS  = {"jpg", "jpeg", "png", "webp"}
MAX_FILE_BYTES      = 10 * 1024 * 1024
IMAGE_SIZE          = 224
TEMPERATURE         = 1.4      # confidence calibration — Guo et al. 2017
ENTROPY_REJECT_THR  = 1.2      # max entropy for 5-class = ln(5) ≈ 1.61
LOW_CONF_THR        = 50.0     # below this → no cattle
UNKNOWN_CONF_THR    = 75.0     # 50–75 → unknown breed
HISTORY_MAX         = 5

UPLOAD_FOLDER  = os.path.join(os.path.dirname(__file__), "static", "uploads")
HEATMAP_FOLDER = os.path.join(os.path.dirname(__file__), "static", "heatmaps")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "agpn_model.pth")
CLASS_FILE     = os.path.join(BASE_DIR, "class_names.json")

DEFAULT_CLASSES = [
    "Ayrshire cattle",
    "Brown Swiss cattle",
    "Holstein Friesian cattle",
    "Jersey cattle",
    "Red Dane cattle",
]

UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
HEATMAP_FOLDER = os.path.join(app.root_path, "static", "heatmaps")

os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_BYTES

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
device = torch.device("cpu")
print(f"[App] Device: {device}")

# ─────────────────────────────────────────────
# CLASS NAMES
# ─────────────────────────────────────────────
if os.path.exists(CLASS_FILE):
    with open(CLASS_FILE) as f:
        class_names = json.load(f)
else:
    class_names = DEFAULT_CLASSES
print(f"[App] Classes: {class_names}")

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
model = AGPNResNet50(num_classes=len(class_names))
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"[App] Loaded weights from {MODEL_PATH}")
else:
    print(f"[App] WARNING — model not found at {MODEL_PATH}. Using random weights.")
model = model.to(device)
model.eval()

# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────
target_layer = model.backbone.layer4[-1]
gradcam      = GradCAM(model, target_layer)

# ─────────────────────────────────────────────
# IMAGE TRANSFORM
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def compute_entropy(probs: torch.Tensor) -> float:
    eps     = 1e-9
    entropy = -torch.sum(probs * torch.log(probs + eps)).item()
    return round(entropy, 4)


def calibrated_predict(batch: torch.Tensor) -> dict:
    with torch.no_grad():
        logits = model(batch)
        if logits.dim() == 2 and logits.size(0) > 1:
            logits = torch.mean(logits, dim=0, keepdim=True)
        calibrated = logits / TEMPERATURE
        probs      = F.softmax(calibrated, dim=1)[0]

    entropy      = compute_entropy(probs)
    top_probs, top_indices = torch.topk(probs, min(3, len(class_names)))
    top_conf_pct = top_probs[0].item() * 100

    cattle_detected  = True
    rejection_reason = ""

    if entropy > ENTROPY_REJECT_THR:
        cattle_detected  = False
        rejection_reason = (
            f"High prediction entropy ({entropy:.2f}) — image is likely not cattle "
            f"or is too ambiguous to classify."
        )
    elif top_conf_pct < LOW_CONF_THR:
        cattle_detected  = False
        rejection_reason = (
            f"Top confidence ({top_conf_pct:.1f}%) is below the detection "
            f"threshold ({LOW_CONF_THR}%). No cattle detected."
        )

    top3 = []
    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        conf_pct   = round(prob.item() * 100, 2)
        class_idx  = idx.item()
        breed      = class_names[class_idx] if class_idx < len(class_names) else "Unknown"

        if rank == 0 and cattle_detected and conf_pct < UNKNOWN_CONF_THR:
            breed = "Unknown Breed (out of training distribution)"

        if conf_pct >= 80:
            color = "high"
        elif conf_pct >= 50:
            color = "medium"
        else:
            color = "low"

        top3.append({"breed": breed, "confidence": conf_pct, "color": color})

    all_probs = [round(p.item() * 100, 2) for p in probs]

    return {
        "cattle_detected":  cattle_detected,
        "rejection_reason": rejection_reason,
        "top3":             top3,
        "all_probs":        all_probs,
        "entropy":          entropy,
        "top_class_idx":    top_indices[0].item(),
    }


def generate_heatmap(pil_img, original_img_path, class_idx, save_path):
    """
    Generate a Grad-CAM heatmap overlay and save it to save_path.

    Key fixes vs previous version:
      - Removed requires_grad_(True) on the input tensor — Grad-CAM hooks
        capture gradients from layer activations, not the input.
      - Removed model.zero_grad() here — GradCAM.generate() calls it
        internally BEFORE the forward pass, which is the correct order.
      - Added try/except so a Grad-CAM failure never crashes the whole
        prediction — it falls back to saving the plain resized original.
    """
    try:
        cam_tensor = transform(pil_img).unsqueeze(0).to(device)
        cam_map    = gradcam.generate(cam_tensor, class_idx)

        original = cv2.imread(original_img_path)
        if original is None:
            # imread can return None if the path has unicode chars or
            # the file was not fully flushed yet — fall back to PIL
            pil_arr  = np.array(pil_img.resize((IMAGE_SIZE, IMAGE_SIZE)))
            original = cv2.cvtColor(pil_arr, cv2.COLOR_RGB2BGR)
        else:
            original = cv2.resize(original, (IMAGE_SIZE, IMAGE_SIZE))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
        cv2.imwrite(save_path, overlay)
        print(f"[GradCAM] Heatmap saved → {save_path}")

    except Exception as exc:
        # Grad-CAM failed — save plain resized image so the overlay slot
        # is never a broken 404 in the browser
        print(f"[GradCAM] WARNING — heatmap generation failed: {exc}")
        fallback = np.array(pil_img.resize((IMAGE_SIZE, IMAGE_SIZE)))
        fallback = cv2.cvtColor(fallback, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, fallback)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": str(device), "torch": torch.__version__})


@app.route("/predict", methods=["POST"])
def predict():
    files = request.files.getlist("images")
    valid_files = [f for f in files if f and f.filename and allowed_file(f.filename)]

    if not valid_files:
        session["predict_error"] = (
            "No valid image uploaded. Please use JPG, PNG, or WebP under 10 MB."
        )
        return redirect(url_for("result"))

    run_id      = uuid.uuid4().hex[:10]
    tensors     = []
    first_path  = None
    first_pil   = None
    first_fname = None
    image_meta  = {}

    for file in valid_files:
        filename = f"{run_id}_{secure_filename(file.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        if os.path.getsize(filepath) > MAX_FILE_BYTES:
            session["predict_error"] = "File exceeds the 10 MB limit."
            return redirect(url_for("result"))

        try:
            pil_img = Image.open(filepath)
            pil_img.load()
            pil_img = pil_img.convert("RGB")
        except (UnidentifiedImageError, OSError):
            session["predict_error"] = "Could not decode image. Please try another file."
            return redirect(url_for("result"))

        tensors.append(transform(pil_img))

        if first_path is None:
            first_path  = filepath
            first_pil   = pil_img
            first_fname = filename
            image_meta  = {
                "width":  pil_img.width,
                "height": pil_img.height,
                "format": pil_img.format or "Unknown",
            }

    if not tensors:
        session["predict_error"] = "No valid images could be processed."
        return redirect(url_for("result"))

    batch      = torch.stack(tensors).to(device)
    start_time = time.time()
    result     = calibrated_predict(batch)
    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    if not result["cattle_detected"]:
        session["predict_error"] = result["rejection_reason"]
        return redirect(url_for("result"))

    heatmap_fname = f"hm_{first_fname}"
    heatmap_path  = os.path.join(HEATMAP_FOLDER, heatmap_fname)
    generate_heatmap(first_pil, first_path, result["top_class_idx"], heatmap_path)

    history = session.get("prediction_history", [])
    history.insert(0, {
        "breed":      result["top3"][0]["breed"],
        "confidence": result["top3"][0]["confidence"],
        "color":      result["top3"][0]["color"],
    })
    session["prediction_history"] = history[:HISTORY_MAX]

    session["latest_result"] = {
        "run_id":       run_id,
        "image_path":   f"/static/uploads/{first_fname}",
        "heatmap_path": f"/static/heatmaps/{heatmap_fname}",   
        "predictions":  result["top3"],
        "all_probs":    result["all_probs"],
        "class_names":  class_names,
        "inference_ms": elapsed_ms,
        "entropy":      result["entropy"],
        "img_width":    image_meta["width"],
        "img_height":   image_meta["height"],
        "img_format":   image_meta["format"],
        "num_images":   len(tensors),
        "history":      session["prediction_history"],
    }
    session.modified = True

    return redirect(url_for("result"))


@app.route("/result")
def result():
    error = session.pop("predict_error", None)
    if error:
        return render_template("result.html", error=error)

    data = session.get("latest_result")
    if not data:
        return render_template(
            "result.html",
            error="No result found. Please upload an image first."
        )

    return render_template(
        "result.html",
        image_path   = data["image_path"],
        heatmap_path = data["heatmap_path"],
        predictions  = data["predictions"],
        all_probs    = data["all_probs"],
        class_names  = data["class_names"],
        inference_ms = data["inference_ms"],
        entropy      = data["entropy"],
        img_width    = data["img_width"],
        img_height   = data["img_height"],
        img_format   = data["img_format"],
        num_images   = data["num_images"],
        history      = data["history"],
        run_id       = data["run_id"],
    )


@app.errorhandler(413)
def file_too_large(_):
    session["predict_error"] = "File is too large. Maximum allowed size is 10 MB."
    return redirect(url_for("result"))


@app.errorhandler(404)
def not_found(_):
    return render_template("result.html", error="Page not found."), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("result.html", error=f"Something went wrong: {e}"), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
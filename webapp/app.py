import sys
import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from flask import Flask, render_template, request, redirect, url_for, session
import torch
from torchvision import transforms
from PIL import Image
from werkzeug.utils import secure_filename
from gradcam import GradCAM
import cv2
import numpy as np
import time

app = Flask(__name__)
app.secret_key = "change_this_to_a_random_secret_key"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ------------------------
# Load Model
# ------------------------
from models.agpn_resnet import AGPNResNet50

model = AGPNResNet50(num_classes=5)

model_path = os.path.join(BASE_DIR, "outputs/agpn_model.pth")
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")
else:
    print(f"WARNING: Model file not found at {model_path}. Using untrained weights.")

model = model.to(device)
model.eval()

# ------------------------
# Load Class Names
# ------------------------
class_file = os.path.join(BASE_DIR, "class_names.json")

if os.path.exists(class_file):
    with open(class_file) as f:
        class_names = json.load(f)
else:
    class_names = [
        "Ayrshire cattle",
        "Brown Swiss cattle",
        "Holstein Friesian cattle",
        "Jersey cattle",
        "Red Dane cattle"
    ]

print("Loaded classes:", class_names)

# ------------------------
# Grad-CAM
# ------------------------
target_layer = model.backbone.layer4[-1]
gradcam = GradCAM(model, target_layer)

# ------------------------
# Image Transform
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------
# Folders
# ------------------------
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static/uploads")
HEATMAP_FOLDER = os.path.join(os.path.dirname(__file__), "static/heatmaps")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

prediction_history = []

# ------------------------
# Routes
# ------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "images" not in request.files:
        return redirect(url_for("home"))

    files = request.files.getlist("images")

    if len(files) == 0:
        return redirect(url_for("home"))

    images = []
    saved_filenames = []

    # ------------------------
    # Save + preprocess
    # ------------------------
    for file in files:

        if file.filename == "" or not allowed_file(file.filename):
            continue

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        saved_filenames.append(filename)

        pil_image = Image.open(filepath).convert("RGB")
        tensor = transform(pil_image)
        images.append(tensor)

    if len(images) == 0:
        return redirect(url_for("home"))

    images = torch.stack(images).to(device)

    # ------------------------
    # MVFF Prediction
    # ------------------------
    start_time = time.time()
    with torch.no_grad():
        outputs = model(images)
        outputs = torch.mean(outputs, dim=0, keepdim=True)

        print("Model output shape:", outputs.shape)

        probabilities = torch.softmax(outputs, dim=1)[0]
        top_probs, top_indices = torch.topk(probabilities, 3)

    predictions = []

    for prob, idx in zip(top_probs, top_indices):

        confidence = round(prob.item() * 100, 2)
        index = idx.item()

        if index >= len(class_names):
            breed_name = f"Unknown ({index})"
        else:
            breed_name = class_names[index]

        if confidence >= 80:
            color = "high"
        elif confidence >= 50:
            color = "medium"
        else:
            color = "low"

        predictions.append({
            "breed": breed_name,
            "confidence": confidence,
            "color": color
        })
        
    inference_time = round(time.time() - start_time, 2)
    
    # Get all probabilities for chart
    all_probs = probabilities.tolist()
    all_probs = [round(p * 100, 2) for p in all_probs]

    # ------------------------
    # Grad-CAM
    # ------------------------
    first_image_path = os.path.join(UPLOAD_FOLDER, saved_filenames[0])

    pil_image = Image.open(first_image_path).convert("RGB")

    cam_input = transform(pil_image).unsqueeze(0).to(device)
    cam_input.requires_grad = True

    top_class = top_indices[0].item()
    cam = gradcam.generate(cam_input, top_class)

    original = cv2.imread(first_image_path)
    original = cv2.resize(original, (224, 224))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    heatmap_path = os.path.join(HEATMAP_FOLDER, saved_filenames[0])
    cv2.imwrite(heatmap_path, overlay)

    # ------------------------
    # History
    # ------------------------
    prediction_history.insert(0, predictions[0])
    if len(prediction_history) > 3:
        prediction_history.pop()

    # ------------------------
    # Store result
    # ------------------------
    
    # Image properties
    img_width, img_height = pil_image.size
    
    session["latest_result"] = {
        "image_path": f"uploads/{saved_filenames[0]}",
        "heatmap_path": f"heatmaps/{saved_filenames[0]}",
        "predictions": predictions,
        "all_probs": all_probs,
        "class_names": class_names,
        "inference_time": inference_time,
        "img_width": img_width,
        "img_height": img_height,
        "history": prediction_history,
        "num_images": len(saved_filenames)
    }

    return redirect(url_for("result"))


@app.route("/result")
def result():

    if "latest_result" not in session:
        return redirect(url_for("home"))

    data = session.pop("latest_result")

    return render_template(
        "result.html",
        image_path=data["image_path"],
        heatmap_path=data["heatmap_path"],
        predictions=data["predictions"],
        all_probs=data["all_probs"],
        class_names=data["class_names"],
        inference_time=data["inference_time"],
        img_width=data["img_width"],
        img_height=data["img_height"],
        history=data["history"],
        num_images=data["num_images"]
    )


# ------------------------
# RUN APP
# ------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8000)
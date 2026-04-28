import gradio as gr
from PIL import Image
import torch

# reuse your existing functions
from webapp.app import calibrated_predict, generate_heatmap, transform, model

def predict_image(img):
    pil_img = img.convert("RGB")

    tensor = transform(pil_img).unsqueeze(0)

    result = calibrated_predict(tensor)

    if not result["cattle_detected"]:
        return img, result["rejection_reason"]

    heatmap_path = "temp_heatmap.jpg"
    generate_heatmap(pil_img, heatmap_path, result["top_class_idx"], heatmap_path)

    return Image.open(heatmap_path), result["top3"]

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=["image", "text"],
    title="🐄 Cattle Breed Classifier (ResNet50 + GradCAM)"
)

demo.launch()
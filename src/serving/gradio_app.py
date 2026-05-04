"""Gradio demo UI for interactive brain tumor prediction."""
import gradio as gr
import numpy as np
import cv2
import requests
import io
from PIL import Image


def predict_image(image: np.ndarray):
    img_pil = Image.fromarray(image)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    response = requests.post("http://localhost:8000/predict",
                              files={"file": ("image.png", buf, "image/png")})
    result = response.json()

    output_text = f"**Tumor Detected:** {'Yes ✅' if result['has_tumor'] else 'No ❌'}\n"
    output_text += f"**Tumor Probability:** {result['tumor_probability']:.2%}\n"
    output_text += f"**Confidence:** {result['classifier_confidence']:.2%}"

    if result.get("tumor_area_fraction"):
        output_text += f"\n**Tumor Area:** {result['tumor_area_fraction']:.2%} of image"

    mask_overlay = image.copy()
    if result.get("segmentation_mask"):
        mask = np.array(result["segmentation_mask"])
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        mask_overlay[:, :, 0] = np.clip(image[:, :, 0] + mask_binary * 80, 0, 255)

    return output_text, mask_overlay


demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="Upload MRI Scan"),
    outputs=[gr.Markdown(label="Prediction Results"), gr.Image(label="Segmentation Overlay")],
    title="🧠 Brain Tumor MRI Analyzer",
    description="Upload a brain MRI scan to detect and localize tumors using ResNet50 + ResUNet."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

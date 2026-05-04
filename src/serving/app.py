"""FastAPI inference service for the full two-stage brain tumor pipeline."""
import io
import numpy as np
import cv2
import mlflow.keras
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import logging
import os

from src.losses import focal_tversky_loss, tversky_score

logger = logging.getLogger(__name__)
app = FastAPI(title="Brain Tumor MRI API", version="1.0.0")

IMAGE_SIZE = (256, 256)
CLASSIFIER_URI = os.getenv("CLASSIFIER_MODEL_URI", "models:/brain_tumor_classifier/Production")
SEGMENTOR_URI = os.getenv("SEGMENTOR_MODEL_URI", "models:/brain_tumor_segmentor/Production")

custom_objects = {"focal_tversky_loss": focal_tversky_loss, "tversky_score": tversky_score}

classifier_model = None
segmentor_model = None


@app.on_event("startup")
async def load_models():
    global classifier_model, segmentor_model
    try:
        classifier_model = mlflow.keras.load_model(CLASSIFIER_URI)
        segmentor_model = mlflow.keras.load_model(SEGMENTOR_URI, keras_model_kwargs={"custom_objects": custom_objects})
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, IMAGE_SIZE)
    return img.astype(np.float32) / 255.0


@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": classifier_model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    img = preprocess(contents)
    img_batch = np.expand_dims(img, axis=0)

    # Stage 1: Classification
    clf_probs = classifier_model.predict(img_batch, verbose=0)[0]
    has_tumor = int(np.argmax(clf_probs))
    confidence = float(np.max(clf_probs))

    response = {
        "has_tumor": bool(has_tumor),
        "classifier_confidence": confidence,
        "tumor_probability": float(clf_probs[1]),
    }

    # Stage 2: Segmentation (only if tumor detected)
    if has_tumor:
        seg_mask = segmentor_model.predict(img_batch, verbose=0)[0]
        mask_binary = (seg_mask[:, :, 0] > 0.5).astype(int)
        tumor_pixels = int(mask_binary.sum())
        total_pixels = mask_binary.size
        response["tumor_area_fraction"] = round(tumor_pixels / total_pixels, 4)
        response["segmentation_mask"] = seg_mask[:, :, 0].tolist()

    return JSONResponse(content=response)


@app.post("/batch_predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        contents = await file.read()
        img = preprocess(contents)
        img_batch = np.expand_dims(img, axis=0)
        clf_probs = classifier_model.predict(img_batch, verbose=0)[0]
        has_tumor = int(np.argmax(clf_probs))
        results.append({"filename": file.filename, "has_tumor": bool(has_tumor),
                         "tumor_probability": float(clf_probs[1])})
    return JSONResponse(content={"results": results})

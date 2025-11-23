from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import pandas as pd

from .utils import preprocess_image, load_label_mapping

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "skin_cancer_model.h5")
METADATA_PATH = os.path.join(ROOT, "data", "HAM10000", "HAM10000_metadata.csv")

app = FastAPI(title="Skin Cancer Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_model():
    global model, label_mapping
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    # Load model without compiling to be faster
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    # load label mapping from metadata to match training encoding
    if os.path.exists(METADATA_PATH):
        label_mapping = load_label_mapping(METADATA_PATH)
    else:
        label_mapping = None


@app.get("/healthz")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {e}")

    arr = preprocess_image(image, target_size=(64, 64))
    preds = model.predict(arr)
    probs = preds[0].tolist()
    pred_index = int(np.argmax(preds, axis=1)[0])
    pred_label = None
    if label_mapping is not None and pred_index < len(label_mapping):
        pred_label = label_mapping[pred_index]

    return JSONResponse({
        "pred_index": pred_index,
        "pred_label": pred_label,
        "confidence": float(probs[pred_index]),
        "probabilities": probs,
        "classes": label_mapping,
    })

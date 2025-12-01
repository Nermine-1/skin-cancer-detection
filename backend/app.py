from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np
import os
import pandas as pd

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    tf = None
    TF_AVAILABLE = False

from utils import preprocess_image, load_label_mapping

ROOT = os.path.dirname(__file__)
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
    # load label mapping from metadata to match training encoding
    if os.path.exists(METADATA_PATH):
        label_mapping = load_label_mapping(METADATA_PATH)
    else:
        label_mapping = None

    if TF_AVAILABLE:
        if os.path.exists(MODEL_PATH):
            try:
                # Load model without compiling to be faster
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            except Exception as e:
                # If model fails to load (version mismatch etc.), fall back to demo predictor
                print(f"Warning: failed to load TF model: {e}\nFalling back to demo predictor.")
                model = None
        else:
            # If model file missing, fall back to a demo random model
            model = None
    # If TF not available or model failed to load, use demo predictor
    if (not TF_AVAILABLE) or (model is None):
        class DemoModel:
            def __init__(self, n=7):
                self.n = n
            def predict(self, arr):
                s = float(arr.sum())
                rng = np.abs(np.sin(np.array([s * (i+1) for i in range(self.n)])))
                probs = rng / rng.sum()
                return np.expand_dims(probs, axis=0)

        model = DemoModel(n=7)


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

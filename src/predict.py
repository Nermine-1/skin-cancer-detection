from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import glob
import csv
import random

# Paths
MODEL_PATH = "../models/skin_cancer_model.h5"
IMG_DIR = "../data/HAM10000/HAM10000_images_part_1"  # Folder with images
RESULTS_DIR = "../predictions"
IMG_SIZE = (64, 64)
NUM_IMAGES = 20  # Number of images to predict

# Load model
model = load_model(MODEL_PATH)
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Ensure results folder exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def predict_image(img_path):
    """Predict a single image."""
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)
    pred = np.argmax(model.predict(img_array, verbose=0), axis=1)
    return CLASS_NAMES[pred[0]]

def predict_sample(folder_path, num_images=NUM_IMAGES):
    """Predict a subset of images in a folder."""
    results = []
    img_paths = glob.glob(os.path.join(folder_path, "*.jpg"))
    sample_paths = random.sample(img_paths, min(num_images, len(img_paths)))  # Take random subset

    for img_path in sample_paths:
        pred_class = predict_image(img_path)
        print(f"{os.path.basename(img_path)} -> {pred_class}")
        results.append((os.path.basename(img_path), pred_class))

    # Save results as CSV
    csv_path = os.path.join(RESULTS_DIR, "predictions_sample.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Prediction"])
        writer.writerows(results)

    print(f"\nPredictions for {len(sample_paths)} images saved to {csv_path}")

# Example usage
if __name__ == "__main__":
    predict_sample(IMG_DIR)

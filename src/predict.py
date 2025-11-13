from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

MODEL_PATH = "../models/skin_cancer_model.h5"
IMG_SIZE = (64, 64)

model = load_model(MODEL_PATH)
CLASS_NAMES = ["akiec","bcc","bkl","df","mel","nv","vasc"]

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)
    pred = np.argmax(model.predict(img_array), axis=1)
    return CLASS_NAMES[pred[0]]

# Exemple d'utilisation
if __name__ == "__main__":
    test_img = "../data/HAM10000/HAM10000_images_part_1/ISIC_0024306.jpg"  # change with your image
    print("Prediction:", predict_image(test_img))

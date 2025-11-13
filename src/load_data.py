import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from PIL import Image

DATA_DIR = "../data/HAM10000"


def load_data(img_size=(64, 64)):
    # Lire le CSV avec les labels
    labels_df = pd.read_csv(os.path.join(DATA_DIR, "HAM10000_metadata.csv"))
    
    # Encoder les labels
    labels_df['dx'] = labels_df['dx'].astype('category').cat.codes

    images = []
    labels = []

    # Parcourir les deux sous-dossiers
    subfolders = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
    for folder in subfolders:
        folder_path = os.path.join(DATA_DIR, folder)
        for _, row in labels_df.iterrows():
            img_path = os.path.join(folder_path, row['image_id'] + ".jpg")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                img = img.resize(img_size)
                images.append(np.array(img))
                labels.append(row['dx'])

    X = np.array(images, dtype="float32") / 255.0
    y = to_categorical(np.array(labels))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from PIL import Image

DATA_DIR = "../data/HAM10000"


def load_data(img_size=(128, 128)):  # Increased from 64x64 to 128x128
    """
    Load and preprocess the HAM10000 dataset with improved efficiency.
    
    Args:
        img_size: Target image size (height, width) - default 128x128 for better detail
    
    Returns:
        X_train, X_test, y_train, y_test: Training and test datasets
    """
    # Lire le CSV avec les labels
    labels_df = pd.read_csv(os.path.join(DATA_DIR, "HAM10000_metadata.csv"))
    
    # Encoder les labels
    label_mapping = {label: idx for idx, label in enumerate(sorted(labels_df['dx'].unique()))}
    labels_df['dx_encoded'] = labels_df['dx'].map(label_mapping)

    images = []
    labels = []

    # Parcourir les deux sous-dossiers de manière plus efficace
    subfolders = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
    
    print(f"Loading images from {len(subfolders)} folders...")
    loaded_count = 0
    
    for _, row in labels_df.iterrows():
        img_found = False
        # Chercher l'image dans les deux dossiers
        for folder in subfolders:
            folder_path = os.path.join(DATA_DIR, folder)
            img_path = os.path.join(folder_path, row['image_id'] + ".jpg")
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(img_size, Image.LANCZOS)  # Better resampling
                    images.append(np.array(img))
                    labels.append(row['dx_encoded'])
                    loaded_count += 1
                    img_found = True
                    break
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
        
        if not img_found and loaded_count % 1000 == 0:
            print(f"Warning: Image {row['image_id']} not found in any folder")
    
    print(f"Loaded {loaded_count} images")
    
    if len(images) == 0:
        raise ValueError("No images were loaded! Check your data directory.")
    
    X = np.array(images, dtype="float32") / 255.0
    y = to_categorical(np.array(labels), num_classes=len(label_mapping))
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(np.argmax(y, axis=1))}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def get_class_weights(y_train):
    """
    Calculate class weights to handle imbalanced dataset.
    
    Args:
        y_train: One-hot encoded training labels
    
    Returns:
        Dictionary of class weights
    """
    y_train_labels = np.argmax(y_train, axis=1)
    classes = np.unique(y_train_labels)
    class_weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=y_train_labels
    )
    weights_dict = dict(zip(classes, class_weights))
    print(f"Class weights: {weights_dict}")
    return weights_dict


from PIL import Image
import numpy as np
import pandas as pd


def preprocess_image(image: Image.Image, target_size=(64, 64)):
    """Convert PIL image to model-ready numpy batch.

    Args:
        image: PIL Image (RGB)
        target_size: (width, height)

    Returns:
        numpy array shape (1, h, w, 3) with float32 values in [0,1]
    """
    img = image.resize(target_size)
    arr = np.asarray(img, dtype="float32") / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr


def load_label_mapping(metadata_csv_path):
    """Recreate label mapping used during training.

    The training code used `labels_df['dx'] = labels_df['dx'].astype('category').cat.codes`.
    To ensure predictions map to the same label strings, reproduce the categories.
    """
    df = pd.read_csv(metadata_csv_path)
    cats = df['dx'].astype('category').cat.categories
    return list(cats)

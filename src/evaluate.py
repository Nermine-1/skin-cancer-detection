import os
import numpy as np
from tensorflow.keras.models import load_model
from load_data import load_data
import matplotlib
matplotlib.use('Agg')  # Utiliser un backend non-gui
import matplotlib.pyplot as plt

# Créer un dossier pour sauvegarder les images avec prédictions
OUTPUT_DIR = "../evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Charger les données
X_train, X_test, y_train, y_test = load_data()

# Charger le modèle entraîné
model = load_model('../models/skin_cancer_model.h5')

# Évaluer le modèle
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Faire des prédictions
pred_probs = model.predict(X_test)
predictions = np.argmax(pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Sauvegarder les premières 20 images avec prédictions
for i in range(min(20, len(X_test))):
    plt.imshow(X_test[i])
    plt.title(f"Predicted: {predictions[i]}, True: {y_true[i]}")
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, f"image_{i+1}.png"))
    plt.close()  # Ferme la figure pour libérer la mémoire

print(f"Images with predictions saved in {OUTPUT_DIR}")

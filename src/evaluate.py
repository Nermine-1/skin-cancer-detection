import os
import numpy as np
from tensorflow.keras.models import load_model
from load_data import load_data
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Utiliser un backend non-gui
import matplotlib.pyplot as plt
import seaborn as sns

# Noms des classes
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Créer un dossier pour sauvegarder les résultats
OUTPUT_DIR = "../evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")
X_train, X_test, y_train, y_test = load_data()

print("Loading model...")
model = load_model('../models/skin_cancer_model.h5')

# Évaluer le modèle
print("\nEvaluating model on test set...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"\n{'='*50}")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*50}\n")

# Faire des prédictions
print("Making predictions...")
pred_probs = model.predict(X_test, verbose=1)
predictions = np.argmax(pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification report détaillé
print("\n" + "="*50)
print("DETAILED CLASSIFICATION REPORT")
print("="*50)
report = classification_report(
    y_true, predictions,
    target_names=CLASS_NAMES,
    digits=4
)
print(report)

# Sauvegarder le rapport
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Test Loss: {loss:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# Matrice de confusion
cm = confusion_matrix(y_true, predictions)

# Visualiser la matrice de confusion
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300)
plt.close()
print(f"\nConfusion matrix saved to {OUTPUT_DIR}/confusion_matrix.png")

# Calculer les métriques par classe
print("\n" + "="*50)
print("PER-CLASS METRICS")
print("="*50)
for i, class_name in enumerate(CLASS_NAMES):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    tn = cm.sum() - tp - fp - fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    support = cm[i, :].sum()
    
    print(f"\n{class_name.upper()}:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Support:   {support}")

# Sauvegarder quelques exemples d'images avec prédictions
print(f"\nSaving sample predictions...")
for i in range(min(20, len(X_test))):
    plt.figure(figsize=(8, 6))
    plt.imshow(X_test[i])
    true_label = CLASS_NAMES[y_true[i]]
    pred_label = CLASS_NAMES[predictions[i]]
    confidence = pred_probs[i][predictions[i]] * 100
    
    # Colorer le titre selon si la prédiction est correcte
    color = 'green' if y_true[i] == predictions[i] else 'red'
    title = f"True: {true_label} | Pred: {pred_label} ({confidence:.1f}%)"
    plt.title(title, color=color, fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"image_{i+1}.png"), dpi=150, bbox_inches='tight')
    plt.close()

print(f"Sample images saved to {OUTPUT_DIR}/")
print(f"\nAll evaluation results saved in {OUTPUT_DIR}/")

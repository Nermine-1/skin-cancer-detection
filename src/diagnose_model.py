"""
Diagnostic script to analyze model performance and identify issues.
"""
import numpy as np
from tensorflow.keras.models import load_model
from load_data import load_data
from sklearn.metrics import confusion_matrix

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

print("="*60)
print("MODEL DIAGNOSTICS")
print("="*60)

# Load data
print("\nLoading data...")
X_train, X_test, y_train, y_test = load_data()

# Load model
print("Loading model...")
model = load_model('../models/skin_cancer_model.h5')

# Evaluate
print("\nEvaluating model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Test Loss: {loss:.4f}")

# Predictions
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test, axis=1)

# Class distribution
print("\n" + "="*60)
print("CLASS DISTRIBUTION IN TEST SET")
print("="*60)
unique, counts = np.unique(y_true, return_counts=True)
for class_idx, count in zip(unique, counts):
    print(f"{CLASS_NAMES[class_idx]:10s}: {count:4d} ({count/len(y_test)*100:5.2f}%)")

# Per-class accuracy
print("\n" + "="*60)
print("PER-CLASS ACCURACY")
print("="*60)
cm = confusion_matrix(y_true, y_pred)
for i, class_name in enumerate(CLASS_NAMES):
    if cm[i, :].sum() > 0:
        class_acc = cm[i, i] / cm[i, :].sum()
        print(f"{class_name:10s}: {class_acc:.4f} ({class_acc*100:.2f}%)")

# Most confused classes
print("\n" + "="*60)
print("MOST CONFUSED CLASSES")
print("="*60)
for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        if i != j and cm[i, j] > 5:  # Show if more than 5 misclassifications
            print(f"{CLASS_NAMES[i]} → {CLASS_NAMES[j]}: {cm[i, j]} times")

# Check if model is just predicting the majority class
majority_class = np.argmax(np.bincount(y_true))
majority_predictions = np.sum(y_pred == majority_class)
majority_accuracy = np.sum((y_pred == majority_class) & (y_true == majority_class)) / np.sum(y_true == majority_class)

print("\n" + "="*60)
print("MAJORITY CLASS ANALYSIS")
print("="*60)
print(f"Majority class: {CLASS_NAMES[majority_class]}")
print(f"Model predicts majority class: {majority_predictions}/{len(y_pred)} ({majority_predictions/len(y_pred)*100:.2f}%)")
print(f"Accuracy on majority class: {majority_accuracy:.4f}")

# Recommendations
print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)
if accuracy < 0.70:
    print("⚠️  Low accuracy detected (<70%)")
    print("   → Try larger image size (128x128 or 224x224)")
    print("   → Consider transfer learning (ResNet, EfficientNet)")
    print("   → Check data quality and preprocessing")
    
if np.max([cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0 for i in range(len(CLASS_NAMES))]) < 0.60:
    print("⚠️  Poor per-class performance detected")
    print("   → Class imbalance may be severe")
    print("   → Try stronger class weighting or oversampling")
    
if majority_predictions / len(y_pred) > 0.50:
    print("⚠️  Model may be biased toward majority class")
    print("   → Increase class weights")
    print("   → Use focal loss instead of categorical crossentropy")

print("\n" + "="*60)


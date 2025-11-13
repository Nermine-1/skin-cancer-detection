import os
import pickle
from load_data import load_data, get_class_weights
from model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Charger les données
print("Loading data...")
X_train, X_test, y_train, y_test = load_data()
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Calculer les poids de classe pour gérer le déséquilibre
class_weights = get_class_weights(y_train)
print(f"Class weights: {class_weights}")

# Créer le modèle
print("Building model...")
model = build_model(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])
model.summary()

# Data augmentation pour améliorer la généralisation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Callbacks pour améliorer l'entraînement
callbacks = [
    # Sauvegarde automatique du meilleur modèle
    ModelCheckpoint(
        '../models/skin_cancer_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Arrêt anticipé si pas d'amélioration
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    # Réduction du learning rate si plateau
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# Entraîner le modèle avec data augmentation
print("Starting training...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    validation_data=(X_test, y_test),
    epochs=50,  # Plus d'epochs avec early stopping
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Sauvegarder l'historique d'entraînement
history_path = '../models/training_history.pkl'
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)
print(f"\nTraining history saved to {history_path}")

# Afficher les métriques finales
final_val_acc = max(history.history['val_accuracy'])
final_train_acc = max(history.history['accuracy'])
print(f"\nBest validation accuracy: {final_val_acc:.4f}")
print(f"Best training accuracy: {final_train_acc:.4f}")

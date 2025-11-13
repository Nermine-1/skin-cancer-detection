import os
import pickle
from load_data_improved import load_data, get_class_weights
from model_improved import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
IMG_SIZE = (128, 128)  # Larger images for better detail
INITIAL_LR = 0.001  # Start with a reasonable learning rate
BATCH_SIZE = 32
EPOCHS = 100

print("="*60)
print("IMPROVED SKIN CANCER DETECTION TRAINING")
print("="*60)

# Charger les données
print("\n[1/5] Loading data...")
X_train, X_test, y_train, y_test = load_data(img_size=IMG_SIZE)
print(f"✓ Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Calculer les poids de classe pour gérer le déséquilibre
print("\n[2/5] Calculating class weights...")
class_weights = get_class_weights(y_train)

# Créer le modèle
print("\n[3/5] Building model...")
model = build_model(
    input_shape=X_train.shape[1:], 
    num_classes=y_train.shape[1],
    learning_rate=INITIAL_LR
)
model.summary()

# Data augmentation avec des paramètres plus conservateurs
print("\n[4/5] Setting up data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=15,  # Reduced from 20
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,  # Usually not needed for skin lesions
    fill_mode='nearest',
    brightness_range=[0.9, 1.1]  # Slight brightness variation
)

# Callbacks améliorés
print("\n[5/5] Setting up callbacks...")
callbacks = [
    # Sauvegarde automatique du meilleur modèle
    ModelCheckpoint(
        '../models/skin_cancer_model_improved.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        save_weights_only=False
    ),
    # Arrêt anticipé avec patience plus longue
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,  # Increased patience
        restore_best_weights=True,
        verbose=1,
        min_delta=0.0001
    ),
    # Réduction du learning rate
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,  # More aggressive reduction
        patience=7,
        min_lr=1e-7,
        verbose=1,
        cooldown=2
    ),
    # Logger pour suivre l'entraînement
    CSVLogger('../models/training_log.csv', append=False)
]

# Entraîner le modèle avec data augmentation
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)
print(f"Image size: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max epochs: {EPOCHS}")
print(f"Initial learning rate: {INITIAL_LR}")
print("="*60 + "\n")

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Sauvegarder l'historique d'entraînement
history_path = '../models/training_history_improved.pkl'
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)
print(f"\n✓ Training history saved to {history_path}")

# Afficher les métriques finales
if len(history.history['val_accuracy']) > 0:
    final_val_acc = max(history.history['val_accuracy'])
    final_train_acc = max(history.history['accuracy'])
    best_epoch = history.history['val_accuracy'].index(final_val_acc) + 1
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    print(f"Best training accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"Best model at epoch: {best_epoch}")
    print("="*60)


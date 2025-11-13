import os
import pickle
import numpy as np
from load_data_improved import load_data, get_class_weights
from model_improved import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Focal Loss implementation for imbalanced datasets
def focal_loss(gamma=2.0, alpha=None):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate p_t
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        
        # Calculate alpha_t
        if alpha is not None:
            alpha_t = tf.constant(alpha, dtype=tf.float32)
            alpha_t = tf.expand_dims(alpha_t, axis=0)
            alpha_t = tf.reduce_sum(alpha_t * y_true, axis=1)
            alpha_t = tf.expand_dims(alpha_t, axis=1)
        else:
            alpha_t = 1.0
        
        # Calculate focal loss
        focal_loss = alpha_t * tf.pow((1 - p_t), gamma) * cross_entropy
        return tf.reduce_mean(focal_loss)
    
    return focal_loss_fixed

# Configuration
IMG_SIZE = (128, 128)  # Larger images
INITIAL_LR = 0.001
BATCH_SIZE = 32
EPOCHS = 100

print("="*60)
print("TRAINING WITH FOCAL LOSS (FOR IMBALANCED DATA)")
print("="*60)

# Load data
print("\n[1/6] Loading data...")
X_train, X_test, y_train, y_test = load_data(img_size=IMG_SIZE)
print(f"✓ Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Calculate class weights
print("\n[2/6] Calculating class weights...")
class_weights = get_class_weights(y_train)

# Calculate alpha for focal loss (inverse of class frequency)
y_train_labels = np.argmax(y_train, axis=1)
class_counts = np.bincount(y_train_labels)
total_samples = len(y_train_labels)
alpha = total_samples / (len(class_counts) * class_counts)
alpha = alpha / alpha.sum() * len(class_counts)  # Normalize
print(f"Focal loss alpha values: {dict(zip(range(len(alpha)), alpha))}")

# Build model
print("\n[3/6] Building model...")
model = build_model(
    input_shape=X_train.shape[1:], 
    num_classes=y_train.shape[1],
    learning_rate=INITIAL_LR
)

# Compile with focal loss
print("\n[4/6] Compiling model with Focal Loss...")
focal_loss_fn = focal_loss(gamma=2.0, alpha=alpha.tolist())
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
    loss=focal_loss_fn,
    metrics=['accuracy']
)
model.summary()

# Data augmentation
print("\n[5/6] Setting up data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.9, 1.1]
)

# Callbacks
print("\n[6/6] Setting up callbacks...")
callbacks = [
    ModelCheckpoint(
        '../models/skin_cancer_model_focal.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=20,  # More patience
        restore_best_weights=True,
        verbose=1,
        min_delta=0.0001
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=8,
        min_lr=1e-7,
        verbose=1
    ),
    CSVLogger('../models/training_log_focal.csv', append=False)
]

# Train
print("\n" + "="*60)
print("STARTING TRAINING WITH FOCAL LOSS")
print("="*60)
print(f"Image size: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max epochs: {EPOCHS}")
print(f"Initial learning rate: {INITIAL_LR}")
print(f"Focal loss gamma: 2.0")
print("="*60 + "\n")

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    class_weight=class_weights,  # Still use class weights
    callbacks=callbacks,
    verbose=1
)

# Save history
history_path = '../models/training_history_focal.pkl'
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)
print(f"\n✓ Training history saved to {history_path}")

# Final metrics
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


from load_data import load_data
from model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint

# Charger les données
X_train, X_test, y_train, y_test = load_data()

# Créer le modèle
model = build_model(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])

# Sauvegarde automatique du meilleur modèle
checkpoint = ModelCheckpoint('../models/skin_cancer_model.h5', monitor='val_accuracy', save_best_only=True)

# Entraîner le modèle
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=25,
    batch_size=32,
    callbacks=[checkpoint]
)

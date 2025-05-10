import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping

# Chemin des données
DATASET_PATH = "C:/Users/manar/Downloads/archive/asl_alphabet_train/asl_alphabet_train"
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20  # Augmenté avec EarlyStopping

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Définition du modèle amélioré avec régularisation
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3),
                  kernel_regularizer=regularizers.l2(0.001)),  # L2 Regularization
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dropout(0.6),  # Augmenter le Dropout pour éviter l'overfitting
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(len(train_data.class_indices), activation='softmax')
])

# Compilation avec Adam et categorical_crossentropy
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Réduction du LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# EarlyStopping pour stopper si pas d'amélioration
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Entraînement
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stopping]  # Activation de EarlyStopping
)

# Sauvegarde du modèle
model.save("model_improved.h5")
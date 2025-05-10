import tensorflow as tf
import numpy as np
import cv2

# Charger le modèle entraîné
model = tf.keras.models.load_model("model.h5")

# Classes du modèle
labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]


# Charger une image
img_path ="C:/Users/manar/Downloads/archive/asl_alphabet_test/asl_alphabet_test/J_test.jpg"
img = cv2.imread(img_path)
img = cv2.resize(img, (64, 64))
img = np.expand_dims(img, axis=0) / 255.0  # Normalisation

# Prédiction
prediction = model.predict(img)
predicted_class = np.argmax(prediction)
print(f"Lettre détectée : {labels[predicted_class]}")

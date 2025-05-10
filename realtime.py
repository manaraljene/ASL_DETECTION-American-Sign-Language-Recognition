import cv2
import numpy as np
import tensorflow as tf

# Charger le modèle entraîné
model = tf.keras.models.load_model("model.h5")

# Labels des signes ASL
labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionner l'image pour correspondre au modèle (64x64)
    img = cv2.resize(frame, (64, 64))
    img = np.expand_dims(img, axis=0) / 255.0  # Normalisation

    # Prédiction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    # Ajouter le texte sur l'image
    text = f"Lettre detectee : {labels[predicted_class]}"
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Afficher l'image
    cv2.imshow("ASL Detection", frame)

    # Appuyer sur 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fermer la webcam
cap.release()
cv2.destroyAllWindows()

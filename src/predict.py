import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

MODEL_PATH = 'model/sign_model.h5'
IMG_SIZE = 28
LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]

model = load_model(MODEL_PATH)

# Error handling for model
if not os.path.exists(MODEL_PATH):
    print(f"Model file not found at {MODEL_PATH}. Please train the model first.")
    exit(1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not found. Please check your camera.")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    x1, y1, x2, y2 = 100, 100, 228, 228  # 128x128 ROI for hand
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # Shape: (1, 28, 28, 1)

    pred = model.predict(img)
    idx = np.argmax(pred)
    letter = LABELS[idx]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, f'Prediction: {letter}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

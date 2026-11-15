import os
import sys
import json
import cv2
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from train import extract_face_features, detect_faces_safe

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cricket_player_model.keras")
FALLBACK_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cricket_player_model.npz")
CLASS_NAMES_PATH = os.path.join(PROJECT_ROOT, "class_names.json")
IMAGE_SIZE = (224, 224)

try:
    import tensorflow as tf
    TF_AVAILABLE = os.path.exists(MODEL_PATH)
except ImportError:
    TF_AVAILABLE = False

# Load Class Names
with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

print("Class names:", class_names)

if TF_AVAILABLE:
    model = tf.keras.models.load_model(MODEL_PATH)
    W_model = b_model = temp_model = None
else:
    weights = np.load(FALLBACK_MODEL_PATH, allow_pickle=True)
    W_model = weights["W"]
    b_model = weights["b"]
    temp_model = float(weights["temperature"][0])


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


def predict_player(image_path):
    if not os.path.isabs(image_path) and not os.path.exists(image_path):
        image_path = os.path.join(PROJECT_ROOT, image_path)

    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Image not found at {image_path}")
        return None, 0

    faces = detect_faces_safe(img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) == 0:
        print("No face detected.")
        return None, 0

    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face

    margin = 20
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img.shape[1], x + w + margin)
    y2 = min(img.shape[0], y + h + margin)

    face_bgr = img[y1:y2, x1:x2]
    if face_bgr.size == 0:
        face_bgr = img

    if TF_AVAILABLE:
        face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, IMAGE_SIZE)
        face = face.astype(np.float32)
        face = np.expand_dims(face, axis=0)
        face = tf.keras.applications.mobilenet_v2.preprocess_input(face)
        predictions = model.predict(face, verbose=0)[0]
    else:
        feat = extract_face_features(face_bgr)
        logits = (feat @ W_model + b_model) * temp_model
        predictions = _softmax(logits)

    predicted_index = int(np.argmax(predictions))
    predicted_player = class_names[predicted_index]
    confidence = float(predictions[predicted_index] * 100)

    return predicted_player, confidence


if __name__ == "__main__":
    image_path = os.path.join(PROJECT_ROOT, "test_images", "virat_test.png")
    player, confidence = predict_player(image_path)

    if player is not None:
        print("\n===============================")
        print("PREDICTION RESULT")
        print("===============================")
        print(f"Player: {player.replace('_', ' ')}")
        print(f"Confidence: {confidence:.2f}%")

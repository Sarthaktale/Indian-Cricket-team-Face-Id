import os
import sys
import json
import cv2
import numpy as np
import streamlit as st
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

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

st.set_page_config(
    page_title="Indian Cricket Player Identification",
    page_icon="🏏",
    layout="centered"
)


@st.cache_resource
def load_model():
    if TF_AVAILABLE:
        return ("tf", tf.keras.models.load_model(MODEL_PATH))
    weights = np.load(FALLBACK_MODEL_PATH, allow_pickle=True)
    return ("npz", (weights["W"], weights["b"], float(weights["temperature"][0])))


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as file:
        return json.load(file)


backend_type, model_obj = load_model()
class_names = load_class_names()

st.title("🏏 Indian Cricket Player Face Identification")
st.write(
    "Upload an image of an Indian cricket player "
    "to identify the player using a trained "
    "computer vision / deep learning model."
)

uploaded_file = st.file_uploader(
    "Upload Player Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Identify Player"):
        img_rgb = np.array(img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        faces = detect_faces_safe(
            img_bgr, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            margin = 20
            x1, y1 = max(0, x - margin), max(0, y - margin)
            x2, y2 = min(img_bgr.shape[1], x + w + margin), min(img_bgr.shape[0], y + h + margin)
            crop_bgr = img_bgr[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                crop_bgr = img_bgr
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        else:
            crop_bgr = img_bgr
            crop_rgb = img_rgb

        if backend_type == "tf":
            resized_img = cv2.resize(crop_rgb, IMAGE_SIZE).astype(np.float32)
            img_array = np.expand_dims(resized_img, axis=0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            probs = model_obj.predict(img_array, verbose=0)[0]
        else:
            W_model, b_model, temp_model = model_obj
            feat = extract_face_features(crop_bgr)
            logits = (feat @ W_model + b_model) * temp_model
            e_x = np.exp(logits - np.max(logits))
            probs = e_x / np.sum(e_x)

        predicted_index = int(np.argmax(probs))
        predicted_player = class_names[predicted_index]
        confidence = float(probs[predicted_index] * 100)

        st.success("Player: " + predicted_player.replace("_", " ").title())
        st.info(f"Confidence: {confidence:.2f}%")

        st.subheader("Top 3 Predictions")
        top_indices = np.argsort(probs)[-3:][::-1]

        for index in top_indices:
            player = class_names[int(index)].replace("_", " ").title()
            probability = float(probs[int(index)] * 100)
            st.write(f"**{player}** - {probability:.2f}%")

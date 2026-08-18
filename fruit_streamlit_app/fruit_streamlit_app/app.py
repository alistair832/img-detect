from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "fruit_resnet50.keras"
CLASS_NAMES_PATH = APP_DIR / "models" / "class_names.json"
IMAGE_SIZE = (100, 100)
TOP_K = 5


st.set_page_config(
    page_title="Fruit Image Detection",
    page_icon="🍎",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train the model first with train_model.py "
            "or copy your exported .keras model into the models folder."
        )
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class label file not found at {CLASS_NAMES_PATH}."
        )
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_image(image: Image.Image) -> np.ndarray:
    """Convert uploaded/camera image to the 100x100 RGB input used by the model."""
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def predict_image(model, class_names, image: Image.Image, top_k: int = TOP_K):
    batch = prepare_image(image)
    logits = model.predict(batch, verbose=0)[0]
    probabilities = tf.nn.softmax(logits).numpy()

    top_indices = np.argsort(probabilities)[::-1][:top_k]
    results = [
        {
            "Fruit": class_names[int(index)],
            "Confidence (%)": float(probabilities[index] * 100.0),
        }
        for index in top_indices
    ]
    return results


st.title("🍎 Fruit Image Detection and Classification")
st.write(
    "Upload a fruit image or take a photo. The system classifies the image using "
    "a ResNet50 transfer-learning model trained on the Fruits-360 dataset."
)

with st.sidebar:
    st.header("Model Information")
    st.write("**Algorithm:** ResNet50 Transfer Learning")
    st.write("**Input size:** 100 × 100 RGB")
    st.write("**Dataset:** Fruits-360")
    st.write("**Output:** Fruit class + confidence")
    st.caption(
        "For best results, use a clear image with one main fruit and a simple background."
    )

try:
    model = load_model()
    class_names = load_class_names()
except Exception as exc:
    st.error(str(exc))
    st.info(
        "Run `python train_model.py` first. If you already trained the notebook model, "
        "save it as `models/fruit_resnet50.keras` and save the class labels as "
        "`models/class_names.json`."
    )
    st.stop()

source = st.radio(
    "Choose image source:",
    ["Upload image", "Use camera"],
    horizontal=True,
)

image_file = None
if source == "Upload image":
    image_file = st.file_uploader(
        "Upload JPG, JPEG, or PNG",
        type=["jpg", "jpeg", "png"],
    )
else:
    image_file = st.camera_input("Take a fruit photo")

if image_file is not None:
    image = Image.open(image_file)

    image_col, result_col = st.columns([1, 1])

    with image_col:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    with result_col:
        st.subheader("Prediction")
        with st.spinner("Classifying image..."):
            results = predict_image(model, class_names, image)

        best = results[0]
        st.success(f"Predicted fruit: **{best['Fruit']}**")
        st.metric("Confidence", f"{best['Confidence (%)']:.2f}%")

        if best["Confidence (%)"] < 50:
            st.warning(
                "The model is not very confident. Try a clearer photo with one fruit "
                "near the centre of the image."
            )

        st.subheader("Top 5 Predictions")
        results_df = pd.DataFrame(results)
        st.dataframe(
            results_df.style.format({"Confidence (%)": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

        chart_df = results_df.set_index("Fruit")[["Confidence (%)"]]
        st.bar_chart(chart_df)
else:
    st.info("Upload or capture a fruit image to start prediction.")

st.divider()
st.caption(
    "AI assignment demo: fruit image classification using Fruits-360 and ResNet50."
)

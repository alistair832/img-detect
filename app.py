from pathlib import Path
import json

import numpy as np
import streamlit as st
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
MODEL_PATH = MODEL_DIR / "fruit_resnet50.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
IMAGE_SIZE = (100, 100)
TOP_K = 5

st.set_page_config(
    page_title="Fruit Image Detection",
    page_icon="🍎",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading fruit recognition model...")
def load_model():
    """Load TensorFlow only when the model is actually available."""
    if not MODEL_PATH.exists():
        return None, "model_missing"

    try:
        import tensorflow as tf
    except ImportError:
        return None, "tensorflow_missing"

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model, None
    except Exception as exc:
        return None, f"model_error: {exc}"


@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        return None

    try:
        with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
            labels = json.load(file)
        return labels
    except Exception:
        return None


def prepare_image(image: Image.Image) -> np.ndarray:
    """Prepare an uploaded image for the 100x100 RGB Fruits-360 model."""
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def to_probabilities(raw_output: np.ndarray) -> np.ndarray:
    """Use model probabilities directly, or softmax raw logits safely."""
    output = np.asarray(raw_output, dtype=np.float64).reshape(-1)

    if (
        np.all(output >= 0.0)
        and np.all(output <= 1.0)
        and np.isclose(output.sum(), 1.0, atol=1e-3)
    ):
        return output

    output = output - np.max(output)
    exp_output = np.exp(output)
    return exp_output / np.sum(exp_output)


def predict_image(model, class_names, image: Image.Image):
    batch = prepare_image(image)
    raw_predictions = model.predict(batch, verbose=0)[0]
    probabilities = to_probabilities(raw_predictions)

    number_of_classes = min(len(class_names), len(probabilities))
    probabilities = probabilities[:number_of_classes]
    class_names = class_names[:number_of_classes]

    top_count = min(TOP_K, number_of_classes)
    top_indices = np.argsort(probabilities)[::-1][:top_count]

    return [
        (str(class_names[int(index)]), float(probabilities[int(index)] * 100.0))
        for index in top_indices
    ]


st.title("🍎 Fruit Image Detection")
st.write(
    "Upload a fruit image and the system will classify it using a ResNet50 "
    "transfer-learning model trained with the Fruits-360 dataset."
)

with st.sidebar:
    st.header("Project Information")
    st.write("**Problem:** Fruit image classification")
    st.write("**Algorithm:** ResNet50 transfer learning")
    st.write("**Dataset:** Fruits-360")
    st.write("**Input:** 100 × 100 RGB image")
    st.write("**Output:** Fruit class and confidence")

model, model_error = load_model()
class_names = load_class_names()

if model is None or class_names is None:
    st.warning("The Streamlit interface is running, but the trained model files are not in the repository yet.")

    if model_error == "tensorflow_missing":
        st.error("TensorFlow is not installed. Check requirements.txt and reboot the Streamlit app.")
    elif model_error and model_error.startswith("model_error:"):
        st.error(model_error)

    st.info(
        "Add these two files to the repository and reboot the app:\n\n"
        "- `models/fruit_resnet50.keras`\n"
        "- `models/class_names.json`"
    )

uploaded_file = st.file_uploader(
    "Upload a fruit image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"Unable to read this image: {exc}")
        st.stop()

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    with right:
        st.subheader("Prediction")

        if model is None or class_names is None:
            st.error("Prediction is unavailable until the trained model and class label files are added.")
        else:
            try:
                with st.spinner("Detecting fruit..."):
                    results = predict_image(model, class_names, image)

                if not results:
                    st.error("The model returned no predictions.")
                else:
                    fruit, confidence = results[0]
                    st.success(f"Predicted fruit: **{fruit}**")
                    st.metric("Confidence", f"{confidence:.2f}%")

                    st.subheader("Top Predictions")
                    for rank, (name, score) in enumerate(results, start=1):
                        st.write(f"**{rank}. {name}** — {score:.2f}%")
                        st.progress(min(max(score / 100.0, 0.0), 1.0))
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.caption("Check that the model output matches the class_names.json labels.")
else:
    st.info("Upload a JPG, JPEG, PNG, or WEBP fruit image to begin.")

st.divider()
st.caption("Artificial Intelligence project — Fruit Image Detection using Fruits-360 and ResNet50")

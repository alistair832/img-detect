from pathlib import Path
import json
import os
import sys
from collections import Counter, deque

# Keep TensorFlow's CPU runtime conservative on small cloud containers.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import av
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_webrtc import webrtc_streamer

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_fruit_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

st.set_page_config(
    page_title="Fruits-360 Live Recognition",
    page_icon="🍎",
    layout="wide",
)

# Do not import TensorFlow until a trained model actually exists. This keeps the
# Streamlit app healthy before training and avoids loading a large native runtime
# unnecessarily during startup.
missing_assets = [
    path.name
    for path in (MODEL_PATH, CLASS_NAMES_PATH)
    if not path.exists()
]

if missing_assets:
    st.title("🍎 Fruits-360 Live Recognition")
    st.success("Streamlit deployment is running correctly.")
    st.warning(
        "The full Fruits-360 model has not been trained/uploaded yet. "
        "Missing: " + ", ".join(missing_assets)
    )
    st.write(f"Current Python: **{sys.version.split()[0]}**")
    st.markdown(
        "**GitHub-first next step:** run the training script, then push the generated "
        "deployment files back to this repository."
    )
    st.code(
        "cd fruits360_full_project\n"
        "python -m pip install -r requirements-training.txt\n"
        "python train.py\n",
        language="bash",
    )
    st.markdown("The training script will generate:")
    st.code(
        "models/\n"
        "├── best_fruit_model.keras\n"
        "├── class_names.json\n"
        "└── model_metadata.json",
        language="text",
    )
    st.info(
        "For a short pipeline test first, use `python train.py --quick`. "
        "Use the normal `python train.py` run for final assignment results."
    )
    st.stop()

if sys.version_info[:2] != (3, 12):
    st.title("🍎 Fruits-360 Live Recognition")
    st.error(
        f"This trained-model deployment requires Python 3.12. "
        f"Current Python: {sys.version.split()[0]}"
    )
    st.info("Redeploy this Streamlit app with Python 3.12 in Advanced settings.")
    st.stop()

# TensorFlow is imported only when model files are available and the Python
# version matches the deployment environment used by the training project.
import tensorflow as tf


@st.cache_resource(show_spinner="Loading trained Fruits-360 model...")
def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing trained model: {MODEL_PATH}. Run train.py first."
        )
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Missing class names: {CLASS_NAMES_PATH}. Run train.py first."
        )

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as f:
        class_names = json.load(f)

    metadata = {}
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    input_h = int(model.input_shape[1])
    input_w = int(model.input_shape[2])
    return model, class_names, metadata, (input_w, input_h)


try:
    model, class_names, metadata, image_size = load_assets()
except Exception as exc:
    st.title("🍎 Fruits-360 Recognition")
    st.error(str(exc))
    st.info(
        "Run `python fruits360_full_project/train.py`, then make sure the generated "
        "deployment files are inside `fruits360_full_project/models/`."
    )
    st.stop()


def predict_image(image: Image.Image, top_k=5):
    image = image.convert("RGB").resize(image_size, Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32)
    batch = np.expand_dims(arr, axis=0)

    logits = model.predict(batch, verbose=0)[0]
    probs = tf.nn.softmax(logits).numpy()

    order = np.argsort(probs)[::-1][:top_k]
    results = [
        (class_names[int(i)], float(probs[int(i)]))
        for i in order
    ]
    return results


prediction_history = deque(maxlen=8)
confidence_history = deque(maxlen=8)
frame_counter = 0
last_label = "Analyzing..."


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    global frame_counter, last_label
    frame_counter += 1

    rgb = frame.to_ndarray(format="rgb24")
    rgb = rgb[:, ::-1].copy()  # selfie mirror
    image = Image.fromarray(rgb)

    width, height = image.size
    box_size = int(min(width, height) * 0.64)
    left = (width - box_size) // 2
    top = (height - box_size) // 2
    right = left + box_size
    bottom = top + box_size

    # Process every fourth frame to reduce CPU load.
    if frame_counter % 4 == 0:
        roi = image.crop((left, top, right, bottom))
        top_result = predict_image(roi, top_k=1)[0]
        name, confidence = top_result

        prediction_history.append(name)
        confidence_history.append(confidence)

        if len(prediction_history) >= 4:
            stable_name, votes = Counter(prediction_history).most_common(1)[0]
            stable_conf = [
                c for n, c in zip(prediction_history, confidence_history)
                if n == stable_name
            ]
            avg_conf = float(np.mean(stable_conf)) if stable_conf else 0.0

            if votes >= 4 and avg_conf >= 0.50:
                last_label = f"{stable_name} — {avg_conf*100:.1f}% confidence"
            else:
                last_label = "Unknown / hold object steady"
        else:
            last_label = "Analyzing..."

    draw = ImageDraw.Draw(image)
    draw.rectangle((left, top, right, bottom), outline=(40, 220, 90), width=5)
    draw.rounded_rectangle(
        (12, 12, min(width - 12, 610), 68),
        radius=12,
        fill=(0, 0, 0),
    )
    draw.text((25, 30), last_label, fill=(255, 255, 255))
    draw.rectangle((left, bottom - 36, right, bottom), fill=(0, 0, 0))
    draw.text(
        (left + 10, bottom - 26),
        "Place one object inside the green box",
        fill=(255, 255, 255),
    )

    return av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")


st.title("🍎 Fruits-360 Live Recognition")
st.write(
    "Use the live front camera or upload a picture. "
    "The system uses the model exported by the GitHub-first training script."
)

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Classes", len(class_names))
with m2:
    st.metric("Model", metadata.get("best_model", "Trained model"))
with m3:
    if "test_accuracy" in metadata:
        st.metric("Test accuracy", f"{metadata['test_accuracy']*100:.2f}%")
    else:
        st.metric("Input size", f"{image_size[0]}×{image_size[1]}")

camera_tab, upload_tab = st.tabs(["🎥 Live Front Camera", "🖼️ Upload Picture"])

with camera_tab:
    st.caption(
        "Click START, allow camera permission, and place one main object inside the green box. "
        "The app processes every few frames to keep inference responsive."
    )
    webrtc_streamer(
        key="fruits360-camera",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "facingMode": "user",
                "width": {"ideal": 640},
                "height": {"ideal": 480},
            },
            "audio": False,
        },
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        async_processing=True,
    )

with upload_tab:
    uploaded = st.file_uploader(
        "Upload JPG, JPEG, PNG or WEBP",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        left_col, right_col = st.columns(2)

        with left_col:
            st.image(image, use_container_width=True)

        with right_col:
            with st.spinner("Classifying..."):
                results = predict_image(image, top_k=5)

            best_name, best_conf = results[0]

            if best_conf >= 0.50:
                st.success(f"Detected: **{best_name}**")
            else:
                st.warning(f"Low-confidence result: **{best_name}**")

            st.metric("Prediction confidence", f"{best_conf*100:.2f}%")

            st.subheader("Top 5 predictions")
            for name, confidence in results:
                st.write(f"**{name}** — {confidence*100:.2f}%")
                st.progress(float(min(max(confidence, 0.0), 1.0)))

st.divider()
st.caption(
    "Assignment deployment — Fruits-360 image classification with TensorFlow and Streamlit."
)

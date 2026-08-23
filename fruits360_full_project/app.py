from pathlib import Path
import json
import os
import sys
import threading
import time

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

# Do not import TensorFlow until a trained model actually exists.
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

# TensorFlow is imported only when model files are available.
import tensorflow as tf


@st.cache_resource(show_spinner="Loading trained Fruits-360 model...")
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as f:
        class_names = json.load(f)

    metadata = {}
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    input_h = int(model.input_shape[1])
    input_w = int(model.input_shape[2])

    # Warm up TensorFlow once during app startup so the first camera prediction
    # does not pay the model initialization cost.
    dummy = np.zeros((1, input_h, input_w, 3), dtype=np.float32)
    _ = model(dummy, training=False)

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


def predict_probabilities(image: Image.Image) -> np.ndarray:
    """Fast single-image inference used by both webcam and uploads."""
    resized = image.convert("RGB").resize(image_size, Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    batch = np.expand_dims(arr, axis=0)

    # Calling the model directly avoids the extra batching/progress machinery
    # used by model.predict(), which matters in a real-time callback.
    logits = model(batch, training=False)[0]
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()
    return np.asarray(probabilities, dtype=np.float32)


def predict_image(image: Image.Image, top_k=5):
    probabilities = predict_probabilities(image)
    order = np.argsort(probabilities)[::-1][:top_k]
    return [
        (class_names[int(index)], float(probabilities[int(index)]))
        for index in order
    ]


# -----------------------------------------------------------------------------
# Smooth webcam state
# -----------------------------------------------------------------------------
# The video can render at ~24 FPS while expensive model inference is intentionally
# limited to roughly 3 predictions per second. Non-blocking locking prevents two
# TensorFlow inferences from overlapping when async WebRTC callbacks overlap.
CAMERA_INFERENCE_INTERVAL = 0.32
SMOOTH_ALPHA = 0.45
MIN_DISPLAY_CONFIDENCE = 0.42
MIN_DISPLAY_MARGIN = 0.04

inference_lock = threading.Lock()
last_inference_time = 0.0
smoothed_probabilities = None
last_label = "Analyzing fruit..."


def _update_camera_prediction(roi: Image.Image):
    global last_inference_time, smoothed_probabilities, last_label

    probabilities = predict_probabilities(roi)

    if smoothed_probabilities is None:
        smoothed_probabilities = probabilities.copy()
    else:
        smoothed_probabilities = (
            SMOOTH_ALPHA * probabilities
            + (1.0 - SMOOTH_ALPHA) * smoothed_probabilities
        )

    order = np.argsort(smoothed_probabilities)[::-1]
    top_index = int(order[0])
    second_index = int(order[1]) if len(order) > 1 else top_index

    confidence = float(smoothed_probabilities[top_index])
    second_confidence = float(smoothed_probabilities[second_index])
    margin = confidence - second_confidence

    if confidence >= MIN_DISPLAY_CONFIDENCE and margin >= MIN_DISPLAY_MARGIN:
        last_label = f"{class_names[top_index]} — {confidence * 100:.1f}% confidence"
    else:
        last_label = "Unknown / hold fruit steady"

    last_inference_time = time.monotonic()


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    global last_inference_time

    rgb = frame.to_ndarray(format="rgb24")
    rgb = rgb[:, ::-1].copy()  # selfie mirror
    image = Image.fromarray(rgb)

    width, height = image.size
    box_size = int(min(width, height) * 0.64)
    left = max(0, (width - box_size) // 2)
    top = max(0, (height - box_size) // 2)
    right = min(width, left + box_size)
    bottom = min(height, top + box_size)

    now = time.monotonic()

    # Only one inference may run at a time. Most frames simply pass through and
    # reuse the latest label, which makes the visible camera much smoother.
    if now - last_inference_time >= CAMERA_INFERENCE_INTERVAL:
        if inference_lock.acquire(blocking=False):
            try:
                # Re-check after acquiring because another callback may have just
                # completed inference while this frame was waiting to run.
                if time.monotonic() - last_inference_time >= CAMERA_INFERENCE_INTERVAL:
                    roi = image.crop((left, top, right, bottom))
                    _update_camera_prediction(roi)
            except Exception:
                # Keep the video stream alive even if one inference fails.
                pass
            finally:
                inference_lock.release()

    draw = ImageDraw.Draw(image)
    draw.rectangle((left, top, right, bottom), outline=(40, 220, 90), width=4)

    label_right = min(width - 12, max(330, width - 12))
    draw.rounded_rectangle(
        (12, 12, label_right, 64),
        radius=10,
        fill=(0, 0, 0),
    )
    draw.text((24, 28), last_label, fill=(255, 255, 255))

    guide_top = max(top, bottom - 34)
    draw.rectangle((left, guide_top, right, bottom), fill=(0, 0, 0))
    draw.text(
        (left + 9, guide_top + 9),
        "Place one fruit inside the green box",
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
        st.metric("Test accuracy", f"{metadata['test_accuracy'] * 100:.2f}%")
    else:
        st.metric("Input size", f"{image_size[0]}×{image_size[1]}")

camera_tab, upload_tab = st.tabs(["🎥 Live Front Camera", "🖼️ Upload Picture"])

with camera_tab:
    st.subheader("Smooth Live Camera")
    st.caption(
        "Click START, allow camera permission, and keep one fruit inside the green box. "
        "Video frames remain smooth while recognition updates several times each second."
    )

    webrtc_streamer(
        key="fruits360-camera",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "facingMode": "user",
                "width": {"ideal": 480, "max": 640},
                "height": {"ideal": 360, "max": 480},
                "frameRate": {"ideal": 24, "max": 30},
            },
            "audio": False,
        },
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        async_processing=True,
    )

    st.caption(
        "Performance mode: camera ≈24 FPS, model inference ≈3 times/second, "
        "with smoothed confidence to reduce label flicker."
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

            best_name, best_confidence = results[0]

            if best_confidence >= 0.50:
                st.success(f"Detected: **{best_name}**")
            else:
                st.warning(f"Low-confidence result: **{best_name}**")

            st.metric(
                "Prediction confidence",
                f"{best_confidence * 100:.2f}%",
            )

            st.subheader("Top 5 predictions")
            for name, confidence in results:
                st.write(f"**{name}** — {confidence * 100:.2f}%")
                st.progress(float(min(max(confidence, 0.0), 1.0)))

st.divider()
st.caption(
    "Assignment deployment — Fruits-360 image classification with TensorFlow and Streamlit."
)

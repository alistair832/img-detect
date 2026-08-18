from pathlib import Path
import ast
import base64
import zlib
from collections import Counter, deque

import av
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_webrtc import webrtc_streamer


st.set_page_config(
    page_title="Fruit Detection",
    page_icon="🍎",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# The working embedded model from the previous version is preserved in
# legacy_model.py. We read only its MODEL_B85 constant, so the old Streamlit UI
# is never executed.
CLASSES = ["Apple", "Banana", "Guava", "Lime", "Orange", "Pomegranate"]
FEATURE_DIM = 164
MODEL_SOURCE_PATH = Path(__file__).resolve().parent / "legacy_model.py"


def _read_model_blob() -> str:
    source = MODEL_SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MODEL_B85":
                    return ast.literal_eval(node.value)

    raise RuntimeError("Embedded fruit model could not be found.")


@st.cache_resource
def load_embedded_model():
    model_b85 = _read_model_blob()
    raw = zlib.decompress(base64.b85decode(model_b85.encode("ascii")))
    values = np.frombuffer(raw, dtype="<f2").astype(np.float32)

    offset = 0
    coef_count = len(CLASSES) * FEATURE_DIM
    coef = values[offset : offset + coef_count].reshape(len(CLASSES), FEATURE_DIM)
    offset += coef_count

    intercept = values[offset : offset + len(CLASSES)]
    offset += len(CLASSES)

    mean = values[offset : offset + FEATURE_DIM]
    offset += FEATURE_DIM

    std = values[offset : offset + FEATURE_DIM]
    std = np.where(std == 0, 1.0, std)

    return coef, intercept, mean, std


COEF, INTERCEPT, FEATURE_MEAN, FEATURE_STD = load_embedded_model()


def extract_features(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)

    feature_parts = []
    for channel, bins in ((0, 24), (1, 16), (2, 16)):
        histogram, _ = np.histogram(hsv[:, :, channel], bins=bins, range=(0, 256))
        histogram = histogram.astype(np.float32)
        histogram /= histogram.sum() + 1e-8
        feature_parts.append(histogram)

    small_rgb = np.asarray(
        image.resize((6, 6), Image.Resampling.BILINEAR), dtype=np.float32
    ).reshape(-1) / 255.0
    feature_parts.append(small_rgb)

    features = np.concatenate(feature_parts).astype(np.float32)
    return (features - FEATURE_MEAN) / FEATURE_STD


def _image_quality_checks(image: Image.Image):
    """Basic checks that help reject empty/dark/grey camera regions."""
    hsv = np.asarray(image.convert("RGB").resize((64, 64)).convert("HSV"), dtype=np.float32)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    colourful_fraction = float(np.mean(saturation > 35))
    mean_brightness = float(np.mean(value))
    brightness_std = float(np.std(value))

    if mean_brightness < 25:
        return False, "Image is too dark"
    if mean_brightness > 248 and brightness_std < 8:
        return False, "Image is almost blank"
    if colourful_fraction < 0.06 and brightness_std < 20:
        return False, "No clear fruit-like object"

    return True, ""


def predict_fruit(image: Image.Image):
    """
    Return prediction details.

    The percentage is prediction confidence for this image, not the overall
    validation accuracy of the model.
    """
    features = extract_features(image)
    scores = COEF @ features + INTERCEPT

    # Temperature softmax prevents the uncalibrated SVM scores from becoming
    # artificially close to 100% too easily.
    temperature = 1.8
    shifted = (scores - np.max(scores)) / temperature
    shifted = np.clip(shifted, -50, 50)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum() + 1e-8

    order = np.argsort(probabilities)[::-1]
    prediction_index = int(order[0])
    second_index = int(order[1])
    confidence = float(probabilities[prediction_index])
    margin = float(confidence - probabilities[second_index])

    image_ok, image_reason = _image_quality_checks(image)

    # Unknown/rejection logic: do not force every object into one fruit class.
    is_confident = image_ok and confidence >= 0.46 and margin >= 0.10

    reason = ""
    if not image_ok:
        reason = image_reason
    elif confidence < 0.46:
        reason = "Low prediction confidence"
    elif margin < 0.10:
        reason = "Two fruit classes look too similar"

    return {
        "index": prediction_index,
        "fruit": CLASSES[prediction_index],
        "confidence": confidence,
        "margin": margin,
        "probabilities": probabilities,
        "known": is_confident,
        "reason": reason,
    }


def centre_square(image: Image.Image, fraction: float = 0.68) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    size = max(1, int(min(width, height) * fraction))
    left = max(0, (width - size) // 2)
    top = max(0, (height - size) // 2)
    return image.crop((left, top, left + size, top + size))


def best_prediction_for_uploaded_image(image: Image.Image):
    """Try both the full image and a centre crop, then use the stronger result."""
    candidates = [image.convert("RGB"), centre_square(image)]
    results = [predict_fruit(candidate) for candidate in candidates]

    confident_results = [result for result in results if result["known"]]
    if confident_results:
        return max(confident_results, key=lambda item: item["confidence"])
    return max(results, key=lambda item: item["confidence"])


# -----------------------------------------------------------------------------
# Live camera
# -----------------------------------------------------------------------------
result_history = deque(maxlen=10)


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    frame_array = frame.to_ndarray(format="rgb24")
    frame_array = frame_array[:, ::-1].copy()  # mirror front camera
    image = Image.fromarray(frame_array)

    width, height = image.size
    box_size = int(min(width, height) * 0.62)
    left = max(0, (width - box_size) // 2)
    top = max(0, (height - box_size) // 2)
    right = min(width, left + box_size)
    bottom = min(height, top + box_size)

    roi = image.crop((left, top, right, bottom))
    result = predict_fruit(roi)
    result_history.append(result)

    draw = ImageDraw.Draw(image)
    draw.rectangle((left, top, right, bottom), outline=(40, 220, 90), width=5)

    if len(result_history) < 4:
        label = "Analyzing fruit..."
    else:
        known_results = [item for item in result_history if item["known"]]

        if len(known_results) >= 5:
            stable_index = Counter(item["index"] for item in known_results).most_common(1)[0][0]
            matching = [item for item in known_results if item["index"] == stable_index]

            if len(matching) >= 5:
                avg_confidence = float(np.mean([item["confidence"] for item in matching]))
                label = f"{CLASSES[stable_index]} - {avg_confidence * 100:.1f}% confidence"
            else:
                label = "Hold fruit steady in the box"
        else:
            best_guess = max(result_history, key=lambda item: item["confidence"])
            label = f"Unknown / not confident - {best_guess['confidence'] * 100:.1f}%"

    text_width = min(width - 12, 500)
    text_box = (12, 12, text_width, 66)
    draw.rounded_rectangle(text_box, radius=12, fill=(0, 0, 0))
    draw.text((25, 30), label, fill=(255, 255, 255))

    guide_text = "Place ONE fruit inside the green box"
    guide_box = (left, max(0, bottom - 36), right, bottom)
    draw.rectangle(guide_box, fill=(0, 0, 0))
    draw.text((left + 10, max(2, bottom - 26)), guide_text, fill=(255, 255, 255))

    return av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("🍎 Fruit Detection")
st.write(
    "Detect fruit using your front camera or upload a picture. The result includes "
    "a confidence percentage and rejects uncertain images as Unknown."
)

info1, info2, info3 = st.columns(3)
with info1:
    st.metric("Fruit classes", "6")
with info2:
    st.metric("Training crops", "2,009")
with info3:
    st.metric("Model validation accuracy", "66.7%")

with st.expander("Supported fruit classes"):
    st.write("Apple • Banana • Guava • Lime • Orange • Pomegranate")
    st.caption(
        "The percentage beside a scan is prediction confidence. The 66.7% value above "
        "is the model's validation accuracy across the supplied dataset."
    )

camera_tab, upload_tab = st.tabs(["🎥 Live Front Camera", "🖼️ Upload Picture"])

with camera_tab:
    st.subheader("Live Front Camera")
    st.caption(
        "Click START, allow camera permission, then hold one fruit inside the green "
        "square. The fruit name and confidence update continuously."
    )

    webrtc_streamer(
        key="live-fruit-camera",
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

    st.info(
        "If the camera sees an unrelated object or the model is unsure, it will show "
        "Unknown / not confident instead of forcing a fruit label."
    )

with upload_tab:
    st.subheader("Upload a Fruit Picture")
    st.caption(
        "Use this when you do not have the real fruit available. For best results, "
        "choose a clear image with one main fruit near the centre."
    )

    uploaded_file = st.file_uploader(
        "Choose JPG, JPEG, PNG, or WEBP",
        type=["jpg", "jpeg", "png", "webp"],
        key="fruit-image-upload",
    )

    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(uploaded_file).convert("RGB")
            result = best_prediction_for_uploaded_image(uploaded_image)

            image_col, result_col = st.columns([1, 1])

            with image_col:
                st.image(uploaded_image, caption="Uploaded image", use_container_width=True)

            with result_col:
                if result["known"]:
                    st.success(f"Detected fruit: **{result['fruit']}**")
                    st.metric("Prediction confidence", f"{result['confidence'] * 100:.1f}%")
                else:
                    st.warning("Result: **Unknown / not confident**")
                    st.metric(
                        f"Best guess: {result['fruit']}",
                        f"{result['confidence'] * 100:.1f}%",
                    )
                    if result["reason"]:
                        st.caption(result["reason"])

                st.subheader("Top 3 predictions")
                order = np.argsort(result["probabilities"])[::-1][:3]
                for index in order:
                    score = float(result["probabilities"][index])
                    st.write(f"**{CLASSES[int(index)]}** — {score * 100:.1f}%")
                    st.progress(min(max(score, 0.0), 1.0))

        except Exception as exc:
            st.error(f"Could not process this picture: {exc}")
    else:
        st.info("Upload a fruit picture to test the model without using the camera.")

st.divider()
st.caption(
    "Artificial Intelligence project — fruit recognition trained from the supplied fruit dataset."
)

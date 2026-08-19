from pathlib import Path
from io import BytesIO
import base64
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
# Upgraded compact model
# -----------------------------------------------------------------------------
# Trained from the user's supplied YOLO fruit dataset. The original quality
# labels are merged into six fruit types for this application.
CLASSES = ["Apple", "Banana", "Guava", "Lime", "Orange", "Pomegranate"]
MODEL_VALIDATION_ACCURACY = 80.3
TRAINING_CROPS = 2120
APP_DIR = Path(__file__).resolve().parent
MODEL_PART_DIR = APP_DIR / "model_data"
MODEL_PARTS = [MODEL_PART_DIR / f"hgb80_{i:02d}.txt" for i in range(9)]


@st.cache_resource(show_spinner="Loading upgraded fruit model...")
def load_model():
    missing = [path.name for path in MODEL_PARTS if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Upgraded model data is incomplete. Missing: " + ", ".join(missing)
        )

    encoded = "".join(path.read_text(encoding="utf-8").strip() for path in MODEL_PARTS)
    raw = base64.b85decode(encoded.encode("ascii"))

    with np.load(BytesIO(raw), allow_pickle=False) as pack:
        nodes = pack["nodes"].copy()
        roots = pack["roots"].copy()
        tree_classes = pack["classes"].copy()
        baseline = pack["baseline"].astype(np.float32).copy()

    return nodes, roots, tree_classes, baseline


NODES, ROOTS, TREE_CLASSES, BASELINE = load_model()


def _normalised_histogram(channel: np.ndarray, bins: int) -> np.ndarray:
    histogram, _ = np.histogram(channel, bins=bins, range=(0, 256))
    histogram = histogram.astype(np.float32)
    histogram /= histogram.sum() + 1e-8
    return histogram


def extract_features(image: Image.Image) -> np.ndarray:
    """Extract the same 304 colour/spatial features used by the upgraded model."""
    image = image.convert("RGB").resize((96, 96), Image.Resampling.BILINEAR)

    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
    ycbcr = np.asarray(image.convert("YCbCr"), dtype=np.uint8)

    parts = [
        _normalised_histogram(hsv[:, :, 0], 32),
        _normalised_histogram(hsv[:, :, 1], 24),
        _normalised_histogram(hsv[:, :, 2], 24),
        _normalised_histogram(ycbcr[:, :, 1], 16),
        _normalised_histogram(ycbcr[:, :, 2], 16),
    ]

    spatial_rgb = np.asarray(
        image.resize((8, 8), Image.Resampling.BILINEAR), dtype=np.float32
    ).reshape(-1) / 255.0
    parts.append(spatial_rgb)

    return np.concatenate(parts).astype(np.float32)


def model_scores(features: np.ndarray) -> np.ndarray:
    """Run the compact histogram-gradient-boosting trees using only NumPy."""
    scores = BASELINE.astype(np.float32).copy()

    for root, class_index in zip(ROOTS, TREE_CLASSES):
        node_index = int(root)

        while int(NODES["f"][node_index]) >= 0:
            feature_index = int(NODES["f"][node_index])
            threshold = float(NODES["t"][node_index])

            if float(features[feature_index]) <= threshold:
                node_index += int(NODES["l"][node_index])
            else:
                node_index += int(NODES["r"][node_index])

        scores[int(class_index)] += float(NODES["v"][node_index])

    return scores


def _image_quality_checks(image: Image.Image):
    """Reject obviously blank, dark, or low-information regions."""
    hsv = np.asarray(
        image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR).convert("HSV"),
        dtype=np.float32,
    )
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    colourful_fraction = float(np.mean(saturation > 35))
    mean_brightness = float(np.mean(value))
    brightness_std = float(np.std(value))

    if mean_brightness < 25:
        return False, "Image is too dark"
    if mean_brightness > 248 and brightness_std < 8:
        return False, "Image is almost blank"
    if colourful_fraction < 0.05 and brightness_std < 18:
        return False, "No clear fruit-like object"

    return True, ""


def predict_fruit(image: Image.Image):
    features = extract_features(image)
    scores = model_scores(features)

    # A mild temperature keeps the displayed confidence from becoming
    # unrealistically close to 100% for ambiguous samples.
    temperature = 1.25
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
    is_confident = image_ok and confidence >= 0.42 and margin >= 0.07

    reason = ""
    if not image_ok:
        reason = image_reason
    elif confidence < 0.42:
        reason = "Low prediction confidence"
    elif margin < 0.07:
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


def centre_square(image: Image.Image, fraction: float) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    size = max(1, int(min(width, height) * fraction))
    left = max(0, (width - size) // 2)
    top = max(0, (height - size) // 2)
    return image.crop((left, top, left + size, top + size))


def best_prediction_for_uploaded_image(image: Image.Image):
    """Try several centre crops because training samples use annotated fruit boxes."""
    candidates = [
        image.convert("RGB"),
        centre_square(image, 0.82),
        centre_square(image, 0.68),
    ]
    results = [predict_fruit(candidate) for candidate in candidates]

    confident_results = [result for result in results if result["known"]]
    if confident_results:
        return max(confident_results, key=lambda item: item["confidence"])
    return max(results, key=lambda item: item["confidence"])


# -----------------------------------------------------------------------------
# Live camera
# -----------------------------------------------------------------------------
result_history = deque(maxlen=12)


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    frame_array = frame.to_ndarray(format="rgb24")
    frame_array = frame_array[:, ::-1].copy()
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

    if len(result_history) < 5:
        label = "Analyzing fruit..."
    else:
        recent = list(result_history)[-10:]
        known_results = [item for item in recent if item["known"]]

        if len(known_results) >= 5:
            stable_index = Counter(
                item["index"] for item in known_results
            ).most_common(1)[0][0]
            matching = [
                item for item in known_results if item["index"] == stable_index
            ]

            if len(matching) >= 5:
                avg_confidence = float(
                    np.mean([item["confidence"] for item in matching[-8:]])
                )
                label = (
                    f"{CLASSES[stable_index]} - "
                    f"{avg_confidence * 100:.1f}% confidence"
                )
            else:
                label = "Hold fruit steady in the box"
        else:
            best_guess = max(recent, key=lambda item: item["confidence"])
            label = (
                "Unknown / not confident - "
                f"{best_guess['confidence'] * 100:.1f}%"
            )

    text_width = min(width - 12, 520)
    draw.rounded_rectangle((12, 12, text_width, 66), radius=12, fill=(0, 0, 0))
    draw.text((25, 30), label, fill=(255, 255, 255))

    guide_text = "Place ONE fruit inside the green box"
    draw.rectangle((left, max(0, bottom - 36), right, bottom), fill=(0, 0, 0))
    draw.text((left + 10, max(2, bottom - 26)), guide_text, fill=(255, 255, 255))

    return av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("🍎 Fruit Detection")
st.write(
    "Detect fruit using your front camera or upload a picture. The upgraded model "
    "uses richer colour and spatial features for better recognition."
)

info1, info2, info3 = st.columns(3)
with info1:
    st.metric("Fruit classes", "6")
with info2:
    st.metric("Training crops", f"{TRAINING_CROPS:,}")
with info3:
    st.metric("Model validation accuracy", f"{MODEL_VALIDATION_ACCURACY:.1f}%", "+13.6 pts")

st.success("Model upgraded: validation accuracy improved from 66.7% to 80.3%.")

with st.expander("Supported fruit classes"):
    st.write("Apple • Banana • Guava • Lime • Orange • Pomegranate")
    st.caption(
        "The percentage beside one scan is prediction confidence. The 80.3% value "
        "above is validation accuracy measured across the supplied validation set."
    )

camera_tab, upload_tab = st.tabs(["🎥 Live Front Camera", "🖼️ Upload Picture"])

with camera_tab:
    st.subheader("Live Front Camera")
    st.caption(
        "Click START, allow camera permission, then hold one fruit inside the green "
        "square. Good lighting and a fruit that fills most of the box give the best result."
    )

    webrtc_streamer(
        key="live-fruit-camera-v2",
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
        "The prediction is smoothed across several frames. If the model is unsure, "
        "it shows Unknown / not confident instead of forcing a fruit label."
    )

with upload_tab:
    st.subheader("Upload a Fruit Picture")
    st.caption(
        "Use this when you do not have a real fruit available. Choose a clear picture "
        "with one main fruit near the centre."
    )

    uploaded_file = st.file_uploader(
        "Choose JPG, JPEG, PNG, or WEBP",
        type=["jpg", "jpeg", "png", "webp"],
        key="fruit-image-upload-v2",
    )

    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(uploaded_file).convert("RGB")
            result = best_prediction_for_uploaded_image(uploaded_image)

            image_col, result_col = st.columns([1, 1])

            with image_col:
                st.image(
                    uploaded_image,
                    caption="Uploaded image",
                    use_container_width=True,
                )

            with result_col:
                if result["known"]:
                    st.success(f"Detected fruit: **{result['fruit']}**")
                    st.metric(
                        "Prediction confidence",
                        f"{result['confidence'] * 100:.1f}%",
                    )
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
        st.info("Upload a fruit picture to test the upgraded model without using the camera.")

st.divider()
st.caption(
    "Artificial Intelligence project — fruit recognition trained from the supplied fruit dataset."
)

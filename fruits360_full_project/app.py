from __future__ import annotations

import base64
from collections import Counter, deque
from io import BytesIO
from pathlib import Path

import av
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_webrtc import webrtc_streamer

st.set_page_config(
    page_title="Fruit Image Detection",
    page_icon="🍎",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
MODEL_DIR = REPO_ROOT / "model_data"
MODEL_PARTS = [MODEL_DIR / f"hgb80_{i:02d}.txt" for i in range(9)]

CLASSES = ["Apple", "Banana", "Guava", "Lime", "Orange", "Pomegranate"]
APPLE, BANANA, GUAVA, LIME, ORANGE, POMEGRANATE = range(6)


@st.cache_resource(show_spinner="Loading fruit detector...")
def load_model():
    missing = [path.name for path in MODEL_PARTS if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Saved detector is incomplete. Missing: " + ", ".join(missing)
        )

    encoded = "".join(
        path.read_text(encoding="utf-8").strip() for path in MODEL_PARTS
    )
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
    return histogram / (histogram.sum() + 1e-8)


def extract_features(image: Image.Image) -> np.ndarray:
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
        image.resize((8, 8), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ).reshape(-1) / 255.0
    parts.append(spatial_rgb)

    return np.concatenate(parts).astype(np.float32)


def model_scores(features: np.ndarray) -> np.ndarray:
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


def softmax(scores: np.ndarray, temperature: float = 1.25) -> np.ndarray:
    shifted = (scores - np.max(scores)) / temperature
    shifted = np.clip(shifted, -50, 50)
    probabilities = np.exp(shifted)
    return probabilities / (probabilities.sum() + 1e-8)


def apple_visual_evidence(image: Image.Image) -> dict[str, float]:
    """Visual cues used only to reduce Apple/Lime/Pomegranate confusion."""
    rgb = np.asarray(
        image.convert("RGB").resize((160, 160), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.uint8)

    red_channel = rgb[:, :, 0].astype(np.float32)
    green_channel = rgb[:, :, 1].astype(np.float32)
    blue_channel = rgb[:, :, 2].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    object_mask = (value < 247) | (saturation > 24)
    denominator = float(max(int(object_mask.sum()), 1))

    red_mask = (
        object_mask
        & (red_channel > 125)
        & (red_channel > green_channel * 1.12)
        & (red_channel > blue_channel * 1.08)
    )
    green_mask = (
        object_mask
        & (green_channel > 85)
        & (green_channel > red_channel * 1.04)
        & (green_channel > blue_channel * 1.08)
    )
    cream_mask = (
        object_mask
        & (red_channel > 175)
        & (green_channel > 150)
        & (blue_channel > 105)
        & (saturation < 105)
        & (value > 165)
    )
    moderate_green_mask = green_mask & (saturation < 165) & (value > 105)

    return {
        "red_fraction": float(red_mask.sum()) / denominator,
        "green_fraction": float(green_mask.sum()) / denominator,
        "cream_fraction": float(cream_mask.sum()) / denominator,
        "moderate_green_fraction": float(moderate_green_mask.sum()) / denominator,
        "mean_saturation": (
            float(saturation[object_mask].mean()) if object_mask.any() else 0.0
        ),
    }


def apply_apple_correction(
    probabilities: np.ndarray,
    image: Image.Image,
) -> tuple[np.ndarray, str]:
    """Keep the saved model primary while reducing common Apple confusions."""
    adjusted = probabilities.astype(np.float32).copy()
    evidence = apple_visual_evidence(image)
    reason = ""

    if evidence["cream_fraction"] >= 0.08 and evidence["red_fraction"] >= 0.08:
        adjusted[APPLE] *= 3.8
        adjusted[POMEGRANATE] *= 0.38
        reason = "Apple correction: pale flesh + red skin detected."
    elif (
        evidence["moderate_green_fraction"] >= 0.16
        and evidence["mean_saturation"] < 155
    ):
        adjusted[APPLE] *= 2.15
        adjusted[LIME] *= 0.72
        reason = "Apple correction: bright, moderately saturated green detected."

    adjusted /= adjusted.sum() + 1e-8
    return adjusted, reason


def image_quality(image: Image.Image) -> tuple[bool, str]:
    hsv = np.asarray(
        image.convert("RGB")
        .resize((64, 64), Image.Resampling.BILINEAR)
        .convert("HSV"),
        dtype=np.float32,
    )
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    colourful_fraction = float(np.mean(saturation > 30))
    mean_brightness = float(np.mean(value))
    brightness_std = float(np.std(value))

    if mean_brightness < 22:
        return False, "Image is too dark."
    if mean_brightness > 249 and brightness_std < 7:
        return False, "Image is almost blank."
    if colourful_fraction < 0.035 and brightness_std < 14:
        return False, "No clear fruit-like object was found."

    return True, ""


def predict_one(image: Image.Image) -> dict:
    probabilities = softmax(model_scores(extract_features(image)))
    probabilities, correction_reason = apply_apple_correction(probabilities, image)

    order = np.argsort(probabilities)[::-1]
    first = int(order[0])
    second = int(order[1])
    confidence = float(probabilities[first])
    margin = float(confidence - probabilities[second])

    image_ok, quality_reason = image_quality(image)
    known = image_ok and confidence >= 0.40 and margin >= 0.055

    if not image_ok:
        reason = quality_reason
    elif confidence < 0.40:
        reason = "Prediction confidence is low."
    elif margin < 0.055:
        reason = "The top fruit classes are too similar."
    else:
        reason = correction_reason

    return {
        "index": first,
        "fruit": CLASSES[first],
        "confidence": confidence,
        "probabilities": probabilities,
        "known": known,
        "reason": reason,
    }


def centre_square(image: Image.Image, fraction: float = 1.0) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    side = max(1, int(min(width, height) * fraction))
    left = max(0, (width - side) // 2)
    top = max(0, (height - side) // 2)
    return image.crop((left, top, left + side, top + side))


def best_prediction(image: Image.Image) -> dict:
    candidates = [
        image.convert("RGB"),
        centre_square(image, 1.00),
        centre_square(image, 0.84),
        centre_square(image, 0.68),
    ]
    results = [predict_one(candidate) for candidate in candidates]

    confident = [result for result in results if result["known"]]
    if confident:
        return max(
            confident,
            key=lambda result: result["confidence"]
            + (0.035 if result["fruit"] == "Apple" else 0.0),
        )

    return max(results, key=lambda result: result["confidence"])


def show_result(image: Image.Image):
    result = best_prediction(image)
    probabilities = result["probabilities"]

    image_col, result_col = st.columns([1.05, 1])

    with image_col:
        st.image(image, caption="Selected image", use_container_width=True)

    with result_col:
        if result["known"]:
            st.success(f"Detected fruit: **{result['fruit']}**")
        else:
            st.warning(f"Best guess: **{result['fruit']}**")

        if result["reason"]:
            st.caption(result["reason"])

        st.metric("Prediction confidence", f"{result['confidence'] * 100:.1f}%")

        st.subheader("Top 3 predictions")
        order = np.argsort(probabilities)[::-1][:3]
        for index in order:
            score = float(probabilities[int(index)])
            st.write(f"**{CLASSES[int(index)]}** — {score * 100:.1f}%")
            st.progress(float(min(max(score, 0.0), 1.0)))


# -----------------------------------------------------------------------------
# Live front-camera detection
# -----------------------------------------------------------------------------
LIVE_HISTORY = deque(maxlen=12)


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    rgb = frame.to_ndarray(format="rgb24")
    # Mirror the front camera so it behaves naturally like a selfie camera.
    rgb = rgb[:, ::-1].copy()
    image = Image.fromarray(rgb)

    width, height = image.size
    box_size = int(min(width, height) * 0.62)
    left = max(0, (width - box_size) // 2)
    top = max(0, (height - box_size) // 2)
    right = min(width, left + box_size)
    bottom = min(height, top + box_size)

    roi = image.crop((left, top, right, bottom))
    result = predict_one(roi)
    LIVE_HISTORY.append(result)

    if len(LIVE_HISTORY) < 4:
        label = "Analyzing fruit..."
    else:
        recent = list(LIVE_HISTORY)[-10:]
        known = [item for item in recent if item["known"]]

        if len(known) >= 4:
            stable_index = Counter(item["index"] for item in known).most_common(1)[0][0]
            matching = [item for item in known if item["index"] == stable_index]
            if len(matching) >= 3:
                avg_confidence = float(
                    np.mean([item["confidence"] for item in matching[-6:]])
                )
                label = f"{CLASSES[stable_index]} — {avg_confidence * 100:.1f}%"
            else:
                label = "Hold fruit steady in the green box"
        else:
            best = max(recent, key=lambda item: item["confidence"])
            label = f"Unknown / best guess {best['fruit']} — {best['confidence'] * 100:.1f}%"

    draw = ImageDraw.Draw(image)
    draw.rectangle((left, top, right, bottom), outline=(40, 220, 90), width=5)

    text_right = min(width - 12, max(360, width - 12))
    draw.rounded_rectangle((12, 12, text_right, 66), radius=12, fill=(0, 0, 0))
    draw.text((24, 31), label, fill=(255, 255, 255))

    guide_top = max(top, bottom - 38)
    draw.rectangle((left, guide_top, right, bottom), fill=(0, 0, 0))
    draw.text(
        (left + 10, guide_top + 10),
        "Place ONE fruit inside the green box",
        fill=(255, 255, 255),
    )

    return av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")


st.title("🍎 Fruit Image Detection")
st.caption("Live front-camera + image upload detection only — no training page.")

st.info(
    "Supported fruits: **Apple, Banana, Guava, Lime, Orange, Pomegranate**. "
    "Apple recognition includes extra checks to reduce Apple → Pomegranate "
    "and Apple → Lime mistakes."
)

camera_tab, upload_tab = st.tabs(["🎥 Live Front Camera", "🖼️ Upload Image"])

with camera_tab:
    st.subheader("Live Front Camera")
    st.write(
        "Press **START**, allow camera permission, then hold one fruit inside the "
        "green square. The prediction updates live and is smoothed across frames."
    )

    webrtc_streamer(
        key="fruit-live-front-camera-v4",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "facingMode": "user",
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 20, "max": 24},
            },
            "audio": False,
        },
        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302",
                        "stun:stun1.l.google.com:19302",
                        "stun:stun2.l.google.com:19302",
                    ]
                },
                {"urls": ["stun:stun.cloudflare.com:3478"]},
            ]
        },
        async_processing=True,
    )

    st.caption(
        "If the browser asks for permission, choose Allow. For best accuracy, "
        "use good lighting and let the fruit fill most of the green box."
    )

with upload_tab:
    st.subheader("Upload a Fruit Image")
    uploaded = st.file_uploader(
        "Choose JPG, JPEG, PNG or WEBP",
        type=["jpg", "jpeg", "png", "webp"],
        key="fruit-detection-upload-v4",
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        show_result(image)

st.divider()
st.caption(
    "Detection-only deployment: live front camera and upload image. "
    "No model-training interface is shown in Streamlit."
)

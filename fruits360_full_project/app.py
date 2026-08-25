from __future__ import annotations

import base64
import threading
import time
from collections import Counter, deque
from io import BytesIO
from pathlib import Path

import av
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_webrtc import webrtc_streamer

st.set_page_config(page_title="Fruit Image Detection", page_icon="🍎", layout="wide")

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
MODEL_DIR = REPO_ROOT / "model_data"
MODEL_PARTS = [MODEL_DIR / f"hgb80_{i:02d}.txt" for i in range(9)]
CLASSES = ["Apple", "Banana", "Guava", "Lime", "Orange", "Pomegranate"]
APPLE, BANANA, GUAVA, LIME, ORANGE, POMEGRANATE = range(6)


@st.cache_resource(show_spinner="Loading fruit detector...")
def load_model():
    missing = [p.name for p in MODEL_PARTS if not p.exists()]
    if missing:
        raise FileNotFoundError("Saved detector incomplete: " + ", ".join(missing))
    encoded = "".join(p.read_text(encoding="utf-8").strip() for p in MODEL_PARTS)
    raw = base64.b85decode(encoded.encode("ascii"))
    with np.load(BytesIO(raw), allow_pickle=False) as pack:
        return (
            pack["nodes"].copy(),
            pack["roots"].copy(),
            pack["classes"].copy(),
            pack["baseline"].astype(np.float32).copy(),
        )


NODES, ROOTS, TREE_CLASSES, BASELINE = load_model()


def norm_hist(channel, bins):
    histogram, _ = np.histogram(channel, bins=bins, range=(0, 256))
    histogram = histogram.astype(np.float32)
    return histogram / (histogram.sum() + 1e-8)


def extract_features(image):
    image = image.convert("RGB").resize((96, 96), Image.Resampling.BILINEAR)
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
    ycbcr = np.asarray(image.convert("YCbCr"), dtype=np.uint8)
    parts = [
        norm_hist(hsv[:, :, 0], 32),
        norm_hist(hsv[:, :, 1], 24),
        norm_hist(hsv[:, :, 2], 24),
        norm_hist(ycbcr[:, :, 1], 16),
        norm_hist(ycbcr[:, :, 2], 16),
        np.asarray(
            image.resize((8, 8), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ).reshape(-1)
        / 255.0,
    ]
    return np.concatenate(parts).astype(np.float32)


def model_scores(features):
    scores = BASELINE.astype(np.float32).copy()
    for root, class_index in zip(ROOTS, TREE_CLASSES):
        node = int(root)
        while int(NODES["f"][node]) >= 0:
            feature_index = int(NODES["f"][node])
            if float(features[feature_index]) <= float(NODES["t"][node]):
                node += int(NODES["l"][node])
            else:
                node += int(NODES["r"][node])
        scores[int(class_index)] += float(NODES["v"][node])
    return scores


def softmax(scores, temperature=1.25):
    z = np.clip((scores - np.max(scores)) / temperature, -50, 50)
    probabilities = np.exp(z)
    return probabilities / (probabilities.sum() + 1e-8)


def components(mask, min_area=1, max_area=None):
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    found = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue

            stack = [(y, x)]
            seen[y, x] = True
            xs, ys = [], []

            while stack:
                cy, cx = stack.pop()
                xs.append(cx)
                ys.append(cy)

                for ny, nx in (
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                ):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not seen[ny, nx]
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))

            area = len(xs)
            if area >= min_area and (max_area is None or area <= max_area):
                found.append(
                    {
                        "area": area,
                        "left": min(xs),
                        "right": max(xs),
                        "top": min(ys),
                        "bottom": max(ys),
                    }
                )
    return found


def structural_measurements(image):
    """
    Measure explainable visual features after resizing to 160x160.

    Length/width are image-relative pixels, not real centimetres.
    HSV hue values use Pillow's 0-255 hue scale.
    """
    img = image.convert("RGB").resize((160, 160), Image.Resampling.BILINEAR)
    rgb = np.asarray(img, dtype=np.uint8)
    hsv = np.asarray(img.convert("HSV"), dtype=np.uint8)

    r, g, b = [rgb[:, :, i].astype(np.float32) for i in range(3)]
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    # Main fruit region. Fruits-360 commonly has a light/white background.
    candidate = (sat > 32) | (val < 232)
    candidate[:5, :] = False
    candidate[-5:, :] = False
    candidate[:, :5] = False
    candidate[:, -5:] = False
    comps = components(candidate, 80)

    if comps:
        def comp_score(component):
            center = not (
                component["right"] < 48
                or component["left"] > 112
                or component["bottom"] < 48
                or component["top"] > 112
            )
            return component["area"] * (1.8 if center else 1.0)

        main = max(comps, key=comp_score)
        left = main["left"]
        right = main["right"]
        top = main["top"]
        bottom = main["bottom"]
        object_area = float(main["area"])
    else:
        left = top = 0
        right = bottom = 159
        object_area = float(candidate.sum())

    box_width = max(1, right - left + 1)
    box_height = max(1, bottom - top + 1)
    length_px = max(box_width, box_height)
    width_px = min(box_width, box_height)
    aspect = float(length_px / max(width_px, 1))
    fill = float(
        min(1.0, object_area / max(float(box_width * box_height), 1.0))
    )

    roi = np.zeros((160, 160), dtype=bool)
    roi[top : bottom + 1, left : right + 1] = True
    denom = float(max(int(roi.sum()), 1))

    red = (
        roi
        & (r > 120)
        & (r > g * 1.10)
        & (r > b * 1.08)
        & (sat > 50)
    )
    green = (
        roi
        & (g > 80)
        & (g > r * 1.03)
        & (g > b * 1.08)
        & (sat > 35)
    )
    pale = (
        roi
        & (r > 165)
        & (g > 145)
        & (b > 95)
        & (sat < 115)
        & (val > 155)
    )

    # Orange peel/flesh: orange hue, strong saturation and medium/high brightness.
    orange_colour = (
        roi
        & (hue >= 7)
        & (hue <= 34)
        & (sat >= 80)
        & (val >= 85)
        & (r > g * 1.02)
    )
    orange_flesh = (
        roi
        & (hue >= 8)
        & (hue <= 38)
        & (sat >= 35)
        & (sat <= 210)
        & (val >= 130)
        & (r >= g)
        & (g > b * 1.10)
    )

    # Lime colour: saturated yellow-green/green. It is deliberately narrower
    # than generic "green" so green apples are not automatically called Lime.
    lime_green = (
        roi
        & (hue >= 42)
        & (hue <= 105)
        & (sat >= 75)
        & (val >= 70)
        & (g > r * 1.04)
        & (g > b * 1.10)
    )
    lime_flesh = (
        roi
        & (hue >= 35)
        & (hue <= 95)
        & (sat >= 25)
        & (sat <= 170)
        & (val >= 125)
        & (g > r * 1.015)
        & (g > b * 1.06)
    )

    # Apple seeds are usually only a few compact dark spots. A cut pomegranate
    # often exposes many compact red/dark seed/aril-like regions.
    dark_seed = roi & (val >= 30) & (val < 125) & (sat > 35)
    red_aril = (
        roi
        & (((hue < 14) | (hue > 238)))
        & (sat > 105)
        & (val > 75)
        & (val < 235)
    )
    seed_candidates = components(dark_seed | red_aril, 4, 190)

    seed_count = 0
    for component in seed_candidates:
        component_width = component["right"] - component["left"] + 1
        component_height = component["bottom"] - component["top"] + 1
        component_aspect = max(component_width, component_height) / max(
            1, min(component_width, component_height)
        )
        if (
            component_width <= 24
            and component_height <= 24
            and component_aspect <= 3.0
        ):
            seed_count += 1

    return {
        "length_px": int(length_px),
        "width_px": int(width_px),
        "aspect_ratio": aspect,
        "fill_ratio": fill,
        "seed_count": int(seed_count),
        "red_fraction": float(red.sum()) / denom,
        "green_fraction": float(green.sum()) / denom,
        "pale_flesh_fraction": float(pale.sum()) / denom,
        "orange_fraction": float(orange_colour.sum()) / denom,
        "orange_flesh_fraction": float(orange_flesh.sum()) / denom,
        "lime_green_fraction": float(lime_green.sum()) / denom,
        "lime_flesh_fraction": float(lime_flesh.sum()) / denom,
        "mean_saturation": (
            float(sat[roi].mean()) if roi.any() else 0.0
        ),
    }


def apply_structure_rules(probabilities, image):
    p = probabilities.astype(np.float32).copy()
    measurements = structural_measurements(image)

    seeds = int(measurements["seed_count"])
    aspect = float(measurements["aspect_ratio"])
    fill = float(measurements["fill_ratio"])
    pale = float(measurements["pale_flesh_fraction"])
    red = float(measurements["red_fraction"])
    green = float(measurements["green_fraction"])
    orange_area = float(measurements["orange_fraction"])
    orange_flesh = float(measurements["orange_flesh_fraction"])
    lime_green = float(measurements["lime_green_fraction"])
    lime_flesh = float(measurements["lime_flesh_fraction"])
    mean_saturation = float(measurements["mean_saturation"])
    reasons = []

    # Pomegranate: many visible seeds/arils together with a red interior.
    if seeds >= 12 and red >= 0.10:
        p[POMEGRANATE] *= 5.0
        p[APPLE] *= 0.35
        p[GUAVA] *= 0.70
        reasons.append(f"many seed-like regions ({seeds})")
    elif seeds >= 8 and red >= 0.06:
        p[POMEGRANATE] *= 2.8
        p[APPLE] *= 0.65
        reasons.append(f"several seed-like regions ({seeds})")

    # Apple: neutral pale cut flesh + only a few seeds + red/green peel.
    # Do not fire this rule when the flesh itself looks strongly citrus-coloured.
    if (
        seeds <= 6
        and pale >= 0.16
        and (green >= 0.06 or red >= 0.06)
        and lime_flesh < 0.14
        and orange_flesh < 0.16
    ):
        p[APPLE] *= 4.2
        p[POMEGRANATE] *= 0.32
        p[GUAVA] *= 0.55
        p[LIME] *= 0.55
        reasons.append(f"pale flesh with only {seeds} seed-like spots")

    # Banana: clearly elongated silhouette.
    if aspect >= 1.55:
        p[BANANA] *= 3.0
        p[LIME] *= 0.70
        p[ORANGE] *= 0.70
        p[POMEGRANATE] *= 0.70
        reasons.append(f"elongated shape L/W={aspect:.2f}")
    elif aspect >= 1.35:
        p[BANANA] *= 1.6

    # Orange: round/compact shape plus a large orange-colour region.
    # The base classifier must still have some Orange support for the strongest rule.
    if (
        aspect <= 1.28
        and fill >= 0.42
        and orange_area >= 0.24
        and mean_saturation >= 85
        and probabilities[ORANGE] >= 0.06
    ):
        p[ORANGE] *= 4.5
        p[APPLE] *= 0.62
        p[LIME] *= 0.45
        p[GUAVA] *= 0.72
        reasons.append(
            f"round shape + orange colour ({orange_area * 100:.0f}% area)"
        )
    elif (
        aspect <= 1.33
        and orange_area >= 0.14
        and probabilities[ORANGE] >= 0.10
    ):
        p[ORANGE] *= 2.2
        reasons.append(
            f"orange-colour support ({orange_area * 100:.0f}% area)"
        )

    # Cut orange: orange-coloured flesh can support Orange even if peel is
    # partly outside the crop.
    if (
        aspect <= 1.38
        and orange_flesh >= 0.20
        and probabilities[ORANGE] >= 0.05
    ):
        p[ORANGE] *= 2.0
        p[LIME] *= 0.72
        reasons.append(
            f"orange-coloured flesh ({orange_flesh * 100:.0f}% area)"
        )

    # Lime: usually round/compact with a strong saturated yellow-green/green area.
    # Require low neutral-pale flesh for a whole-lime rule so a green apple is
    # less likely to be forced into Lime.
    if (
        aspect <= 1.28
        and fill >= 0.40
        and lime_green >= 0.28
        and pale < 0.12
        and orange_area < 0.08
        and mean_saturation >= 90
        and probabilities[LIME] >= 0.06
    ):
        p[LIME] *= 4.2
        p[APPLE] *= 0.55
        p[GUAVA] *= 0.70
        p[ORANGE] *= 0.48
        reasons.append(
            f"round shape + saturated lime-green ({lime_green * 100:.0f}% area)"
        )
    elif (
        aspect <= 1.33
        and lime_green >= 0.18
        and pale < 0.10
        and probabilities[LIME] >= 0.10
    ):
        p[LIME] *= 2.0
        reasons.append(
            f"lime-green colour support ({lime_green * 100:.0f}% area)"
        )

    # Cut lime: green-tinted citrus flesh + green peel is a useful cue.
    if (
        aspect <= 1.38
        and lime_flesh >= 0.16
        and lime_green >= 0.06
        and probabilities[LIME] >= 0.05
    ):
        p[LIME] *= 2.3
        p[APPLE] *= 0.60
        p[ORANGE] *= 0.75
        reasons.append(
            f"green citrus flesh ({lime_flesh * 100:.0f}% area)"
        )

    # Gentle support for whole green apple; green alone is not decisive.
    if (
        seeds <= 6
        and pale >= 0.08
        and green >= 0.18
        and lime_flesh < 0.12
        and aspect < 1.35
    ):
        p[APPLE] *= 1.7
        p[GUAVA] *= 0.82
        p[LIME] *= 0.82

    p /= p.sum() + 1e-8
    return p, "; ".join(reasons), measurements


def image_quality(image):
    hsv = np.asarray(
        image.convert("RGB")
        .resize((64, 64), Image.Resampling.BILINEAR)
        .convert("HSV"),
        dtype=np.float32,
    )
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    colourful = float(np.mean(sat > 30))
    mean_value = float(np.mean(val))
    value_std = float(np.std(val))

    if mean_value < 22:
        return False, "Image is too dark."
    if mean_value > 249 and value_std < 7:
        return False, "Image is almost blank."
    if colourful < 0.035 and value_std < 14:
        return False, "No clear fruit-like object was found."
    return True, ""


def predict_one(image):
    base = softmax(model_scores(extract_features(image)))
    probabilities, rule_reason, measurements = apply_structure_rules(base, image)
    order = np.argsort(probabilities)[::-1]
    first = int(order[0])
    second = int(order[1])
    confidence = float(probabilities[first])
    margin = confidence - float(probabilities[second])

    ok, quality_reason = image_quality(image)
    known = ok and confidence >= 0.40 and margin >= 0.055

    if not ok:
        reason = quality_reason
    elif confidence < 0.40:
        reason = "Prediction confidence is low."
    elif margin < 0.055:
        reason = "Top fruit classes are too similar."
    else:
        reason = rule_reason

    return {
        "index": first,
        "fruit": CLASSES[first],
        "confidence": confidence,
        "probabilities": probabilities,
        "known": known,
        "reason": reason,
        "measurements": measurements,
    }


def centre_square(image, fraction=1.0):
    image = image.convert("RGB")
    width, height = image.size
    side = max(1, int(min(width, height) * fraction))
    left = max(0, (width - side) // 2)
    top = max(0, (height - side) // 2)
    return image.crop((left, top, left + side, top + side))


def best_prediction(image):
    candidates = [
        image.convert("RGB"),
        centre_square(image, 1.0),
        centre_square(image, 0.84),
        centre_square(image, 0.68),
    ]
    results = [predict_one(candidate) for candidate in candidates]
    confident = [result for result in results if result["known"]]
    return max(confident or results, key=lambda result: result["confidence"])


def show_measurements(measurements):
    with st.expander("🔎 Algorithm measurements", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Length", f"{int(measurements['length_px'])} px")
        c2.metric("Width", f"{int(measurements['width_px'])} px")
        c3.metric(
            "Length / Width",
            f"{float(measurements['aspect_ratio']):.2f}",
        )
        c4.metric("Seed-like spots", str(int(measurements["seed_count"])))

        st.write(
            f"**Pale flesh:** {float(measurements['pale_flesh_fraction']) * 100:.1f}%  ·  "
            f"**Red area:** {float(measurements['red_fraction']) * 100:.1f}%  ·  "
            f"**Green area:** {float(measurements['green_fraction']) * 100:.1f}%"
        )
        st.write(
            f"**Orange area:** {float(measurements['orange_fraction']) * 100:.1f}%  ·  "
            f"**Orange flesh:** {float(measurements['orange_flesh_fraction']) * 100:.1f}%  ·  "
            f"**Lime-green area:** {float(measurements['lime_green_fraction']) * 100:.1f}%  ·  "
            f"**Lime flesh:** {float(measurements['lime_flesh_fraction']) * 100:.1f}%"
        )
        st.write(
            f"**Shape fill:** {float(measurements['fill_ratio']) * 100:.1f}%  ·  "
            f"**Mean saturation:** {float(measurements['mean_saturation']):.0f}/255"
        )
        st.caption(
            "Length/width are measured after normalising to 160×160 pixels, "
            "so they describe image shape rather than real centimetres. "
            "Seed counting works best when the cut interior is visible."
        )


def show_result(image):
    result = best_prediction(image)
    probabilities = result["probabilities"]
    left, right = st.columns([1.05, 1])

    with left:
        st.image(image, caption="Selected image", use_container_width=True)

    with right:
        if result["known"]:
            st.success(f"Detected fruit: **{result['fruit']}**")
        else:
            st.warning(f"Best guess: **{result['fruit']}**")

        if result["reason"]:
            st.caption("Algorithm: " + result["reason"])

        st.metric(
            "Prediction confidence",
            f"{result['confidence'] * 100:.1f}%",
        )
        st.subheader("Top 3 predictions")

        for index in np.argsort(probabilities)[::-1][:3]:
            score = float(probabilities[int(index)])
            st.write(f"**{CLASSES[int(index)]}** — {score * 100:.1f}%")
            st.progress(float(min(max(score, 0.0), 1.0)))

        show_measurements(result["measurements"])


LIVE_HISTORY = deque(maxlen=12)
LIVE_STATE = {"last_time": 0.0, "result": None}
LIVE_LOCK = threading.Lock()


def video_frame_callback(frame):
    rgb = frame.to_ndarray(format="rgb24")[:, ::-1].copy()
    image = Image.fromarray(rgb)

    width, height = image.size
    size = int(min(width, height) * 0.62)
    left = max(0, (width - size) // 2)
    top = max(0, (height - size) // 2)
    right = min(width, left + size)
    bottom = min(height, top + size)
    roi = image.crop((left, top, right, bottom))
    now = time.monotonic()

    if (
        now - float(LIVE_STATE["last_time"]) >= 0.28
        and LIVE_LOCK.acquire(False)
    ):
        try:
            LIVE_STATE["result"] = predict_one(roi)
            LIVE_STATE["last_time"] = now
            LIVE_HISTORY.append(LIVE_STATE["result"])
        except Exception:
            pass
        finally:
            LIVE_LOCK.release()

    result = LIVE_STATE["result"]

    if result is None or len(LIVE_HISTORY) < 3:
        label = "Analyzing fruit..."
    else:
        recent = list(LIVE_HISTORY)[-8:]
        known = [item for item in recent if item["known"]]

        if len(known) >= 3:
            index = Counter(
                item["index"] for item in known
            ).most_common(1)[0][0]
            matching = [item for item in known if item["index"] == index]
            confidence = float(
                np.mean([item["confidence"] for item in matching[-5:]])
            )
            measurements = matching[-1]["measurements"]
            label = (
                f"{CLASSES[index]} — {confidence * 100:.1f}%  "
                f"L/W {float(measurements['aspect_ratio']):.2f}  "
                f"Seeds {int(measurements['seed_count'])}"
            )
        else:
            label = (
                f"Unknown / best guess {result['fruit']} — "
                f"{result['confidence'] * 100:.1f}%"
            )

    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (left, top, right, bottom),
        outline=(40, 220, 90),
        width=5,
    )
    draw.rounded_rectangle(
        (12, 12, max(360, width - 12), 68),
        radius=12,
        fill=(0, 0, 0),
    )
    draw.text((24, 31), label, fill=(255, 255, 255))

    guide_top = max(top, bottom - 38)
    draw.rectangle(
        (left, guide_top, right, bottom),
        fill=(0, 0, 0),
    )
    draw.text(
        (left + 10, guide_top + 10),
        "Place ONE fruit inside the green box",
        fill=(255, 255, 255),
    )

    return av.VideoFrame.from_ndarray(
        np.asarray(image),
        format="rgb24",
    )


st.title("🍎 Fruit Image Detection")
st.caption(
    "Live front camera + upload image with seed, shape and citrus-colour analysis."
)
st.info(
    "The detector combines the saved model with explainable rules. "
    "**Pomegranate:** many visible seed-like regions. "
    "**Apple:** pale flesh with only a few seeds. "
    "**Banana:** length is much larger than width. "
    "**Orange:** round shape + orange colour/flesh. "
    "**Lime:** round shape + saturated lime-green colour or green citrus flesh."
)

camera_tab, upload_tab = st.tabs(
    ["🎥 Live Front Camera", "🖼️ Upload Image"]
)

with camera_tab:
    st.subheader("Live Front Camera")
    st.write(
        "Press **START**, allow camera permission, and place one fruit inside "
        "the green box. For seed/flesh analysis, show the cut interior when possible."
    )
    webrtc_streamer(
        key="fruit-live-structure-v2",
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
        "Live label shows L/W (length-to-width ratio) and estimated seed-like spots."
    )

with upload_tab:
    st.subheader("Upload a Fruit Image")
    uploaded = st.file_uploader(
        "Choose JPG, JPEG, PNG or WEBP",
        type=["jpg", "jpeg", "png", "webp"],
        key="fruit-structure-upload-v2",
    )
    if uploaded is not None:
        show_result(Image.open(uploaded).convert("RGB"))

st.divider()
st.caption(
    "Rule-assisted fruit recognition. Measurements are image-relative, "
    "not physical centimetres."
)

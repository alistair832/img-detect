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
        ).reshape(-1) / 255.0,
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


def dilate(mask, steps=3):
    out = mask.astype(bool).copy()
    for _ in range(steps):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            p[1:-1, 1:-1]
            | p[:-2, 1:-1]
            | p[2:, 1:-1]
            | p[1:-1, :-2]
            | p[1:-1, 2:]
            | p[:-2, :-2]
            | p[:-2, 2:]
            | p[2:, :-2]
            | p[2:, 2:]
        )
    return out


def structural_measurements(image):
    """Explainable measurements after normalising the image to 160x160."""
    img = image.convert("RGB").resize((160, 160), Image.Resampling.BILINEAR)
    rgb = np.asarray(img, dtype=np.uint8)
    hsv = np.asarray(img.convert("HSV"), dtype=np.uint8)

    r, g, b = [rgb[:, :, i].astype(np.float32) for i in range(3)]
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    candidate = ((sat > 30) & (val > 35)) | ((val < 175) & (sat > 18))
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
        left, right = main["left"], main["right"]
        top, bottom = main["top"], main["bottom"]
    else:
        left = top = 0
        right = bottom = 159

    box_width = max(1, right - left + 1)
    box_height = max(1, bottom - top + 1)
    length_px = max(box_width, box_height)
    width_px = min(box_width, box_height)
    aspect = float(length_px / max(width_px, 1))

    roi = np.zeros((160, 160), dtype=bool)
    roi[top : bottom + 1, left : right + 1] = True
    fruit_mask = roi & candidate
    fruit_pixels = float(max(int(fruit_mask.sum()), 1))
    fill = float(min(1.0, fruit_pixels / max(float(box_width * box_height), 1.0)))

    red = fruit_mask & (r > 120) & (r > g * 1.10) & (r > b * 1.08) & (sat > 50)
    green = fruit_mask & (g > 80) & (g > r * 1.03) & (g > b * 1.08) & (sat > 35)
    pale = fruit_mask & (r > 165) & (g > 145) & (b > 95) & (sat < 115) & (val > 155)
    light_membrane = fruit_mask & (sat < 72) & (val > 145)

    orange_colour = (
        fruit_mask & (hue >= 7) & (hue <= 34) & (sat >= 80) & (val >= 85) & (r > g * 1.02)
    )
    orange_flesh = (
        fruit_mask & (hue >= 8) & (hue <= 38) & (sat >= 35) & (sat <= 210)
        & (val >= 130) & (r >= g) & (g > b * 1.10)
    )

    lime_green = (
        fruit_mask & (hue >= 42) & (hue <= 105) & (sat >= 75) & (val >= 70)
        & (g > r * 1.04) & (g > b * 1.10)
    )
    lime_flesh = (
        fruit_mask & (hue >= 35) & (hue <= 95) & (sat >= 25) & (sat <= 170)
        & (val >= 125) & (g > r * 1.015) & (g > b * 1.06)
    )

    # Guava: greener but usually softer/less saturated than Lime.
    guava_green = (
        fruit_mask & (hue >= 42) & (hue <= 115) & (sat >= 32) & (sat <= 165)
        & (val >= 75) & (g > r * 1.01) & (g > b * 1.04)
    )
    guava_soft_green = (
        fruit_mask & (hue >= 45) & (hue <= 112) & (sat >= 32) & (sat <= 140) & (val >= 90)
    )
    guava_pink_flesh = (
        fruit_mask & (((hue <= 14) | (hue >= 238))) & (sat >= 18) & (sat <= 150)
        & (val >= 120) & (r > g * 1.02) & (r > b * 1.02)
    )

    red_aril_pixels = (
        fruit_mask & (((hue < 14) | (hue > 238))) & (sat > 105) & (val > 75) & (val < 235)
    )

    pale_fraction = float(pale.sum()) / fruit_pixels
    membrane_fraction = float(light_membrane.sum()) / fruit_pixels
    red_aril_fraction = float(red_aril_pixels.sum()) / fruit_pixels
    pink_fraction = float(guava_pink_flesh.sum()) / fruit_pixels

    seed_analysis_valid = (
        pale_fraction >= 0.06
        or pink_fraction >= 0.06
        or (membrane_fraction >= 0.035 and red_aril_fraction >= 0.08)
    )

    seed_count = 0
    guava_seed_count = 0
    if seed_analysis_valid:
        flesh_region = dilate(pale | light_membrane | guava_pink_flesh, steps=4) & roi
        dark_seed = flesh_region & (val >= 22) & (val < 125) & (sat > 24)
        red_aril = flesh_region & red_aril_pixels
        seed_candidates = components(dark_seed | red_aril, 4, 190)
        for component in seed_candidates:
            cw = component["right"] - component["left"] + 1
            ch = component["bottom"] - component["top"] + 1
            comp_aspect = max(cw, ch) / max(1, min(cw, ch))
            if cw <= 24 and ch <= 24 and comp_aspect <= 3.0:
                seed_count += 1

        # Pale/tan Guava seeds.
        tan_seed = (
            flesh_region & (val >= 95) & (val <= 225) & (sat >= 8) & (sat <= 105)
            & (r >= b * 1.03) & (g >= b * 1.01)
        )
        tan_candidates = components(tan_seed, 3, 95)
        for component in tan_candidates:
            cw = component["right"] - component["left"] + 1
            ch = component["bottom"] - component["top"] + 1
            comp_aspect = max(cw, ch) / max(1, min(cw, ch))
            if cw <= 18 and ch <= 18 and comp_aspect <= 2.8:
                guava_seed_count += 1

    local = fruit_mask[top : bottom + 1, left : right + 1]
    if box_width >= box_height:
        profile = local.sum(axis=0).astype(np.float32)
    else:
        profile = local.sum(axis=1).astype(np.float32)

    n = len(profile)
    edge_n = max(1, int(n * 0.16))
    mid_start = max(0, int(n * 0.40))
    mid_end = min(n, max(mid_start + 1, int(n * 0.60)))
    edge_mean = float(np.mean(np.concatenate([profile[:edge_n], profile[-edge_n:]])))
    middle_mean = float(np.mean(profile[mid_start:mid_end]))
    taper_score = float(np.clip(1.0 - edge_mean / max(middle_mean, 1e-6), 0.0, 1.0))

    mean_saturation = float(sat[fruit_mask].mean()) if fruit_mask.any() else 0.0
    mean_value = float(val[fruit_mask].mean()) if fruit_mask.any() else 0.0

    return {
        "length_px": int(length_px),
        "width_px": int(width_px),
        "aspect_ratio": aspect,
        "fill_ratio": fill,
        "seed_count": int(seed_count),
        "guava_seed_count": int(guava_seed_count),
        "seed_analysis_valid": bool(seed_analysis_valid),
        "red_fraction": float(red.sum()) / fruit_pixels,
        "green_fraction": float(green.sum()) / fruit_pixels,
        "pale_flesh_fraction": pale_fraction,
        "orange_fraction": float(orange_colour.sum()) / fruit_pixels,
        "orange_flesh_fraction": float(orange_flesh.sum()) / fruit_pixels,
        "lime_green_fraction": float(lime_green.sum()) / fruit_pixels,
        "lime_flesh_fraction": float(lime_flesh.sum()) / fruit_pixels,
        "lime_hue_purity": float(lime_green.sum()) / fruit_pixels,
        "orange_hue_purity": float(orange_colour.sum()) / fruit_pixels,
        "guava_green_fraction": float(guava_green.sum()) / fruit_pixels,
        "guava_soft_green_fraction": float(guava_soft_green.sum()) / fruit_pixels,
        "guava_pink_flesh_fraction": pink_fraction,
        "mean_saturation": mean_saturation,
        "mean_value": mean_value,
        "taper_score": taper_score,
        "red_aril_fraction": red_aril_fraction,
    }


def apply_structure_rules(probabilities, image):
    p = probabilities.astype(np.float32).copy()
    m = structural_measurements(image)

    seeds = int(m["seed_count"])
    guava_seeds = int(m["guava_seed_count"])
    seed_valid = bool(m["seed_analysis_valid"])
    aspect = float(m["aspect_ratio"])
    fill = float(m["fill_ratio"])
    pale = float(m["pale_flesh_fraction"])
    red = float(m["red_fraction"])
    green = float(m["green_fraction"])
    orange_area = float(m["orange_fraction"])
    orange_flesh = float(m["orange_flesh_fraction"])
    lime_green = float(m["lime_green_fraction"])
    lime_flesh = float(m["lime_flesh_fraction"])
    lime_purity = float(m["lime_hue_purity"])
    orange_purity = float(m["orange_hue_purity"])
    guava_green = float(m["guava_green_fraction"])
    guava_soft_green = float(m["guava_soft_green_fraction"])
    guava_pink = float(m["guava_pink_flesh_fraction"])
    mean_sat = float(m["mean_saturation"])
    taper = float(m["taper_score"])
    red_aril_fraction = float(m["red_aril_fraction"])
    reasons = []

    if seed_valid and seeds >= 12 and red >= 0.10:
        p[POMEGRANATE] *= 5.5
        p[APPLE] *= 0.30
        p[GUAVA] *= 0.65
        reasons.append(f"many visible seed/aril regions ({seeds})")
    elif seed_valid and seeds >= 8 and red >= 0.06:
        p[POMEGRANATE] *= 3.0
        p[APPLE] *= 0.60
        reasons.append(f"several visible seed/aril regions ({seeds})")

    if (
        seed_valid and seeds <= 6 and pale >= 0.16 and (green >= 0.06 or red >= 0.06)
        and lime_flesh < 0.14 and orange_flesh < 0.16 and guava_pink < 0.10
    ):
        p[APPLE] *= 4.5
        p[POMEGRANATE] *= 0.30
        p[GUAVA] *= 0.52
        p[LIME] *= 0.55
        reasons.append(f"pale flesh with only {seeds} visible seed-like spots")

    if aspect >= 1.55:
        p[BANANA] *= 3.2
        p[LIME] *= 0.72
        p[ORANGE] *= 0.72
        p[POMEGRANATE] *= 0.72
        reasons.append(f"elongated shape L/W={aspect:.2f}")
    elif aspect >= 1.35:
        p[BANANA] *= 1.7

    if (
        aspect <= 1.45 and fill >= 0.40 and orange_purity >= 0.62
        and mean_sat >= 145 and lime_green < 0.10
    ):
        p[ORANGE] *= 24.0
        p[GUAVA] *= 0.18
        p[APPLE] *= 0.40
        p[LIME] *= 0.22
        reasons.append(f"strong orange peel signature ({orange_purity * 100:.0f}% orange pixels)")
    elif (
        aspect <= 1.45 and orange_area >= 0.24 and mean_sat >= 105
        and probabilities[ORANGE] >= 0.03
    ):
        p[ORANGE] *= 6.0
        p[GUAVA] *= 0.55
        reasons.append(f"orange-colour support ({orange_area * 100:.0f}% area)")

    if aspect <= 1.45 and orange_flesh >= 0.20 and probabilities[ORANGE] >= 0.03:
        p[ORANGE] *= 2.4
        p[LIME] *= 0.72
        reasons.append(f"orange-coloured flesh ({orange_flesh * 100:.0f}% area)")

    # Guava: cut white/pink flesh with several small pale/tan seeds.
    cut_guava_signature = (
        seed_valid and aspect <= 1.55 and (pale >= 0.14 or guava_pink >= 0.10)
        and guava_seeds >= 4 and red_aril_fraction < 0.10 and probabilities[GUAVA] >= 0.015
    )
    if cut_guava_signature:
        p[GUAVA] *= 12.0
        p[LIME] *= 0.25
        p[APPLE] *= 0.55
        p[POMEGRANATE] *= 0.45
        reasons.append(f"guava flesh + several small pale seeds ({guava_seeds})")

    # Whole Guava: softer green and less citrus-like than Lime.
    whole_guava_signature = (
        not seed_valid and aspect <= 1.48 and fill >= 0.38
        and guava_soft_green >= 0.34 and guava_green >= 0.52
        and 55 <= mean_sat <= 155 and lime_purity < 0.68
        and orange_purity < 0.08 and taper < 0.32 and probabilities[GUAVA] >= 0.015
    )
    if whole_guava_signature:
        p[GUAVA] *= 14.0
        p[LIME] *= 0.22
        p[APPLE] *= 0.72
        p[ORANGE] *= 0.70
        reasons.append(f"soft green guava signature ({guava_soft_green * 100:.0f}% moderate-green pixels)")
    elif (
        aspect <= 1.52 and guava_soft_green >= 0.24 and mean_sat < 150
        and lime_purity < 0.62 and probabilities[GUAVA] >= 0.04
    ):
        p[GUAVA] *= 3.5
        p[LIME] *= 0.65
        reasons.append(f"moderate green guava support ({guava_soft_green * 100:.0f}% area)")

    # Lime: requires more saturated/pure citrus green than Guava.
    whole_lime_signature = (
        not seed_valid and aspect <= 1.50 and fill >= 0.40 and lime_purity >= 0.70
        and guava_soft_green < 0.42 and pale < 0.08 and orange_area < 0.06 and mean_sat >= 150
    )
    if whole_lime_signature:
        p[LIME] *= 32.0
        p[GUAVA] *= 0.10
        p[APPLE] *= 0.42
        p[ORANGE] *= 0.35
        reasons.append(
            f"strong lime peel signature ({lime_purity * 100:.0f}% lime-green pixels, saturation {mean_sat:.0f}/255)"
        )
    elif (
        aspect <= 1.50 and fill >= 0.36 and lime_purity >= 0.48
        and guava_soft_green < 0.40 and pale < 0.10 and mean_sat >= 120
        and probabilities[LIME] >= 0.02
    ):
        p[LIME] *= 7.0
        p[GUAVA] *= 0.55
        p[APPLE] *= 0.72
        reasons.append(f"lime-green colour support ({lime_purity * 100:.0f}% area)")

    if (
        lime_purity >= 0.55 and mean_sat >= 130 and taper >= 0.18
        and aspect <= 1.55 and guava_soft_green < 0.40
    ):
        p[LIME] *= 1.8
        p[GUAVA] *= 0.78
        reasons.append(f"tapered citrus shape ({taper * 100:.0f}% taper)")

    if (
        aspect <= 1.45 and lime_flesh >= 0.16 and lime_green >= 0.06
        and probabilities[LIME] >= 0.03 and guava_pink < 0.08
    ):
        p[LIME] *= 2.5
        p[APPLE] *= 0.58
        p[ORANGE] *= 0.72
        reasons.append(f"green citrus flesh ({lime_flesh * 100:.0f}% area)")

    if (
        pale >= 0.08 and green >= 0.18 and lime_purity < 0.55
        and guava_soft_green < 0.38 and mean_sat < 150 and aspect < 1.40
    ):
        p[APPLE] *= 1.8
        p[GUAVA] *= 0.82
        p[LIME] *= 0.78

    p /= p.sum() + 1e-8
    return p, "; ".join(reasons), m


def image_quality(image):
    hsv = np.asarray(
        image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR).convert("HSV"),
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
        c3.metric("Length / Width", f"{float(measurements['aspect_ratio']):.2f}")
        seed_text = str(int(measurements["seed_count"])) if measurements["seed_analysis_valid"] else "N/A"
        c4.metric("Seed-like spots", seed_text)

        guava_seed_text = str(int(measurements["guava_seed_count"])) if measurements["seed_analysis_valid"] else "N/A"
        st.write(
            f"**Guava seed-like spots:** {guava_seed_text}  ·  "
            f"**Guava-green area:** {float(measurements['guava_green_fraction']) * 100:.1f}%  ·  "
            f"**Soft green area:** {float(measurements['guava_soft_green_fraction']) * 100:.1f}%  ·  "
            f"**Pink guava flesh:** {float(measurements['guava_pink_flesh_fraction']) * 100:.1f}%"
        )
        st.write(
            f"**Pale flesh:** {float(measurements['pale_flesh_fraction']) * 100:.1f}%  ·  "
            f"**Red area:** {float(measurements['red_fraction']) * 100:.1f}%  ·  "
            f"**Green area:** {float(measurements['green_fraction']) * 100:.1f}%"
        )
        st.write(
            f"**Orange area:** {float(measurements['orange_fraction']) * 100:.1f}%  ·  "
            f"**Lime-green area:** {float(measurements['lime_green_fraction']) * 100:.1f}%  ·  "
            f"**Mean saturation:** {float(measurements['mean_saturation']):.0f}/255"
        )
        st.write(
            f"**Shape fill:** {float(measurements['fill_ratio']) * 100:.1f}%  ·  "
            f"**Citrus taper:** {float(measurements['taper_score']) * 100:.1f}%"
        )
        st.caption(
            "Guava uses softer/moderate green, round/oval shape and visible flesh/seed cues. "
            "Lime requires a much stronger saturated lime-green signature. Seed counting is disabled for whole fruit."
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
        st.metric("Prediction confidence", f"{result['confidence'] * 100:.1f}%")
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

    if now - float(LIVE_STATE["last_time"]) >= 0.28 and LIVE_LOCK.acquire(False):
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
            index = Counter(item["index"] for item in known).most_common(1)[0][0]
            matching = [item for item in known if item["index"] == index]
            confidence = float(np.mean([item["confidence"] for item in matching[-5:]]))
            measurements = matching[-1]["measurements"]
            seed_text = str(int(measurements["seed_count"])) if measurements["seed_analysis_valid"] else "N/A"
            label = (
                f"{CLASSES[index]} — {confidence * 100:.1f}%  "
                f"L/W {float(measurements['aspect_ratio']):.2f}  Seeds {seed_text}"
            )
        else:
            label = f"Unknown / best guess {result['fruit']} — {result['confidence'] * 100:.1f}%"

    draw = ImageDraw.Draw(image)
    draw.rectangle((left, top, right, bottom), outline=(40, 220, 90), width=5)
    draw.rounded_rectangle((12, 12, max(360, width - 12), 68), radius=12, fill=(0, 0, 0))
    draw.text((24, 31), label, fill=(255, 255, 255))
    guide_top = max(top, bottom - 38)
    draw.rectangle((left, guide_top, right, bottom), fill=(0, 0, 0))
    draw.text((left + 10, guide_top + 10), "Place ONE fruit inside the green box", fill=(255, 255, 255))
    return av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")


st.title("🍎 Fruit Image Detection")
st.caption("Live front camera + upload image with seed, shape, Guava and citrus-colour analysis.")
st.info(
    "The detector combines the saved model with explainable rules. "
    "**Pomegranate:** visible cut seeds/arils. "
    "**Apple:** pale cut flesh with only a few seeds. "
    "**Banana:** high length/width ratio. "
    "**Guava:** softer green, round/oval shape, or pale/pink flesh with small seeds. "
    "**Orange:** saturated orange peel/flesh. "
    "**Lime:** highly saturated lime-green peel + compact/tapered citrus shape."
)

camera_tab, upload_tab = st.tabs(["🎥 Live Front Camera", "🖼️ Upload Image"])

with camera_tab:
    st.subheader("Live Front Camera")
    st.write("Press **START**, allow camera permission, and place one fruit inside the green box.")
    webrtc_streamer(
        key="fruit-live-structure-v4",
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
                {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302", "stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun.cloudflare.com:3478"]},
            ]
        },
        async_processing=True,
    )
    st.caption("Seed counts show N/A for whole fruit; cut Guava can also use pale/tan seed cues.")

with upload_tab:
    st.subheader("Upload a Fruit Image")
    uploaded = st.file_uploader(
        "Choose JPG, JPEG, PNG or WEBP",
        type=["jpg", "jpeg", "png", "webp"],
        key="fruit-structure-upload-v4",
    )
    if uploaded is not None:
        show_result(Image.open(uploaded).convert("RGB"))

st.divider()
st.caption("Rule-assisted fruit recognition. Measurements are image-relative, not physical centimetres.")
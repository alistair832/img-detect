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
    h, _ = np.histogram(channel, bins=bins, range=(0, 256))
    h = h.astype(np.float32)
    return h / (h.sum() + 1e-8)


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
        np.asarray(image.resize((8, 8), Image.Resampling.BILINEAR), dtype=np.float32).reshape(-1) / 255.0,
    ]
    return np.concatenate(parts).astype(np.float32)


def model_scores(features):
    scores = BASELINE.astype(np.float32).copy()
    for root, class_index in zip(ROOTS, TREE_CLASSES):
        node = int(root)
        while int(NODES["f"][node]) >= 0:
            f = int(NODES["f"][node])
            if float(features[f]) <= float(NODES["t"][node]):
                node += int(NODES["l"][node])
            else:
                node += int(NODES["r"][node])
        scores[int(class_index)] += float(NODES["v"][node])
    return scores


def softmax(scores, temperature=1.25):
    z = np.clip((scores - np.max(scores)) / temperature, -50, 50)
    p = np.exp(z)
    return p / (p.sum() + 1e-8)


def components(mask, min_area=1, max_area=None):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    found = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            xs, ys = [], []
            while stack:
                cy, cx = stack.pop()
                xs.append(cx); ys.append(cy)
                for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            area = len(xs)
            if area >= min_area and (max_area is None or area <= max_area):
                found.append({"area": area, "left": min(xs), "right": max(xs), "top": min(ys), "bottom": max(ys)})
    return found


def structural_measurements(image):
    img = image.convert("RGB").resize((160, 160), Image.Resampling.BILINEAR)
    rgb = np.asarray(img, dtype=np.uint8)
    hsv = np.asarray(img.convert("HSV"), dtype=np.uint8)
    r, g, b = [rgb[:, :, i].astype(np.float32) for i in range(3)]
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    # Main fruit region. Especially effective for Fruits-360 white backgrounds.
    candidate = (sat > 32) | (val < 232)
    candidate[:5, :] = candidate[-5:, :] = False
    candidate[:, :5] = candidate[:, -5:] = False
    comps = components(candidate, 80)

    if comps:
        def comp_score(c):
            center = not (c["right"] < 48 or c["left"] > 112 or c["bottom"] < 48 or c["top"] > 112)
            return c["area"] * (1.8 if center else 1.0)
        main = max(comps, key=comp_score)
        left, right, top, bottom = main["left"], main["right"], main["top"], main["bottom"]
        object_area = float(main["area"])
    else:
        left = top = 0; right = bottom = 159
        object_area = float(candidate.sum())

    bw, bh = max(1, right-left+1), max(1, bottom-top+1)
    length_px, width_px = max(bw, bh), min(bw, bh)
    aspect = float(length_px / max(width_px, 1))
    fill = float(min(1.0, object_area / max(float(bw*bh), 1.0)))

    roi = np.zeros((160, 160), dtype=bool)
    roi[top:bottom+1, left:right+1] = True
    denom = float(max(int(roi.sum()), 1))

    red = roi & (r > 120) & (r > g*1.10) & (r > b*1.08) & (sat > 50)
    green = roi & (g > 80) & (g > r*1.03) & (g > b*1.08) & (sat > 35)
    pale = roi & (r > 165) & (g > 145) & (b > 95) & (sat < 115) & (val > 155)

    # Apple seeds are usually a few dark compact spots. A cut pomegranate often
    # exposes many compact red/dark aril-like regions.
    dark_seed = roi & (val >= 30) & (val < 125) & (sat > 35)
    red_aril = roi & (((hue < 14) | (hue > 238))) & (sat > 105) & (val > 75) & (val < 235)
    seed_candidates = components(dark_seed | red_aril, 4, 190)
    seed_count = 0
    for c in seed_candidates:
        cw, ch = c["right"]-c["left"]+1, c["bottom"]-c["top"]+1
        if cw <= 24 and ch <= 24 and max(cw,ch)/max(1,min(cw,ch)) <= 3.0:
            seed_count += 1

    return {
        "length_px": int(length_px),
        "width_px": int(width_px),
        "aspect_ratio": aspect,
        "fill_ratio": fill,
        "seed_count": int(seed_count),
        "red_fraction": float(red.sum())/denom,
        "green_fraction": float(green.sum())/denom,
        "pale_flesh_fraction": float(pale.sum())/denom,
    }


def apply_structure_rules(probabilities, image):
    p = probabilities.astype(np.float32).copy()
    m = structural_measurements(image)
    seeds = int(m["seed_count"])
    aspect = float(m["aspect_ratio"])
    pale = float(m["pale_flesh_fraction"])
    red = float(m["red_fraction"])
    green = float(m["green_fraction"])
    reasons = []

    # Pomegranate: many visible seeds/arils + red interior.
    if seeds >= 12 and red >= 0.10:
        p[POMEGRANATE] *= 5.0; p[APPLE] *= 0.35; p[GUAVA] *= 0.70
        reasons.append(f"many seed-like regions ({seeds})")
    elif seeds >= 8 and red >= 0.06:
        p[POMEGRANATE] *= 2.8; p[APPLE] *= 0.65
        reasons.append(f"several seed-like regions ({seeds})")

    # Apple: pale cut flesh + only a few seeds + red/green peel.
    if seeds <= 6 and pale >= 0.16 and (green >= 0.06 or red >= 0.06):
        p[APPLE] *= 4.2; p[POMEGRANATE] *= 0.32; p[GUAVA] *= 0.55; p[LIME] *= 0.55
        reasons.append(f"pale flesh with only {seeds} seed-like spots")

    # Banana: length is clearly larger than width.
    if aspect >= 1.55:
        p[BANANA] *= 3.0; p[LIME] *= 0.70; p[ORANGE] *= 0.70; p[POMEGRANATE] *= 0.70
        reasons.append(f"elongated shape L/W={aspect:.2f}")
    elif aspect >= 1.35:
        p[BANANA] *= 1.6

    # Gentle support for whole green apple; green alone is not decisive.
    if seeds <= 6 and pale >= 0.08 and green >= 0.18 and aspect < 1.35:
        p[APPLE] *= 1.7; p[GUAVA] *= 0.82; p[LIME] *= 0.82

    p /= p.sum() + 1e-8
    return p, "; ".join(reasons), m


def image_quality(image):
    hsv = np.asarray(image.convert("RGB").resize((64,64), Image.Resampling.BILINEAR).convert("HSV"), dtype=np.float32)
    sat, val = hsv[:,:,1], hsv[:,:,2]
    colourful, mean_v, std_v = float(np.mean(sat>30)), float(np.mean(val)), float(np.std(val))
    if mean_v < 22: return False, "Image is too dark."
    if mean_v > 249 and std_v < 7: return False, "Image is almost blank."
    if colourful < 0.035 and std_v < 14: return False, "No clear fruit-like object was found."
    return True, ""


def predict_one(image):
    base = softmax(model_scores(extract_features(image)))
    probs, rule_reason, measurements = apply_structure_rules(base, image)
    order = np.argsort(probs)[::-1]
    first, second = int(order[0]), int(order[1])
    confidence = float(probs[first]); margin = confidence - float(probs[second])
    ok, quality_reason = image_quality(image)
    known = ok and confidence >= 0.40 and margin >= 0.055
    reason = quality_reason if not ok else ("Prediction confidence is low." if confidence < 0.40 else ("Top fruit classes are too similar." if margin < 0.055 else rule_reason))
    return {"index": first, "fruit": CLASSES[first], "confidence": confidence, "probabilities": probs, "known": known, "reason": reason, "measurements": measurements}


def centre_square(image, fraction=1.0):
    image = image.convert("RGB")
    w, h = image.size; side = max(1, int(min(w,h)*fraction))
    left, top = max(0,(w-side)//2), max(0,(h-side)//2)
    return image.crop((left, top, left+side, top+side))


def best_prediction(image):
    candidates = [image.convert("RGB"), centre_square(image,1.0), centre_square(image,0.84), centre_square(image,0.68)]
    results = [predict_one(c) for c in candidates]
    confident = [r for r in results if r["known"]]
    return max(confident or results, key=lambda r: r["confidence"])


def show_measurements(m):
    with st.expander("🔎 Algorithm measurements", expanded=True):
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Length", f"{int(m['length_px'])} px")
        c2.metric("Width", f"{int(m['width_px'])} px")
        c3.metric("Length / Width", f"{float(m['aspect_ratio']):.2f}")
        c4.metric("Seed-like spots", str(int(m["seed_count"])))
        st.write(
            f"**Pale flesh:** {float(m['pale_flesh_fraction'])*100:.1f}%  ·  "
            f"**Red area:** {float(m['red_fraction'])*100:.1f}%  ·  "
            f"**Green area:** {float(m['green_fraction'])*100:.1f}%  ·  "
            f"**Shape fill:** {float(m['fill_ratio'])*100:.1f}%"
        )
        st.caption("Length/width are measured after normalising to 160×160 pixels. Seed counting works best when the cut interior is visible.")


def show_result(image):
    result = best_prediction(image); probs = result["probabilities"]
    left, right = st.columns([1.05,1])
    with left:
        st.image(image, caption="Selected image", use_container_width=True)
    with right:
        if result["known"]: st.success(f"Detected fruit: **{result['fruit']}**")
        else: st.warning(f"Best guess: **{result['fruit']}**")
        if result["reason"]: st.caption("Algorithm: " + result["reason"])
        st.metric("Prediction confidence", f"{result['confidence']*100:.1f}%")
        st.subheader("Top 3 predictions")
        for i in np.argsort(probs)[::-1][:3]:
            score = float(probs[int(i)])
            st.write(f"**{CLASSES[int(i)]}** — {score*100:.1f}%")
            st.progress(float(min(max(score,0.0),1.0)))
        show_measurements(result["measurements"])


LIVE_HISTORY = deque(maxlen=12)
LIVE_STATE = {"last_time":0.0, "result":None}
LIVE_LOCK = threading.Lock()


def video_frame_callback(frame):
    rgb = frame.to_ndarray(format="rgb24")[:, ::-1].copy()
    image = Image.fromarray(rgb)
    w,h = image.size; size = int(min(w,h)*0.62)
    left,top = max(0,(w-size)//2), max(0,(h-size)//2)
    right,bottom = min(w,left+size), min(h,top+size)
    roi = image.crop((left,top,right,bottom))
    now = time.monotonic()

    if now-float(LIVE_STATE["last_time"]) >= 0.28 and LIVE_LOCK.acquire(False):
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
        known = [x for x in recent if x["known"]]
        if len(known) >= 3:
            idx = Counter(x["index"] for x in known).most_common(1)[0][0]
            matching = [x for x in known if x["index"] == idx]
            conf = float(np.mean([x["confidence"] for x in matching[-5:]]))
            m = matching[-1]["measurements"]
            label = f"{CLASSES[idx]} — {conf*100:.1f}%  L/W {float(m['aspect_ratio']):.2f}  Seeds {int(m['seed_count'])}"
        else:
            label = f"Unknown / best guess {result['fruit']} — {result['confidence']*100:.1f}%"

    draw = ImageDraw.Draw(image)
    draw.rectangle((left,top,right,bottom), outline=(40,220,90), width=5)
    draw.rounded_rectangle((12,12,max(360,w-12),68), radius=12, fill=(0,0,0))
    draw.text((24,31), label, fill=(255,255,255))
    guide_top = max(top,bottom-38)
    draw.rectangle((left,guide_top,right,bottom), fill=(0,0,0))
    draw.text((left+10,guide_top+10), "Place ONE fruit inside the green box", fill=(255,255,255))
    return av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")


st.title("🍎 Fruit Image Detection")
st.caption("Live front camera + upload image with seed-count and length/width analysis.")
st.info(
    "The detector combines the saved model with explainable rules. "
    "**Pomegranate:** many visible seed-like regions. **Apple:** pale flesh with only a few seeds. "
    "**Banana:** length is much larger than width."
)

camera_tab, upload_tab = st.tabs(["🎥 Live Front Camera", "🖼️ Upload Image"])

with camera_tab:
    st.subheader("Live Front Camera")
    st.write("Press **START**, allow camera permission, and place one fruit inside the green box. For seed analysis, show the cut interior when possible.")
    webrtc_streamer(
        key="fruit-live-structure-v1",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video":{"facingMode":"user","width":{"ideal":640},"height":{"ideal":480},"frameRate":{"ideal":20,"max":24}},"audio":False},
        rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302","stun:stun1.l.google.com:19302","stun:stun2.l.google.com:19302"]},{"urls":["stun:stun.cloudflare.com:3478"]}]},
        async_processing=True,
    )
    st.caption("Live label shows L/W (length-to-width ratio) and estimated seed-like spots.")

with upload_tab:
    st.subheader("Upload a Fruit Image")
    uploaded = st.file_uploader("Choose JPG, JPEG, PNG or WEBP", type=["jpg","jpeg","png","webp"], key="fruit-structure-upload-v1")
    if uploaded is not None:
        show_result(Image.open(uploaded).convert("RGB"))

st.divider()
st.caption("Rule-assisted fruit recognition. Measurements are image-relative, not physical centimetres.")
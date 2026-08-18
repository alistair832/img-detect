import base64
import zlib
from collections import Counter, deque

import av
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_webrtc import webrtc_streamer


st.set_page_config(
    page_title="Live Fruit Detection",
    page_icon="🍎",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Trained from the user's uploaded Fruit Quality Classification YOLO dataset.
# The original 14 quality labels were merged into 6 fruit categories:
# Apple, Banana, Guava, Lime, Orange and Pomegranate.
#
# The compact model is a Linear SVM trained on HSV colour histograms + low-
# resolution spatial RGB features. It is embedded here so Streamlit Cloud does
# not need TensorFlow, PyTorch or a separate model download.
# -----------------------------------------------------------------------------
CLASSES = ["Apple", "Banana", "Guava", "Lime", "Orange", "Pomegranate"]
FEATURE_DIM = 164
MODEL_B85 = """c-jq=X<St00>%j~!39A=L&#x<8JL;#p5;C7Szc0zRFv@M@^eFR*R-3$b;0zyYwlaAxS$Lo+id5YIcMg~FrYx)+|4pg%^RkrX`$WRE!-|&p5OC)c>W)r4^OVGgj{8gbY(I50L66|pVg&1##WvIr;RbSIr6K}dbbqb;%>BMx(`*PSgspaI_GKk(@$0mx0$?y$qCNUL_^wzUhC6N0g{Tan^JcjTq+UdTxq5&)oOFiwTv}}T;H3sj3Id<QRr&6UuPo9VicPt%KsN!LuW`Pcax#kKZux{eD2^XCRO#ZvGO!;3p3H(Lwm`1y$h8DCc)j2ZE%(mv*i>vpbbF3*Zs;3Ew|`Pm?WV`oe?wzy9wQa1FY3+Y)Q%^VuQ5RQOUOXlB|nJN?9TE+#qQ$6B{U{Wi=08^Is-Qm^n<qbJTw=xrjNa16Ed6!qe1gEzG8C<E+^9H|_WCv9a3P&o?N$IkRsBeN*m3EN~V(_5-Lb^iEBSk`jpXzIFT4m=c)aiSlRgZA=t+#FW)FXp*Z*5*XDQ;l2<o2pBBO<;N*E%Acb~^#u5`=PvLS$PN6@os52m6;wyK0o|@n0E^kn;JRZcD5DQEg<&Q%1D_EJ_?`Uw+-~j!a{#TP@($Tt4}50d`9MSH+t8v=Gu7&zrTmG1v}fp#sgVAyy#OsQ?F6adQ%||tH)E{)f^OutD?Pw8<%+*ryTn0pf&88E1H(O`IjytpkX9NhU_IzT1!A6R4{P&TC)=oH%2sZr+n|oo){#41cWdI%5qb~|=)cyO+?yyyC-GCy`*0{z5G?X#bNR|;zM7e!x56WyF4(LrlfOXy;A<)ynjK8#T>MY$rn*>Rj6RMR%^9JQ@&Nybut_cC?@6gJmv^9M<Gi{!gN?hv+;w*jlyQq~P>W<D1%RF$x<vM_(o|TLbm)!rOgSR>rn{nRAh^iWRb}O0+z-)JwT75PerYXJ^Xw<QAas;0qdFu{hcBi5Oct?*FnWHq4|6fXM0x}?!)K7DTHrFY#eRUQq}F&ySB!RBBEy4BiJmU-cw%Lu?EVZ?!9LO~%}#d%M)x)P2WhgkNa@JU*DrCyRFg%Zo`c3pn|ep;$`;A>=B2?@W{A+QdLf(>+Qv4J_|2(-SeQqjHa~=hYG++MH<4uUa(T4+wf;Kwg7Nq|N((?ToFtDk=IixBp~@mbb|sBbyU;@e&(zPkFgefl8<pt0B^6rd`}*rXv6xute?g{LMygr1%|bVJu(DYSn@3B1;1kxQ$8!BWLGw}9Qpw7cm@H_uLMSESwYYEX-V|r`YG0({uN~`~Ltd=V$O!9S=FY(^^&WSutV;YC?o(6w_3&2RHQRdXuH;s3cs<gs^y`&XV4G_&dCB#W@vMD0Il%g}Yp0v!D*0%}0NrQ?n+TKPc`C~K%D&Fi*5|(Ku0p8gp*ZOTdjRAxC&9kJ2LD#(W2o?l)rsnpw4m}jn=0j@eE)0+^$B1+za+Q@ZKmck9fJMp{uUhk0dA6xbTe9(I$fIT>JG<H5B2eQXmAKSIJ}qpgx{%+SL>B0DCW%v2X{HAz%8&@n`|l9#|yLAMKC*XNi%S5)E)IsT@xtgJ2JIsu78Pr4@)aoxZJR_b}zr#`VLbOevEcN6aH6lnm?7v*Jco9+E8X>MNd@j`j4_h&xN&?GjtY+0VnD4@FVrUe<=vVmO~XVk=@k$sHZ|3=)dOcm0q8|R!&z|P%G5;_=`kN#aJ*zfzr=(j;Dz<`3smo2~uj5kIKu`v(6mXr_u#wo!lZhgPH6p?+(i2>SrC`5k1j0!@Ql%v{nELT|;X7+7>vMOGP!`!Tof$bdmSiNq<s!r2D8h4@Xm(zME=4)Xn%PP@|8M%7r`BAF5FgDQ^+|#WG_Wm8WM0S8zE>lJYh@Qd__pT-oI_l`7j5@-upoHs8C1KU!-DrhuOQX{e_y$Jy0-&2mEdMXi=&m<%n-w8x$y&8vIYF<sb#Za7<%dH!B>x%$H24gXuE;X3^r2$<ohk2>5jO?#{s(kfaTXp>68d2;xH{edes*%Hqu=+TNX;Dyob=8|eAMauz$)xP9aU%9jaMM}Kg!Ed)%!=3bj@(JGoeS~fTw^BFI8&RwHUv(nuDqp7*IhHfGwIVpG<{wrQGm>#p0PH2U+q*LsKxlCw)>SHZk#uE_cY^m+u&uh@^@h7Y`D5Dd^ntdGavlhP<DMzx6ab_%M2YXdenj0)uRs)e*<*tv^quyh!K*^K=W|#JE(KpOcYV*@IjC9v9lfkyl<_h!iaDmnSM=mh)n?Uob={#JS(Cye_*gicz~&iYhcLTtA#Sziv-5((sGIC7|0ZT~cm`F>@N|~`UZ_Ec=dX%F{jtD2bupS#y)Wn|r(?uV3+a9Kam|6o-hXL{@<nw%H-`TtnCo@~Zwe3I*->w)Ym=gd3jg#l6kl^4<BTNbjtdo`_q1joPLHJLf?CF|m+C9!8-gvPCis=M^zglArSWXTZ%3v!IZ!bbMW-<r$id`f+fviPq_fFchGI*bxud~g`XW8f^yAxtVN&m=Uaf}g)Ozb-DA|8XA8F-n(e{h>DnbM+!Cb(CZ3IJHwZ{X(ip-9*0&BgkEAcKk2MWoF<PYE?5Xl|nYH*p1#gTY`fN>XW6Y|>i4Ph3x-~?eDrtxUOij8=IFc|m5nZjHgk6-Ysac3Ohx8WGPhhK$L@oj!Mwqaf<z;XCBVJ+^2Yx(WCRqW62#VukdekYE`llW;k5g+FV;TPg*egXbNT*$A-=frq^D{c}`a@+7@(aO)l-SBqaj_-;gegeKDe##f)^I~7V2>ZoX+)VtFIFcWTBX9%XB9@Au@f7|>+|0j^kBRwwKkO7Ud0RVeBi|~Ph*dn~6JQ1`gkLgU;7PK;`iW^rYIfRJ;1LL0IbtCFGqK!KXDv>NPZ@39WY4C0^XI6i?fd0}D9|0yM3AtzzeWepM&yE5;2F(=Ro1U<x%Qj(Y$6%-1`%W%SOX4$9^^vmIN8Y1+!HjDdyEdCA>0aNLtVH`^euZGZA2Xr!8M_wXcYGY+JxTaW}^P68)rm4&{os|JzyszjPA4FqnRik8MtDUiF$B6qS0~m26_wiLJ=qieaFV5=j^!l+`=}Y708U9qtS>*M^I`z=MUD563`|VwR;ieqgQMt+JVNma~7kI&`b0&8isbYYYut`{Sz^$29+WfMRLbbIZEVe&~DV3Q_vQ40(sDf$bk-`4^S(*(%y*S{ts~fMG*"""


@st.cache_resource
def load_embedded_model():
    raw = zlib.decompress(base64.b85decode(MODEL_B85.encode("ascii")))
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
    """Create the same 164 features used to train the compact fruit model."""
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


def predict_fruit(image: Image.Image):
    features = extract_features(image)
    scores = COEF @ features + INTERCEPT
    prediction_index = int(np.argmax(scores))

    # This softmax is only used as a relative live display score. The SVM itself
    # is not probability-calibrated, so the fruit name is the important output.
    shifted = scores - scores.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum() + 1e-8

    return prediction_index, float(probabilities[prediction_index])


prediction_history = deque(maxlen=10)
score_history = deque(maxlen=10)


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    frame_array = frame.to_ndarray(format="rgb24")

    # Mirror the front camera so it behaves like a normal selfie camera.
    frame_array = frame_array[:, ::-1].copy()
    image = Image.fromarray(frame_array)

    width, height = image.size
    box_size = int(min(width, height) * 0.62)
    left = max(0, (width - box_size) // 2)
    top = max(0, (height - box_size) // 2)
    right = min(width, left + box_size)
    bottom = min(height, top + box_size)

    roi = image.crop((left, top, right, bottom))
    prediction_index, prediction_score = predict_fruit(roi)

    prediction_history.append(prediction_index)
    score_history.append(prediction_score)

    draw = ImageDraw.Draw(image)
    draw.rectangle((left, top, right, bottom), outline=(40, 220, 90), width=5)

    if len(prediction_history) < 4:
        label = "Analyzing fruit..."
    else:
        stable_index = Counter(prediction_history).most_common(1)[0][0]
        stable_votes = prediction_history.count(stable_index)

        if stable_votes >= 5:
            label = f"Detected: {CLASSES[stable_index]}"
        else:
            label = "Hold fruit steady in the box"

    text_box = (12, 12, min(width - 12, 370), 62)
    draw.rounded_rectangle(text_box, radius=12, fill=(0, 0, 0))
    draw.text((25, 28), label, fill=(255, 255, 255))

    guide_text = "Place ONE fruit inside the green box"
    guide_box = (left, max(0, bottom - 36), right, bottom)
    draw.rectangle(guide_box, fill=(0, 0, 0))
    draw.text((left + 10, max(2, bottom - 26)), guide_text, fill=(255, 255, 255))

    return av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")


st.title("🍎 Live Fruit Detection")
st.write(
    "Turn on your front camera, hold one fruit inside the green box, and the system "
    "will continuously identify the fruit on the video."
)

info1, info2, info3 = st.columns(3)
with info1:
    st.metric("Fruit classes", "6")
with info2:
    st.metric("Training crops", "2,009")
with info3:
    st.metric("Validation accuracy", "66.7%")

with st.expander("Supported fruit classes"):
    st.write("Apple • Banana • Guava • Lime • Orange • Pomegranate")

st.subheader("Live Front Camera")
st.caption(
    "Click START, allow camera permission, then hold one fruit close to the camera "
    "inside the green square. Keep the fruit steady for about a second."
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
    "For the best result, use good lighting and let the fruit occupy most of the green box. "
    "This version identifies the fruit type; the original dataset also contains good/bad quality labels."
)

st.divider()
st.caption(
    "Artificial Intelligence project — live fruit recognition trained from the supplied fruit dataset."
)

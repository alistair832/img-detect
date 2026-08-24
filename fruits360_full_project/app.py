from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = Path(__file__).resolve()
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"

MODEL = MODELS / "best_fruit_model.keras"
CLASSES = MODELS / "class_names.json"
META = MODELS / "model_metadata.json"
CSV = OUT / "model_comparison.csv"
PNG = OUT / "model_comparison.png"

CFG = {
    "dataset_handle": "moltean/fruits",
    "image_size": 128,
    "batch_size": 32,
    "validation_split": 0.20,
    "seed": 123,
    "cnn_epochs": 8,
    "transfer_epochs": 8,
    "finetune_epochs": 4,
    "deployment_max_mb": 90.0,
}

GENERIC_RULES = [
    ("Pomegranate", ("pomegranate",)),
    ("Pineapple", ("pineapple",)),
    ("Dragon Fruit", ("dragon fruit", "pitahaya")),
    ("Passion Fruit", ("passion fruit", "maracuja")),
    ("Grapefruit", ("grapefruit",)),
    ("Watermelon", ("watermelon",)),
    ("Cantaloupe", ("cantaloupe",)),
    ("Mandarin", ("mandarin", "mandarine", "clementine")),
    ("Coconut", ("coconut", "cocos")),
    ("Blackberry", ("blackberry",)),
    ("Blueberry", ("blueberry",)),
    ("Raspberry", ("raspberry",)),
    ("Strawberry", ("strawberry",)),
    ("Apple", ("apple",)),
    ("Banana", ("banana",)),
    ("Orange", ("orange",)),
    ("Lemon", ("lemon",)),
    ("Lime", ("lime", "limetta")),
    ("Mango", ("mango",)),
    ("Papaya", ("papaya",)),
    ("Avocado", ("avocado",)),
    ("Apricot", ("apricot",)),
    ("Peach", ("peach",)),
    ("Nectarine", ("nectarine",)),
    ("Pear", ("pear",)),
    ("Plum", ("plum",)),
    ("Cherry", ("cherry",)),
    ("Grape", ("grape",)),
    ("Kiwi", ("kiwi",)),
    ("Guava", ("guava",)),
    ("Lychee", ("lychee", "litchi")),
    ("Fig", ("fig",)),
    ("Persimmon", ("persimmon", "kaki")),
    ("Carambola", ("carambola", "star fruit")),
    ("Melon", ("melon",)),
    ("Tomato", ("tomato",)),
    ("Pepper", ("pepper", "capsicum")),
    ("Cucumber", ("cucumber",)),
    ("Eggplant", ("eggplant", "aubergine")),
    ("Potato", ("potato",)),
    ("Onion", ("onion",)),
    ("Garlic", ("garlic",)),
    ("Carrot", ("carrot",)),
    ("Beetroot", ("beetroot", "beet")),
    ("Cabbage", ("cabbage",)),
    ("Cauliflower", ("cauliflower",)),
    ("Broccoli", ("broccoli",)),
    ("Corn", ("corn",)),
    ("Ginger", ("ginger",)),
    ("Hazelnut", ("hazelnut",)),
    ("Walnut", ("walnut",)),
    ("Almond", ("almond",)),
    ("Chestnut", ("chestnut",)),
]


def generic_label(name: str) -> str:
    normalized = " ".join(name.lower().replace("_", " ").replace("-", " ").split())
    for display, keys in GENERIC_RULES:
        if any(key in normalized for key in keys):
            return display
    return name


def build_groups(names: list[str]):
    order: list[str] = []
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        label = generic_label(name)
        if label not in groups:
            groups[label] = []
            order.append(label)
        groups[label].append(index)
    return order, [groups[name] for name in order]


# =============================================================================
# TRAINING WORKER
# =============================================================================


def worker_imports():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

    import kagglehub
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    return kagglehub, plt, np, pd, tf, accuracy_score, precision_recall_fscore_support


def find_split(root: Path, split_name: str) -> Path:
    candidates = [p for p in root.rglob(split_name) if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No {split_name} folder found below {root}")

    def folder_count(path: Path) -> int:
        try:
            return sum(item.is_dir() for item in path.iterdir())
        except OSError:
            return 0

    candidates.sort(
        key=lambda p: ("100x100" in str(p).lower(), folder_count(p)),
        reverse=True,
    )
    chosen = candidates[0]
    print(f"[DATA] Selected {split_name}: {chosen}", flush=True)
    print(f"[DATA] Class folders: {folder_count(chosen)}", flush=True)
    return chosen


def make_datasets(tf, train_dir: Path, test_dir: Path):
    image_size = (CFG["image_size"], CFG["image_size"])
    common = {
        "image_size": image_size,
        "batch_size": CFG["batch_size"],
        "label_mode": "int",
    }

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=CFG["validation_split"],
        subset="training",
        seed=CFG["seed"],
        shuffle=True,
        **common,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=CFG["validation_split"],
        subset="validation",
        seed=CFG["seed"],
        shuffle=False,
        **common,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        shuffle=False,
        **common,
    )

    class_names = list(train_ds.class_names)
    autotune = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(autotune),
        val_ds.prefetch(autotune),
        test_ds.prefetch(autotune),
        class_names,
        image_size,
    )


def augmentation(tf):
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.20),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
        ],
        name="augmentation",
    )


def compile_model(tf, model, learning_rate: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )


def build_custom_cnn(tf, num_classes: int, image_size: tuple[int, int]):
    inputs = tf.keras.Input(shape=image_size + (3,))
    x = augmentation(tf)(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", activation="relu"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(192, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(num_classes)(x), name="Custom_CNN")
    compile_model(tf, model, 1e-3)
    return model


def build_transfer(tf, name: str, num_classes: int, image_size: tuple[int, int]):
    if name == "MobileNetV2":
        base = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=image_size + (3,)
        )
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    else:
        base = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=image_size + (3,)
        )
        preprocess = tf.keras.applications.resnet50.preprocess_input

    base.trainable = False
    inputs = tf.keras.Input(shape=image_size + (3,))
    x = augmentation(tf)(inputs)
    x = preprocess(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(num_classes)(x), name=name)
    compile_model(tf, model, 1e-3)
    return model, base


def callbacks(tf):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=1, min_lr=1e-7
        ),
    ]


def fit_model(tf, model, train_ds, val_ds, epochs: int, quick: bool):
    kwargs = {
        "validation_data": val_ds,
        "epochs": epochs,
        "callbacks": callbacks(tf),
        "verbose": 2,
    }
    if quick:
        kwargs["steps_per_epoch"] = 30
        kwargs["validation_steps"] = 10
    model.fit(train_ds, **kwargs)


def evaluate_model(tf, accuracy_score, prf, model, test_ds, quick: bool):
    y_true, y_pred = [], []
    start = time.perf_counter()
    for batch_index, (images, labels) in enumerate(test_ds):
        logits = model(images, training=False)
        predictions = tf.argmax(logits, axis=1).numpy()
        y_true.extend(labels.numpy().astype(int).tolist())
        y_pred.extend(predictions.astype(int).tolist())
        if quick and batch_index >= 19:
            break

    elapsed = time.perf_counter() - start
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = prf(
        y_true, y_pred, average="macro", zero_division=0
    )
    return accuracy, precision, recall, f1, elapsed, len(y_true) / elapsed if elapsed else 0.0


def training_worker(quick: bool = False):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"Use Python 3.12. Current Python: {sys.version.split()[0]}"
        )

    kagglehub, plt, np, pd, tf, accuracy_score, prf = worker_imports()
    tf.keras.utils.set_random_seed(CFG["seed"])

    MODELS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    print("[1/6] Downloading / locating Fruits-360...", flush=True)
    root = Path(
        kagglehub.dataset_download(
            CFG["dataset_handle"],
            output_dir=str(DATA),
        )
    )
    train_dir = find_split(root, "Training")
    test_dir = find_split(root, "Test")
    train_ds, val_ds, test_ds, class_names, image_size = make_datasets(
        tf, train_dir, test_dir
    )
    print(f"[2/6] Detailed classes: {len(class_names)}", flush=True)

    num_classes = len(class_names)
    cnn_epochs = 1 if quick else CFG["cnn_epochs"]
    transfer_epochs = 1 if quick else CFG["transfer_epochs"]
    finetune_epochs = 1 if quick else CFG["finetune_epochs"]

    trained = {}
    paths = {
        "Custom_CNN": MODELS / "custom_cnn.keras",
        "MobileNetV2": MODELS / "mobilenetv2.keras",
        "ResNet50": MODELS / "resnet50.keras",
    }

    print("[3/6] Training Custom CNN...", flush=True)
    model = build_custom_cnn(tf, num_classes, image_size)
    fit_model(tf, model, train_ds, val_ds, cnn_epochs, quick)
    model.save(paths["Custom_CNN"])
    trained["Custom_CNN"] = model

    for name, unfreeze in (("MobileNetV2", 30), ("ResNet50", 25)):
        print(f"[3/6] Training {name}...", flush=True)
        model, base = build_transfer(tf, name, num_classes, image_size)
        fit_model(tf, model, train_ds, val_ds, transfer_epochs, quick)
        base.trainable = True
        for layer in base.layers[:-unfreeze]:
            layer.trainable = False
        compile_model(tf, model, 1e-5)
        fit_model(tf, model, train_ds, val_ds, finetune_epochs, quick)
        model.save(paths[name])
        trained[name] = model

    print("[4/6] Evaluating models...", flush=True)
    rows = []
    for name, model in trained.items():
        accuracy, precision, recall, f1, seconds, ips = evaluate_model(
            tf, accuracy_score, prf, model, test_ds, quick
        )
        rows.append(
            {
                "Model": name,
                "Accuracy": float(accuracy),
                "Macro Precision": float(precision),
                "Macro Recall": float(recall),
                "Macro F1": float(f1),
                "Inference seconds": float(seconds),
                "Images / second": float(ips),
                "Model MB": paths[name].stat().st_size / (1024 * 1024),
            }
        )

    print("[5/6] Saving comparison...", flush=True)
    results = pd.DataFrame(rows).sort_values(
        "Macro F1", ascending=False
    ).reset_index(drop=True)
    results.to_csv(CSV, index=False)

    ax = results.set_index("Model")[[
        "Accuracy", "Macro Precision", "Macro Recall", "Macro F1"
    ]].plot(kind="bar", figsize=(10, 5))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Fruits-360 Model Comparison")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PNG, dpi=160)
    plt.close()

    deployable = results[results["Model MB"] <= CFG["deployment_max_mb"]]
    selected = deployable.iloc[0] if not deployable.empty else results.iloc[0]
    best_name = str(selected["Model"])
    shutil.copy2(paths[best_name], MODEL)

    CLASSES.write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    generic_names, _ = build_groups(class_names)
    META.write_text(
        json.dumps(
            {
                "dataset_handle": CFG["dataset_handle"],
                "best_model": best_name,
                "num_classes": len(class_names),
                "num_generic_classes": len(generic_names),
                "test_accuracy": float(selected["Accuracy"]),
                "macro_f1": float(selected["Macro F1"]),
                "quick_run": bool(quick),
                "tensorflow_version": tf.__version__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[6/6] Deployment model exported.", flush=True)


if "--train-worker" in sys.argv:
    training_worker(quick="--quick" in sys.argv)
    raise SystemExit(0)


# =============================================================================
# STREAMLIT APP
# =============================================================================

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Fruits-360 AI System",
    page_icon="🍎",
    layout="wide",
)


def secret(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    try:
        return str(st.secrets.get(name, "") or "")
    except Exception:
        return ""


def load_metadata() -> dict:
    if not META.exists():
        return {}
    try:
        return json.loads(META.read_text(encoding="utf-8"))
    except Exception:
        return {}


def model_ready() -> bool:
    return MODEL.exists() and CLASSES.exists()


st.title("🍎 Fruits-360 All-in-One AI System")
st.caption("Reliable camera version — no WebRTC START button.")

meta = load_metadata()
ready = model_ready()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Python", sys.version.split()[0])
c2.metric("Model", "Ready ✅" if ready else "Not trained")
c3.metric("Classes", meta.get("num_classes", "—"))
c4.metric(
    "Accuracy",
    f"{meta['test_accuracy'] * 100:.2f}%" if "test_accuracy" in meta else "—",
)

train_tab, results_tab, detect_tab = st.tabs(
    ["1️⃣ Train Model", "2️⃣ Results", "3️⃣ Detect Fruit"]
)

with train_tab:
    st.subheader("Train Fruits-360 Models")
    token = secret("KAGGLE_API_TOKEN")
    entered = "" if token else st.text_input("Kaggle API token", type="password")

    mode = st.radio(
        "Training mode",
        ["Quick pipeline test", "Full assignment training"],
        horizontal=True,
    )

    if mode == "Quick pipeline test":
        st.warning(
            "Quick Test is only for checking the pipeline and is blocked from final recognition."
        )
    else:
        st.info(
            "Full assignment training uses the full selected Fruits-360 classification split."
        )

    if st.button(
        "🚀 Start Training",
        disabled=not bool(token or entered),
        type="primary",
        use_container_width=True,
    ):
        env = os.environ.copy()
        env["KAGGLE_API_TOKEN"] = entered or token
        env["PYTHONUNBUFFERED"] = "1"
        command = [sys.executable, str(APP), "--train-worker"]
        if mode == "Quick pipeline test":
            command.append("--quick")

        status = st.status("Training started...", expanded=True)
        log_box = st.empty()
        recent = []

        try:
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert process.stdout is not None
            for line in process.stdout:
                recent.append(line.rstrip())
                recent = recent[-80:]
                log_box.code("\n".join(recent), language="text")

            code = process.wait()
            if code == 0:
                status.update(
                    label="Training completed",
                    state="complete",
                    expanded=False,
                )
                st.cache_resource.clear()
                time.sleep(1)
                st.rerun()
            else:
                status.update(
                    label=f"Training stopped ({code})",
                    state="error",
                    expanded=True,
                )
        except Exception as exc:
            status.update(label="Could not start training", state="error")
            st.exception(exc)

with results_tab:
    st.subheader("Model Performance")
    if CSV.exists():
        import pandas as pd

        df = pd.read_csv(CSV)
        display = df.copy()
        for column in (
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
        ):
            if column in display.columns:
                display[column] = display[column].map(
                    lambda value: f"{float(value) * 100:.2f}%"
                )
        st.dataframe(display, use_container_width=True, hide_index=True)
        if PNG.exists():
            st.image(str(PNG), use_container_width=True)
    else:
        st.info("No results yet.")

with detect_tab:
    st.subheader("Fruit Recognition")

    if not model_ready():
        st.info("Train a full model first.")
    elif meta.get("quick_run", False):
        st.error(
            "Recognition is disabled because the saved model came from Quick Test. "
            "Run Full assignment training."
        )
    elif sys.version_info[:2] != (3, 12):
        st.error("The trained TensorFlow model requires Python 3.12.")
    else:
        import tensorflow as tf

        @st.cache_resource(show_spinner="Loading trained fruit model...")
        def runtime():
            model = tf.keras.models.load_model(MODEL, compile=False)
            detailed_names = json.loads(CLASSES.read_text(encoding="utf-8"))
            generic_names, generic_groups = build_groups(detailed_names)
            image_size = (
                int(model.input_shape[2]),
                int(model.input_shape[1]),
            )
            return model, detailed_names, generic_names, generic_groups, image_size

        model, detailed_names, generic_names, generic_groups, image_size = runtime()

        def predict(image: Image.Image, top_k: int = 5):
            resized = image.convert("RGB").resize(
                image_size,
                Image.Resampling.BILINEAR,
            )
            array = np.asarray(resized, dtype=np.float32)[None, ...]
            logits = model(array, training=False)[0]
            detailed_probs = tf.nn.softmax(logits, axis=-1).numpy().astype(np.float32)

            generic_probs = np.array(
                [
                    float(detailed_probs[indexes].sum())
                    for indexes in generic_groups
                ],
                dtype=np.float32,
            )
            total = float(generic_probs.sum())
            if total > 0:
                generic_probs /= total

            generic_order = np.argsort(generic_probs)[::-1][:top_k]
            detailed_order = np.argsort(detailed_probs)[::-1][:top_k]

            generic_results = [
                (
                    generic_names[int(index)],
                    float(generic_probs[int(index)]),
                )
                for index in generic_order
            ]
            detailed_results = [
                (
                    detailed_names[int(index)],
                    float(detailed_probs[int(index)]),
                )
                for index in detailed_order
            ]
            return generic_results, detailed_results

        st.success(
            f"Model ready: {len(detailed_names)} detailed classes → "
            f"{len(generic_names)} grouped output labels."
        )

        camera_tab, upload_tab = st.tabs(
            ["📸 Take Photo", "🖼️ Upload Image"]
        )

        with camera_tab:
            st.info(
                "This camera uses Streamlit's built-in camera capture. "
                "There is no WebRTC START button, so it will not auto-disconnect."
            )
            captured = st.camera_input(
                "Take a photo of ONE fruit",
                key="fruit-camera-reliable-v1",
            )

            if captured is not None:
                image = Image.open(captured).convert("RGB")
                left, right = st.columns(2)

                with left:
                    st.image(image, use_container_width=True)

                with right:
                    results, detailed = predict(image)
                    best_name, best_conf = results[0]

                    if best_conf >= 0.38:
                        st.success(f"Detected: **{best_name}**")
                    else:
                        st.warning(f"Low confidence: **{best_name}**")

                    st.metric("Confidence", f"{best_conf * 100:.2f}%")
                    st.subheader("Top predictions")
                    for name, confidence in results:
                        st.write(f"**{name}** — {confidence * 100:.2f}%")
                        st.progress(float(min(max(confidence, 0.0), 1.0)))

                    with st.expander("Detailed Fruits-360 classes"):
                        for name, confidence in detailed:
                            st.write(f"{name} — {confidence * 100:.2f}%")

        with upload_tab:
            uploaded = st.file_uploader(
                "Upload JPG, JPEG, PNG or WEBP",
                type=["jpg", "jpeg", "png", "webp"],
                key="fruit-upload-reliable-v1",
            )

            if uploaded is not None:
                image = Image.open(uploaded).convert("RGB")
                left, right = st.columns(2)

                with left:
                    st.image(image, use_container_width=True)

                with right:
                    results, detailed = predict(image)
                    best_name, best_conf = results[0]

                    if best_conf >= 0.38:
                        st.success(f"Detected: **{best_name}**")
                    else:
                        st.warning(f"Low confidence: **{best_name}**")

                    st.metric("Confidence", f"{best_conf * 100:.2f}%")
                    st.subheader("Top predictions")
                    for name, confidence in results:
                        st.write(f"**{name}** — {confidence * 100:.2f}%")
                        st.progress(float(min(max(confidence, 0.0), 1.0)))

                    with st.expander("Detailed Fruits-360 classes"):
                        for name, confidence in detailed:
                            st.write(f"{name} — {confidence * 100:.2f}%")

st.divider()
st.caption(
    "Reliable camera build: Streamlit camera_input replaces the unstable WebRTC START button."
)

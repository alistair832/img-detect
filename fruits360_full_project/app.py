from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
APP_FILE = Path(__file__).resolve()
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "best_fruit_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
COMPARISON_CSV = OUTPUT_DIR / "model_comparison.csv"
COMPARISON_PNG = OUTPUT_DIR / "model_comparison.png"

TRAINING_CONFIG = {
    "dataset_handle": "moltean/fruits",
    "image_size": 128,
    "batch_size": 32,
    "validation_split": 0.20,
    "seed": 123,
    "cnn_epochs": 8,
    "transfer_epochs": 8,
    "finetune_epochs": 4,
    "early_stopping_patience": 2,
    "deployment_max_mb": 90.0,
}

# ---------------------------------------------------------------------------
# Generic label grouping
# ---------------------------------------------------------------------------
# Fruits-360 contains many detailed varieties (for example several Apple and
# Banana classes). For the live demo we sum probabilities belonging to the same
# generic food name before choosing the final label.
GENERIC_RULES = [
    ("Pineapple", ("pineapple",)),
    ("Dragon Fruit", ("dragon fruit", "pitahaya")),
    ("Passion Fruit", ("passion fruit", "maracuja")),
    ("Pomegranate", ("pomegranate",)),
    ("Grapefruit", ("grapefruit",)),
    ("Cantaloupe", ("cantaloupe",)),
    ("Watermelon", ("watermelon",)),
    ("Mandarin", ("mandarin", "mandarine", "clementine")),
    ("Coconut", ("coconut", "cocos")),
    ("Blackberry", ("blackberry",)),
    ("Blueberry", ("blueberry",)),
    ("Raspberry", ("raspberry",)),
    ("Strawberry", ("strawberry",)),
    ("Cranberry", ("cranberry",)),
    ("Mulberry", ("mulberry",)),
    ("Gooseberry", ("gooseberry",)),
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
    ("Date", ("date",)),
    ("Persimmon", ("persimmon", "kaki")),
    ("Quince", ("quince",)),
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

def generic_label(class_name: str) -> str:
    normalized = class_name.lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    for display_name, keywords in GENERIC_RULES:
        if any(keyword in normalized for keyword in keywords):
            return display_name
    # Preserve the original label for classes without a safe generic rule.
    return class_name

def build_generic_groups(class_names: list[str]):
    names: list[str] = []
    group_to_indices: dict[str, list[int]] = {}
    for index, class_name in enumerate(class_names):
        grouped = generic_label(class_name)
        if grouped not in group_to_indices:
            group_to_indices[grouped] = []
            names.append(grouped)
        group_to_indices[grouped].append(index)
    return names, [group_to_indices[name] for name in names]


# =============================================================================
# TRAINING WORKER
# =============================================================================

def _worker_imports():
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
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    return {
        "kagglehub": kagglehub,
        "plt": plt,
        "np": np,
        "pd": pd,
        "tf": tf,
        "accuracy_score": accuracy_score,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "precision_recall_fscore_support": precision_recall_fscore_support,
    }

def _class_folder_count(path: Path) -> int:
    try:
        return sum(1 for item in path.iterdir() if item.is_dir())
    except OSError:
        return 0

def _find_split(root: Path, split_name: str) -> Path:
    candidates = [p for p in root.rglob(split_name) if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No {split_name} directory found below {root}")
    candidates.sort(
        key=lambda p: ("100x100" in str(p).lower(), _class_folder_count(p)),
        reverse=True,
    )
    chosen = candidates[0]
    print(f"[DATA] Selected {split_name}: {chosen}", flush=True)
    print(f"[DATA] Class folders: {_class_folder_count(chosen)}", flush=True)
    return chosen

def _download_dataset(lib: dict) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    handle = TRAINING_CONFIG["dataset_handle"]
    print(f"[1/6] Downloading exact Kaggle dataset: {handle}", flush=True)
    print("[DATA] Source: https://www.kaggle.com/datasets/moltean/fruits", flush=True)
    try:
        path = lib["kagglehub"].dataset_download(handle, output_dir=str(DATA_DIR))
    except Exception as exc:
        raise RuntimeError(
            "Kaggle download failed. Add KAGGLE_API_TOKEN in Streamlit Secrets. "
            f"Original error: {exc}"
        ) from exc
    root = Path(path)
    print(f"[DATA] Dataset root: {root}", flush=True)
    return root

def _make_datasets(lib: dict, train_dir: Path, test_dir: Path):
    tf = lib["tf"]
    image_size = (TRAINING_CONFIG["image_size"], TRAINING_CONFIG["image_size"])
    batch_size = TRAINING_CONFIG["batch_size"]
    seed = TRAINING_CONFIG["seed"]
    val_split = TRAINING_CONFIG["validation_split"]

    print("[2/6] Building TensorFlow datasets...", flush=True)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
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

def _augmentation(tf):
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.15),
            tf.keras.layers.RandomTranslation(0.06, 0.06),
        ],
        name="augmentation",
    )

def _compile(tf, model, learning_rate: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

def _build_custom_cnn(tf, num_classes: int, image_size: tuple[int, int]):
    inputs = tf.keras.Input(shape=image_size + (3,))
    x = _augmentation(tf)(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(192, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs, outputs, name="Custom_CNN")
    _compile(tf, model, 1e-3)
    return model

def _build_transfer(tf, name: str, num_classes: int, image_size: tuple[int, int]):
    if name == "MobileNetV2":
        base = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=image_size + (3,)
        )
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif name == "ResNet50":
        base = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=image_size + (3,)
        )
        preprocess = tf.keras.applications.resnet50.preprocess_input
    else:
        raise ValueError(name)

    base.trainable = False
    inputs = tf.keras.Input(shape=image_size + (3,))
    x = _augmentation(tf)(inputs)
    x = preprocess(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs, outputs, name=name)
    _compile(tf, model, 1e-3)
    return model, base

def _callbacks(tf):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=TRAINING_CONFIG["early_stopping_patience"],
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=1, min_lr=1e-7
        ),
    ]

def _fit_model(tf, model, train_ds, val_ds, epochs: int, quick: bool):
    kwargs = {
        "validation_data": val_ds,
        "epochs": epochs,
        "callbacks": _callbacks(tf),
        "verbose": 2,
    }
    if quick:
        kwargs["steps_per_epoch"] = 30
        kwargs["validation_steps"] = 10
    model.fit(train_ds, **kwargs)

def _train_all(lib: dict, train_ds, val_ds, class_names, image_size, quick: bool):
    tf = lib["tf"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    num_classes = len(class_names)

    cnn_epochs = 1 if quick else TRAINING_CONFIG["cnn_epochs"]
    transfer_epochs = 1 if quick else TRAINING_CONFIG["transfer_epochs"]
    finetune_epochs = 1 if quick else TRAINING_CONFIG["finetune_epochs"]

    trained = {}

    print("[3/6] Training Custom CNN...", flush=True)
    cnn = _build_custom_cnn(tf, num_classes, image_size)
    _fit_model(tf, cnn, train_ds, val_ds, cnn_epochs, quick)
    cnn_path = MODEL_DIR / "custom_cnn.keras"
    cnn.save(cnn_path)
    trained["Custom_CNN"] = cnn

    print("[3/6] Training MobileNetV2...", flush=True)
    mobile, mobile_base = _build_transfer(tf, "MobileNetV2", num_classes, image_size)
    _fit_model(tf, mobile, train_ds, val_ds, transfer_epochs, quick)
    mobile_base.trainable = True
    for layer in mobile_base.layers[:-30]:
        layer.trainable = False
    _compile(tf, mobile, 1e-5)
    _fit_model(tf, mobile, train_ds, val_ds, finetune_epochs, quick)
    mobile_path = MODEL_DIR / "mobilenetv2.keras"
    mobile.save(mobile_path)
    trained["MobileNetV2"] = mobile

    print("[3/6] Training ResNet50...", flush=True)
    resnet, resnet_base = _build_transfer(tf, "ResNet50", num_classes, image_size)
    _fit_model(tf, resnet, train_ds, val_ds, transfer_epochs, quick)
    resnet_base.trainable = True
    for layer in resnet_base.layers[:-25]:
        layer.trainable = False
    _compile(tf, resnet, 1e-5)
    _fit_model(tf, resnet, train_ds, val_ds, finetune_epochs, quick)
    resnet_path = MODEL_DIR / "resnet50.keras"
    resnet.save(resnet_path)
    trained["ResNet50"] = resnet

    return trained

def _evaluate_one(lib: dict, model, test_ds, class_names, name: str, quick: bool):
    np = lib["np"]
    pd = lib["pd"]
    tf = lib["tf"]

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
    accuracy = lib["accuracy_score"](y_true, y_pred)
    precision, recall, f1, _ = lib["precision_recall_fscore_support"](
        y_true, y_pred, average="macro", zero_division=0
    )

    labels_present = sorted(set(y_true) | set(y_pred))
    report_names = [class_names[i] for i in labels_present]
    report = pd.DataFrame(
        lib["classification_report"](
            y_true,
            y_pred,
            labels=labels_present,
            target_names=report_names,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_DIR / f"{name}_classification_report.csv")

    cm = lib["confusion_matrix"](y_true, y_pred, labels=labels_present)
    np.save(OUTPUT_DIR / f"{name}_confusion_matrix.npy", cm)

    source_path = {
        "Custom_CNN": MODEL_DIR / "custom_cnn.keras",
        "MobileNetV2": MODEL_DIR / "mobilenetv2.keras",
        "ResNet50": MODEL_DIR / "resnet50.keras",
    }[name]
    size_mb = source_path.stat().st_size / (1024 * 1024)

    return {
        "Model": name,
        "Accuracy": float(accuracy),
        "Macro Precision": float(precision),
        "Macro Recall": float(recall),
        "Macro F1": float(f1),
        "Inference seconds": float(elapsed),
        "Images / second": float(len(y_true) / elapsed if elapsed else 0.0),
        "Model MB": float(size_mb),
    }

def _save_comparison(lib: dict, rows: list[dict]):
    pd = lib["pd"]
    plt = lib["plt"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)
    results.to_csv(COMPARISON_CSV, index=False)

    metrics = results.set_index("Model")[["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]]
    ax = metrics.plot(kind="bar", figsize=(10, 5))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Fruits-360 Model Comparison")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(COMPARISON_PNG, dpi=180)
    plt.close()
    return results

def _export_best(lib: dict, results, class_names, image_size, quick: bool):
    tf = lib["tf"]
    ranked = results.sort_values("Macro F1", ascending=False).reset_index(drop=True)
    best_overall = ranked.iloc[0]

    max_mb = TRAINING_CONFIG["deployment_max_mb"]
    deployable = ranked[ranked["Model MB"] <= max_mb]
    deployment_row = deployable.iloc[0] if not deployable.empty else best_overall

    deployment_name = str(deployment_row["Model"])
    source_path = {
        "Custom_CNN": MODEL_DIR / "custom_cnn.keras",
        "MobileNetV2": MODEL_DIR / "mobilenetv2.keras",
        "ResNet50": MODEL_DIR / "resnet50.keras",
    }[deployment_name]

    shutil.copy2(source_path, MODEL_PATH)
    CLASS_NAMES_PATH.write_text(
        json.dumps(class_names, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    generic_names, generic_groups = build_generic_groups(class_names)
    metadata = {
        "dataset_handle": TRAINING_CONFIG["dataset_handle"],
        "best_overall_model": str(best_overall["Model"]),
        "deployment_model": deployment_name,
        "best_model": deployment_name,
        "image_size": list(image_size),
        "num_classes": len(class_names),
        "num_generic_classes": len(generic_names),
        "test_accuracy": float(deployment_row["Accuracy"]),
        "macro_precision": float(deployment_row["Macro Precision"]),
        "macro_recall": float(deployment_row["Macro Recall"]),
        "macro_f1": float(deployment_row["Macro F1"]),
        "model_mb": float(deployment_row["Model MB"]),
        "quick_run": bool(quick),
        "tensorflow_version": tf.__version__,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("[6/6] Deployment model exported successfully.", flush=True)
    print(json.dumps(metadata, indent=2), flush=True)

def run_training_worker(quick: bool):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Use Python 3.12. Current Python: {sys.version.split()[0]}")

    print("=" * 54, flush=True)
    print("FRUITS-360 ALL-IN-ONE TRAINING WORKER", flush=True)
    print("=" * 54, flush=True)
    print(f"Dataset: {TRAINING_CONFIG['dataset_handle']}", flush=True)
    print(f"Mode: {'QUICK TEST' if quick else 'FULL ASSIGNMENT'}", flush=True)

    lib = _worker_imports()
    tf = lib["tf"]
    tf.keras.utils.set_random_seed(TRAINING_CONFIG["seed"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    root = _download_dataset(lib)
    train_dir = _find_split(root, "Training")
    test_dir = _find_split(root, "Test")
    train_ds, val_ds, test_ds, class_names, image_size = _make_datasets(lib, train_dir, test_dir)
    print(f"[DATA] Detected detailed classes: {len(class_names)}", flush=True)

    trained = _train_all(lib, train_ds, val_ds, class_names, image_size, quick)

    print("[4/6] Evaluating models...", flush=True)
    rows = [_evaluate_one(lib, model, test_ds, class_names, name, quick) for name, model in trained.items()]

    print("[5/6] Saving comparison results...", flush=True)
    results = _save_comparison(lib, rows)
    print(results.to_string(index=False), flush=True)
    _export_best(lib, results, class_names, image_size, quick)

if "--train-worker" in sys.argv:
    run_training_worker(quick="--quick" in sys.argv)
    raise SystemExit(0)


# =============================================================================
# STREAMLIT WEBSITE
# =============================================================================

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

st.set_page_config(page_title="Fruits-360 AI System", page_icon="🍎", layout="wide")

def _secret(name: str) -> str:
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        value = st.secrets.get(name, "")
        return str(value) if value else ""
    except Exception:
        return ""

def _load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _model_ready() -> bool:
    return MODEL_PATH.exists() and CLASS_NAMES_PATH.exists()

st.title("🍎 Fruits-360 All-in-One AI System")
st.caption("One Streamlit program using Kaggle `moltean/fruits`: Train → Evaluate → Detect.")

meta = _load_metadata()
model_ready = _model_ready()

summary1, summary2, summary3, summary4 = st.columns(4)
summary1.metric("Python", sys.version.split()[0])
summary2.metric("Model status", "Ready ✅" if model_ready else "Not trained")
summary3.metric("Detailed classes", meta.get("num_classes", "—"))
summary4.metric("Test accuracy", f"{meta['test_accuracy'] * 100:.2f}%" if "test_accuracy" in meta else "—")

train_tab, results_tab, detect_tab = st.tabs(["1️⃣ Train Model", "2️⃣ Results", "3️⃣ Detect Fruit"])

with train_tab:
    st.subheader("Train from the exact Fruits-360 Kaggle dataset")
    st.code("https://www.kaggle.com/datasets/moltean/fruits", language="text")
    st.write("Full training uses the selected 100x100 Training/Test classification branch from the downloaded Fruits-360 package.")

    existing_token = _secret("KAGGLE_API_TOKEN")
    if existing_token:
        st.success("Kaggle API token detected.")
        token_input = ""
    else:
        token_input = st.text_input("Kaggle API token", type="password")

    mode = st.radio("Training mode", ["Quick pipeline test", "Full assignment training"], horizontal=True)

    if mode == "Quick pipeline test":
        st.warning("Quick mode is ONLY a pipeline test. It trains on a tiny fraction of the dataset and is blocked from camera/upload recognition.")
    else:
        st.info("Full mode trains on the full selected training split. This is the mode required before using the recognition tab.")

    disabled = not bool(existing_token or token_input)
    if st.button("🚀 Start Training", type="primary", disabled=disabled, use_container_width=True):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["KAGGLE_API_TOKEN"] = token_input or existing_token

        command = [sys.executable, str(APP_FILE), "--train-worker"]
        if mode == "Quick pipeline test":
            command.append("--quick")

        status = st.status("Training started...", expanded=True)
        log_box = st.empty()
        recent = []
        try:
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert process.stdout is not None
            for line in process.stdout:
                recent.append(line.rstrip())
                recent = recent[-100:]
                log_box.code("\n".join(recent), language="text")
            code = process.wait()

            if code == 0:
                status.update(label="Training completed", state="complete", expanded=False)
                st.cache_resource.clear()
                time.sleep(1)
                st.rerun()
            else:
                status.update(label=f"Training stopped (exit code {code})", state="error", expanded=True)
        except Exception as exc:
            status.update(label="Could not start training", state="error")
            st.exception(exc)

    if model_ready:
        st.divider()
        if meta.get("quick_run"):
            st.warning("Current saved model came from Quick Test. Run Full assignment training before recognition will be enabled.")
        else:
            st.success("A full-training model is ready for recognition.")

with results_tab:
    st.subheader("Model Performance Results")
    if COMPARISON_CSV.exists():
        import pandas as pd

        results_df = pd.read_csv(COMPARISON_CSV)
        display_df = results_df.copy()
        for col in ("Accuracy", "Macro Precision", "Macro Recall", "Macro F1"):
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda value: f"{float(value) * 100:.2f}%")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        if COMPARISON_PNG.exists():
            st.image(str(COMPARISON_PNG), caption="Model comparison", use_container_width=True)

        if meta.get("quick_run"):
            st.warning("These metrics are from Quick Test and are not final assignment results.")
    else:
        st.info("No results yet. Train a model first.")

with detect_tab:
    st.subheader("Generic Fruit Recognition")

    if not _model_ready():
        st.info("Train a full model first.")
    elif meta.get("quick_run", False):
        st.error("Recognition is disabled because the current model is a Quick Test model.")
        st.write("Go to **Train Model → Full assignment training → Start Training**. Quick Test is intentionally blocked because it is not accurate enough.")
    elif sys.version_info[:2] != (3, 12):
        st.error("The trained TensorFlow model requires Python 3.12.")
    else:
        import tensorflow as tf

        @st.cache_resource(show_spinner="Loading trained fruit model...")
        def load_runtime():
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            detailed_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
            generic_names, generic_groups = build_generic_groups(detailed_names)
            input_h = int(model.input_shape[1])
            input_w = int(model.input_shape[2])
            return model, detailed_names, generic_names, generic_groups, (input_w, input_h)

        model, detailed_names, generic_names, generic_groups, image_size = load_runtime()

        st.success(f"Full model ready: {len(detailed_names)} detailed classes are grouped into {len(generic_names)} generic output labels.")

        def predict_probabilities(image: Image.Image):
            resized = image.convert("RGB").resize(image_size, Image.Resampling.BILINEAR)
            arr = np.asarray(resized, dtype=np.float32)
            batch = np.expand_dims(arr, axis=0)
            logits = model(batch, training=False)[0]
            detailed_probs = tf.nn.softmax(logits, axis=-1).numpy().astype(np.float32)

            generic_probs = np.array(
                [float(detailed_probs[indexes].sum()) for indexes in generic_groups],
                dtype=np.float32,
            )
            total = float(generic_probs.sum())
            if total > 0:
                generic_probs /= total
            return detailed_probs, generic_probs

        def predict_image(image: Image.Image, top_k: int = 5):
            detailed_probs, generic_probs = predict_probabilities(image)

            generic_order = np.argsort(generic_probs)[::-1][:top_k]
            generic_results = [(generic_names[int(i)], float(generic_probs[int(i)])) for i in generic_order]

            detailed_order = np.argsort(detailed_probs)[::-1][:top_k]
            detailed_results = [(detailed_names[int(i)], float(detailed_probs[int(i)])) for i in detailed_order]
            return generic_results, detailed_results

        CAMERA_INTERVAL = 0.34
        SMOOTH_ALPHA = 0.50
        MIN_CONFIDENCE = 0.38
        MIN_MARGIN = 0.03

        if "camera_lock" not in st.session_state:
            st.session_state.camera_lock = threading.Lock()

        if "_camera_state_v2" not in globals():
            _camera_state_v2 = {"last_inference": 0.0, "smoothed": None, "label": "Analyzing fruit..."}

        camera_lock = st.session_state.camera_lock

        def update_camera_prediction(roi: Image.Image):
            _, generic_probs = predict_probabilities(roi)
            smoothed = _camera_state_v2["smoothed"]
            if smoothed is None or len(smoothed) != len(generic_probs):
                smoothed = generic_probs.copy()
            else:
                smoothed = SMOOTH_ALPHA * generic_probs + (1.0 - SMOOTH_ALPHA) * smoothed
            _camera_state_v2["smoothed"] = smoothed

            order = np.argsort(smoothed)[::-1]
            top_index = int(order[0])
            second_index = int(order[1]) if len(order) > 1 else top_index
            confidence = float(smoothed[top_index])
            margin = confidence - float(smoothed[second_index])

            if confidence >= MIN_CONFIDENCE and margin >= MIN_MARGIN:
                _camera_state_v2["label"] = f"{generic_names[top_index]} — {confidence * 100:.1f}%"
            else:
                _camera_state_v2["label"] = "Unknown / hold fruit steady"
            _camera_state_v2["last_inference"] = time.monotonic()

        def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            rgb = frame.to_ndarray(format="rgb24")
            rgb = rgb[:, ::-1].copy()
            image = Image.fromarray(rgb)

            width, height = image.size
            box_size = int(min(width, height) * 0.60)
            left = max(0, (width - box_size) // 2)
            top = max(0, (height - box_size) // 2)
            right = min(width, left + box_size)
            bottom = min(height, top + box_size)

            now = time.monotonic()
            if now - _camera_state_v2["last_inference"] >= CAMERA_INTERVAL:
                if camera_lock.acquire(blocking=False):
                    try:
                        if time.monotonic() - _camera_state_v2["last_inference"] >= CAMERA_INTERVAL:
                            roi = image.crop((left, top, right, bottom))
                            update_camera_prediction(roi)
                    except Exception:
                        pass
                    finally:
                        camera_lock.release()

            draw = ImageDraw.Draw(image)
            draw.rectangle((left, top, right, bottom), outline=(40, 220, 90), width=4)
            draw.rounded_rectangle((12, 12, max(330, width - 12), 64), radius=10, fill=(0, 0, 0))
            draw.text((24, 28), _camera_state_v2["label"], fill=(255, 255, 255))

            guide_top = max(top, bottom - 34)
            draw.rectangle((left, guide_top, right, bottom), fill=(0, 0, 0))
            draw.text((left + 9, guide_top + 9), "Place ONE fruit inside the green box", fill=(255, 255, 255))
            return av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")

        camera_tab, upload_tab = st.tabs(["🎥 Live Camera", "🖼️ Upload Picture"])

        with camera_tab:
            st.caption("The camera combines all detailed varieties into generic labels (for example all Apple varieties → Apple).")
            webrtc_streamer(
                key="fruits360-camera-v2",
                video_frame_callback=video_frame_callback,
                media_stream_constraints={
                    "video": {
                        "facingMode": "user",
                        "width": {"ideal": 480},
                        "height": {"ideal": 360},
                        "frameRate": {"ideal": 24, "max": 30},
                    },
                    "audio": False,
                },
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                async_processing=True,
            )

        with upload_tab:
            uploaded = st.file_uploader("Upload JPG, JPEG, PNG or WEBP", type=["jpg", "jpeg", "png", "webp"])
            if uploaded is not None:
                image = Image.open(uploaded).convert("RGB")
                left_col, right_col = st.columns(2)

                with left_col:
                    st.image(image, use_container_width=True)

                with right_col:
                    generic_results, detailed_results = predict_image(image, top_k=5)
                    best_name, best_conf = generic_results[0]

                    if best_conf >= MIN_CONFIDENCE:
                        st.success(f"Detected: **{best_name}**")
                    else:
                        st.warning(f"Low-confidence result: **{best_name}**")

                    st.metric("Grouped confidence", f"{best_conf * 100:.2f}%")
                    st.subheader("Top generic predictions")
                    for name, confidence in generic_results:
                        st.write(f"**{name}** — {confidence * 100:.2f}%")
                        st.progress(float(min(max(confidence, 0.0), 1.0)))

                    with st.expander("Show detailed Fruits-360 varieties"):
                        for name, confidence in detailed_results:
                            st.write(f"{name} — {confidence * 100:.2f}%")

st.divider()
st.caption("Fruits-360 generic recognition: detailed Kaggle classes are grouped before the final camera/upload prediction is shown.")

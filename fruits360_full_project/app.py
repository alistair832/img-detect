from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


# =============================================================================
# ONE-FILE TRAINING WORKER
# =============================================================================
# Streamlit runs this same app.py as the website. When the website starts a
# training job, it launches this exact file again with --train-worker. This keeps
# TensorFlow training isolated from the Streamlit UI while still using one Python
# program only.

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


def _worker_imports():
    """Import heavy training libraries only inside the isolated worker."""
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
        key=lambda p: (
            "100x100" in str(p).lower(),
            _class_folder_count(p),
        ),
        reverse=True,
    )
    chosen = candidates[0]
    print(f"[DATA] Selected {split_name}: {chosen}", flush=True)
    print(f"[DATA] Class folders: {_class_folder_count(chosen)}", flush=True)
    return chosen


def _download_dataset(lib: dict) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    kagglehub = lib["kagglehub"]
    handle = TRAINING_CONFIG["dataset_handle"]

    print(f"[1/6] Downloading / locating Kaggle dataset: {handle}", flush=True)
    try:
        path = kagglehub.dataset_download(handle, output_dir=str(DATA_DIR))
    except Exception as exc:
        raise RuntimeError(
            "Kaggle download failed. Add KAGGLE_API_TOKEN in Streamlit "
            f"Secrets or enter it in the Train tab. Original error: {exc}"
        ) from exc

    dataset_root = Path(path)
    print(f"[DATA] Dataset root: {dataset_root}", flush=True)
    return dataset_root


def _make_datasets(lib: dict, train_dir: Path, test_dir: Path):
    tf = lib["tf"]
    image_size = (
        int(TRAINING_CONFIG["image_size"]),
        int(TRAINING_CONFIG["image_size"]),
    )
    batch_size = int(TRAINING_CONFIG["batch_size"])
    seed = int(TRAINING_CONFIG["seed"])
    val_split = float(TRAINING_CONFIG["validation_split"])

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
            tf.keras.layers.RandomRotation(0.12),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
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
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", activation="relu"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(
        192, 3, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)

    model = tf.keras.Model(inputs, outputs, name="Custom_CNN")
    _compile(tf, model, 1e-3)
    return model


def _build_transfer(tf, name: str, num_classes: int, image_size: tuple[int, int]):
    if name == "MobileNetV2":
        base = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=image_size + (3,),
        )
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif name == "ResNet50":
        base = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=image_size + (3,),
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
            patience=int(TRAINING_CONFIG["early_stopping_patience"]),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=1,
            min_lr=1e-7,
        ),
    ]


def _fit_model(
    tf,
    model,
    train_ds,
    val_ds,
    epochs: int,
    quick: bool,
):
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

    cnn_epochs = 1 if quick else int(TRAINING_CONFIG["cnn_epochs"])
    transfer_epochs = 1 if quick else int(TRAINING_CONFIG["transfer_epochs"])
    finetune_epochs = 1 if quick else int(TRAINING_CONFIG["finetune_epochs"])

    trained = {}

    print("[3/6] Training Custom CNN...", flush=True)
    cnn = _build_custom_cnn(tf, num_classes, image_size)
    _fit_model(tf, cnn, train_ds, val_ds, cnn_epochs, quick)
    cnn_path = MODEL_DIR / "custom_cnn.keras"
    cnn.save(cnn_path)
    trained["Custom_CNN"] = cnn

    print("[3/6] Training MobileNetV2...", flush=True)
    mobile, mobile_base = _build_transfer(
        tf, "MobileNetV2", num_classes, image_size
    )
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
    resnet, resnet_base = _build_transfer(
        tf, "ResNet50", num_classes, image_size
    )
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

    y_true = []
    y_pred = []
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
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
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

    results = pd.DataFrame(rows).sort_values(
        "Macro F1", ascending=False
    ).reset_index(drop=True)
    results.to_csv(COMPARISON_CSV, index=False)

    metrics = results.set_index("Model")[[
        "Accuracy", "Macro Precision", "Macro Recall", "Macro F1"
    ]]
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

    ranked = results.sort_values(
        "Macro F1", ascending=False
    ).reset_index(drop=True)
    best_overall = ranked.iloc[0]

    max_mb = float(TRAINING_CONFIG["deployment_max_mb"])
    deployable = ranked[ranked["Model MB"] <= max_mb]
    deployment_row = (
        deployable.iloc[0] if not deployable.empty else best_overall
    )

    deployment_name = str(deployment_row["Model"])
    source_path = {
        "Custom_CNN": MODEL_DIR / "custom_cnn.keras",
        "MobileNetV2": MODEL_DIR / "mobilenetv2.keras",
        "ResNet50": MODEL_DIR / "resnet50.keras",
    }[deployment_name]

    shutil.copy2(source_path, MODEL_PATH)
    CLASS_NAMES_PATH.write_text(
        json.dumps(class_names, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metadata = {
        "best_overall_model": str(best_overall["Model"]),
        "deployment_model": deployment_name,
        "best_model": deployment_name,
        "image_size": list(image_size),
        "num_classes": len(class_names),
        "test_accuracy": float(deployment_row["Accuracy"]),
        "macro_precision": float(deployment_row["Macro Precision"]),
        "macro_recall": float(deployment_row["Macro Recall"]),
        "macro_f1": float(deployment_row["Macro F1"]),
        "model_mb": float(deployment_row["Model MB"]),
        "quick_run": bool(quick),
        "tensorflow_version": tf.__version__,
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Keep only the deployment model to reduce cloud disk usage.
    for candidate in (
        MODEL_DIR / "custom_cnn.keras",
        MODEL_DIR / "mobilenetv2.keras",
        MODEL_DIR / "resnet50.keras",
    ):
        if candidate.exists() and candidate.resolve() != source_path.resolve():
            try:
                candidate.unlink()
            except OSError:
                pass

    print("[6/6] Deployment model exported successfully.", flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


def run_training_worker(quick: bool):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "This training/deployment project requires Python 3.12. "
            f"Current Python: {sys.version.split()[0]}"
        )

    print("==================================================", flush=True)
    print("FRUITS-360 ALL-IN-ONE TRAINING WORKER", flush=True)
    print("==================================================", flush=True)
    print(f"Mode: {'QUICK TEST' if quick else 'FULL ASSIGNMENT'}", flush=True)

    lib = _worker_imports()
    tf = lib["tf"]
    tf.keras.utils.set_random_seed(int(TRAINING_CONFIG["seed"]))

    print(f"TensorFlow: {tf.__version__}", flush=True)
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}", flush=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    root = _download_dataset(lib)
    train_dir = _find_split(root, "Training")
    test_dir = _find_split(root, "Test")

    train_ds, val_ds, test_ds, class_names, image_size = _make_datasets(
        lib, train_dir, test_dir
    )
    print(f"[DATA] Detected classes: {len(class_names)}", flush=True)

    trained = _train_all(
        lib, train_ds, val_ds, class_names, image_size, quick
    )

    print("[4/6] Evaluating models...", flush=True)
    rows = []
    for name, model in trained.items():
        print(f"[EVAL] {name}", flush=True)
        rows.append(
            _evaluate_one(
                lib, model, test_ds, class_names, name, quick
            )
        )

    print("[5/6] Saving comparison results...", flush=True)
    results = _save_comparison(lib, rows)
    print(results.to_string(index=False), flush=True)

    _export_best(lib, results, class_names, image_size, quick)


# If this same file is launched as an isolated worker, do training and exit
# before importing Streamlit or WebRTC.
if "--train-worker" in sys.argv:
    run_training_worker(quick="--quick" in sys.argv)
    raise SystemExit(0)


# =============================================================================
# STREAMLIT WEBSITE
# =============================================================================

# Keep TensorFlow conservative on small Streamlit Cloud containers.
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


st.set_page_config(
    page_title="Fruits-360 AI System",
    page_icon="🍎",
    layout="wide",
)


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
st.caption(
    "One Streamlit program: Train → Evaluate → Detect. "
    "You only deploy and run this app.py."
)

meta = _load_metadata()
model_ready = _model_ready()

summary1, summary2, summary3, summary4 = st.columns(4)
summary1.metric("Python", sys.version.split()[0])
summary2.metric("Model status", "Ready ✅" if model_ready else "Not trained")
summary3.metric("Classes", meta.get("num_classes", "—"))
if "test_accuracy" in meta:
    summary4.metric("Test accuracy", f"{meta['test_accuracy'] * 100:.2f}%")
else:
    summary4.metric("Test accuracy", "—")

train_tab, results_tab, detect_tab = st.tabs(
    ["1️⃣ Train Model", "2️⃣ Results", "3️⃣ Detect Fruit"]
)


# -----------------------------------------------------------------------------
# TAB 1 — TRAIN
# -----------------------------------------------------------------------------
with train_tab:
    st.subheader("Train the Fruits-360 Models")
    st.write(
        "This page performs the complete workflow automatically: Kaggle download, "
        "preprocessing, Custom CNN, MobileNetV2, ResNet50, evaluation, model "
        "comparison, and deployment-model export."
    )

    if sys.version_info[:2] != (3, 12):
        st.error(
            f"Training requires Python 3.12. Current Python: {sys.version.split()[0]}"
        )
        st.stop()

    existing_token = _secret("KAGGLE_API_TOKEN")
    if existing_token:
        st.success("Kaggle API token detected in Streamlit Secrets.")
        token_input = ""
    else:
        st.warning(
            "Kaggle authentication is required before the dataset can be downloaded."
        )
        token_input = st.text_input(
            "Kaggle API token",
            type="password",
            help=(
                "Recommended: save KAGGLE_API_TOKEN in Streamlit "
                "Manage app → Settings → Secrets."
            ),
        )

    mode = st.radio(
        "Training mode",
        [
            "Quick pipeline test",
            "Full assignment training",
        ],
        horizontal=True,
    )

    if mode == "Quick pipeline test":
        st.info(
            "Quick mode uses 1 epoch per stage and only part of the evaluation "
            "set. Use it to confirm the pipeline works. Do not use its metrics "
            "as your final assignment results."
        )
    else:
        st.warning(
            "Full training downloads the large Fruits-360 dataset and trains "
            "three neural networks. This is computationally demanding and can "
            "take a long time on a CPU-only Streamlit Cloud instance."
        )

    start_disabled = not bool(existing_token or token_input)

    if st.button(
        "🚀 Start Training",
        type="primary",
        disabled=start_disabled,
        use_container_width=True,
    ):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if token_input:
            env["KAGGLE_API_TOKEN"] = token_input
        elif existing_token:
            env["KAGGLE_API_TOKEN"] = existing_token

        command = [
            sys.executable,
            str(APP_FILE),
            "--train-worker",
        ]
        if mode == "Quick pipeline test":
            command.append("--quick")

        st.warning(
            "Do not close or reboot the app while training is running. "
            "The training worker is isolated from the Streamlit UI."
        )

        log_box = st.empty()
        status = st.status("Training started...", expanded=True)
        recent_lines = []

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
                clean = line.rstrip()
                recent_lines.append(clean)
                recent_lines = recent_lines[-80:]
                log_box.code("\n".join(recent_lines), language="text")

            return_code = process.wait()

            if return_code == 0:
                status.update(
                    label="Training completed successfully!",
                    state="complete",
                    expanded=False,
                )
                st.success(
                    "The best model and evaluation results are ready. "
                    "Open Results or Detect Fruit."
                )
                st.cache_resource.clear()
                time.sleep(1)
                st.rerun()
            else:
                status.update(
                    label=f"Training stopped with exit code {return_code}",
                    state="error",
                    expanded=True,
                )
                st.error(
                    "Training did not complete. Check the last log lines above."
                )
        except Exception as exc:
            status.update(
                label="Could not start training",
                state="error",
                expanded=True,
            )
            st.exception(exc)

    if model_ready:
        st.divider()
        st.success("A trained deployment model is already available.")
        st.json(meta)


# -----------------------------------------------------------------------------
# TAB 2 — RESULTS
# -----------------------------------------------------------------------------
with results_tab:
    st.subheader("Model Performance Results")

    if COMPARISON_CSV.exists():
        import pandas as pd

        results_df = pd.read_csv(COMPARISON_CSV)
        display_df = results_df.copy()

        percentage_cols = [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
        ]
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(
                    lambda value: f"{float(value) * 100:.2f}%"
                )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        if COMPARISON_PNG.exists():
            st.image(
                str(COMPARISON_PNG),
                caption="Accuracy / Precision / Recall / F1 Comparison",
                use_container_width=True,
            )

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "Download model comparison CSV",
                data=COMPARISON_CSV.read_bytes(),
                file_name="model_comparison.csv",
                mime="text/csv",
            )
        with col_b:
            if COMPARISON_PNG.exists():
                st.download_button(
                    "Download comparison chart",
                    data=COMPARISON_PNG.read_bytes(),
                    file_name="model_comparison.png",
                    mime="image/png",
                )

        if meta:
            st.subheader("Selected Deployment Model")
            st.json(meta)

        if meta.get("quick_run"):
            st.warning(
                "These results came from QUICK TEST mode. Run Full assignment "
                "training before using the values in the final report."
            )
    else:
        st.info(
            "No evaluation results yet. Open **Train Model** and run the training workflow."
        )


# -----------------------------------------------------------------------------
# TAB 3 — DETECTION
# -----------------------------------------------------------------------------
with detect_tab:
    st.subheader("Live Camera and Image Upload")

    if not _model_ready():
        st.info(
            "The prediction interface will unlock after training creates "
            "`best_fruit_model.keras` and `class_names.json`."
        )
    elif sys.version_info[:2] != (3, 12):
        st.error("The trained TensorFlow model requires Python 3.12.")
    else:
        import tensorflow as tf

        @st.cache_resource(show_spinner="Loading trained fruit model...")
        def load_runtime():
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            class_names = json.loads(
                CLASS_NAMES_PATH.read_text(encoding="utf-8")
            )
            input_h = int(model.input_shape[1])
            input_w = int(model.input_shape[2])
            return model, class_names, (input_w, input_h)

        model, class_names, image_size = load_runtime()

        def predict_probabilities(image: Image.Image) -> np.ndarray:
            resized = image.convert("RGB").resize(
                image_size, Image.Resampling.BILINEAR
            )
            arr = np.asarray(resized, dtype=np.float32)
            batch = np.expand_dims(arr, axis=0)
            logits = model(batch, training=False)[0]
            probs = tf.nn.softmax(logits, axis=-1).numpy()
            return np.asarray(probs, dtype=np.float32)

        def predict_image(image: Image.Image, top_k: int = 5):
            probabilities = predict_probabilities(image)
            order = np.argsort(probabilities)[::-1][:top_k]
            return [
                (
                    class_names[int(index)],
                    float(probabilities[int(index)]),
                )
                for index in order
            ]

        # Smooth camera inference state.
        CAMERA_INFERENCE_INTERVAL = 0.32
        SMOOTH_ALPHA = 0.45
        MIN_DISPLAY_CONFIDENCE = 0.42
        MIN_DISPLAY_MARGIN = 0.04

        if "camera_lock" not in st.session_state:
            st.session_state.camera_lock = threading.Lock()

        # Module-level state is needed by the WebRTC callback.
        if "_camera_state" not in globals():
            _camera_state = {
                "last_inference_time": 0.0,
                "smoothed_probabilities": None,
                "last_label": "Analyzing fruit...",
            }

        camera_lock = st.session_state.camera_lock

        def update_camera_prediction(roi: Image.Image):
            probabilities = predict_probabilities(roi)
            smoothed = _camera_state["smoothed_probabilities"]

            if smoothed is None:
                smoothed = probabilities.copy()
            else:
                smoothed = (
                    SMOOTH_ALPHA * probabilities
                    + (1.0 - SMOOTH_ALPHA) * smoothed
                )

            _camera_state["smoothed_probabilities"] = smoothed

            order = np.argsort(smoothed)[::-1]
            top_index = int(order[0])
            second_index = int(order[1]) if len(order) > 1 else top_index

            confidence = float(smoothed[top_index])
            second_confidence = float(smoothed[second_index])
            margin = confidence - second_confidence

            if (
                confidence >= MIN_DISPLAY_CONFIDENCE
                and margin >= MIN_DISPLAY_MARGIN
            ):
                _camera_state["last_label"] = (
                    f"{class_names[top_index]} — "
                    f"{confidence * 100:.1f}% confidence"
                )
            else:
                _camera_state["last_label"] = "Unknown / hold fruit steady"

            _camera_state["last_inference_time"] = time.monotonic()

        def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            rgb = frame.to_ndarray(format="rgb24")
            rgb = rgb[:, ::-1].copy()
            image = Image.fromarray(rgb)

            width, height = image.size
            box_size = int(min(width, height) * 0.64)
            left = max(0, (width - box_size) // 2)
            top = max(0, (height - box_size) // 2)
            right = min(width, left + box_size)
            bottom = min(height, top + box_size)

            now = time.monotonic()
            if (
                now - _camera_state["last_inference_time"]
                >= CAMERA_INFERENCE_INTERVAL
            ):
                if camera_lock.acquire(blocking=False):
                    try:
                        if (
                            time.monotonic()
                            - _camera_state["last_inference_time"]
                            >= CAMERA_INFERENCE_INTERVAL
                        ):
                            roi = image.crop((left, top, right, bottom))
                            update_camera_prediction(roi)
                    except Exception:
                        pass
                    finally:
                        camera_lock.release()

            draw = ImageDraw.Draw(image)
            draw.rectangle(
                (left, top, right, bottom),
                outline=(40, 220, 90),
                width=4,
            )
            draw.rounded_rectangle(
                (12, 12, max(330, width - 12), 64),
                radius=10,
                fill=(0, 0, 0),
            )
            draw.text(
                (24, 28),
                _camera_state["last_label"],
                fill=(255, 255, 255),
            )

            guide_top = max(top, bottom - 34)
            draw.rectangle(
                (left, guide_top, right, bottom),
                fill=(0, 0, 0),
            )
            draw.text(
                (left + 9, guide_top + 9),
                "Place one fruit inside the green box",
                fill=(255, 255, 255),
            )

            return av.VideoFrame.from_ndarray(
                np.asarray(image),
                format="rgb24",
            )

        camera_mode, upload_mode = st.tabs(
            ["🎥 Live Front Camera", "🖼️ Upload Picture"]
        )

        with camera_mode:
            st.caption(
                "The video targets ~24 FPS while model inference is throttled "
                "to keep the camera smooth."
            )
            webrtc_streamer(
                key="fruits360-camera",
                video_frame_callback=video_frame_callback,
                media_stream_constraints={
                    "video": {
                        "facingMode": "user",
                        "width": {"ideal": 480},
                        "height": {"ideal": 360},
                        "frameRate": {"ideal": 24, "max": 24},
                    },
                    "audio": False,
                },
                rtc_configuration={
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]}
                    ]
                },
                async_processing=True,
            )

        with upload_mode:
            uploaded = st.file_uploader(
                "Upload JPG, JPEG, PNG or WEBP",
                type=["jpg", "jpeg", "png", "webp"],
                key="fruit-upload",
            )

            if uploaded is not None:
                image = Image.open(uploaded).convert("RGB")
                image_col, result_col = st.columns(2)

                with image_col:
                    st.image(image, use_container_width=True)

                with result_col:
                    results = predict_image(image, top_k=5)
                    best_name, best_conf = results[0]

                    if best_conf >= 0.50:
                        st.success(f"Detected: **{best_name}**")
                    else:
                        st.warning(
                            f"Low-confidence result: **{best_name}**"
                        )

                    st.metric(
                        "Prediction confidence",
                        f"{best_conf * 100:.2f}%",
                    )

                    st.subheader("Top 5 predictions")
                    for name, confidence in results:
                        st.write(
                            f"**{name}** — {confidence * 100:.2f}%"
                        )
                        st.progress(
                            float(min(max(confidence, 0.0), 1.0))
                        )


st.divider()
st.caption(
    "All-in-one assignment application — Kaggle Fruits-360 → Training → "
    "Evaluation → Streamlit Camera/Upload Recognition"
)

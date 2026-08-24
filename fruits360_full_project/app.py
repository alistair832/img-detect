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
    # Slightly emphasize Apple examples without forcing green fruits to Apple.
    "apple_train_weight": 1.30,
}

# Fruits-360 contains many varieties of the same everyday fruit.  The new
# training pipeline remaps those detailed folders to generic labels BEFORE
# training, so all Apple varieties teach one APPLE output neuron.
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
    print(f"[DATA] Detailed class folders: {folder_count(chosen)}", flush=True)
    return chosen


def make_datasets(tf, train_dir: Path, test_dir: Path):
    """Create Fruits-360 datasets and remap detailed labels to generic labels."""
    image_size = (CFG["image_size"], CFG["image_size"])
    common = {
        "image_size": image_size,
        "batch_size": CFG["batch_size"],
        "label_mode": "int",
    }

    raw_train = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=CFG["validation_split"],
        subset="training",
        seed=CFG["seed"],
        shuffle=True,
        **common,
    )
    raw_val = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=CFG["validation_split"],
        subset="validation",
        seed=CFG["seed"],
        shuffle=False,
        **common,
    )
    raw_test = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        shuffle=False,
        **common,
    )

    detailed_names = list(raw_train.class_names)
    generic_names, generic_groups = build_groups(detailed_names)

    detailed_to_generic = [0] * len(detailed_names)
    for generic_index, detailed_indexes in enumerate(generic_groups):
        for detailed_index in detailed_indexes:
            detailed_to_generic[detailed_index] = generic_index

    mapping = tf.constant(detailed_to_generic, dtype=tf.int64)
    apple_index = generic_names.index("Apple") if "Apple" in generic_names else -1

    def remap_pair(images, labels):
        return images, tf.gather(mapping, tf.cast(labels, tf.int32))

    def remap_train(images, labels):
        generic_labels = tf.gather(mapping, tf.cast(labels, tf.int32))
        if apple_index >= 0:
            weights = tf.where(
                tf.equal(generic_labels, tf.cast(apple_index, tf.int64)),
                tf.cast(CFG["apple_train_weight"], tf.float32),
                tf.constant(1.0, tf.float32),
            )
            return images, generic_labels, weights
        return images, generic_labels

    autotune = tf.data.AUTOTUNE
    train_ds = raw_train.map(remap_train, num_parallel_calls=autotune).prefetch(autotune)
    val_ds = raw_val.map(remap_pair, num_parallel_calls=autotune).prefetch(autotune)
    test_ds = raw_test.map(remap_pair, num_parallel_calls=autotune).prefetch(autotune)

    return (
        train_ds,
        val_ds,
        test_ds,
        detailed_names,
        generic_names,
        image_size,
        apple_index,
    )


def augmentation(tf):
    """Augmentation is intentionally stronger to resemble phone/webcam photos."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.18),
            tf.keras.layers.RandomZoom(0.18),
            tf.keras.layers.RandomContrast(0.25),
            tf.keras.layers.RandomBrightness(0.18, value_range=(0.0, 255.0)),
            tf.keras.layers.RandomTranslation(0.10, 0.10),
            tf.keras.layers.GaussianNoise(4.0),
        ],
        name="camera_style_augmentation",
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
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(192, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    model = tf.keras.Model(
        inputs,
        tf.keras.layers.Dense(num_classes)(x),
        name="Custom_CNN_Generic",
    )
    compile_model(tf, model, 1e-3)
    return model


def build_transfer(tf, name: str, num_classes: int, image_size: tuple[int, int]):
    if name == "MobileNetV2":
        base = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=image_size + (3,),
        )
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    else:
        base = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=image_size + (3,),
        )
        preprocess = tf.keras.applications.resnet50.preprocess_input

    base.trainable = False
    inputs = tf.keras.Input(shape=image_size + (3,))
    x = augmentation(tf)(inputs)
    x = preprocess(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    model = tf.keras.Model(
        inputs,
        tf.keras.layers.Dense(num_classes)(x),
        name=f"{name}_Generic",
    )
    compile_model(tf, model, 1e-3)
    return model, base


def callbacks(tf):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=1,
            min_lr=1e-7,
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


def evaluate_model(
    tf,
    accuracy_score,
    prf,
    model,
    test_ds,
    apple_index: int,
    quick: bool,
):
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
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    apple_precision = apple_recall = apple_f1 = 0.0
    if apple_index >= 0:
        apple_true = [1 if value == apple_index else 0 for value in y_true]
        apple_pred = [1 if value == apple_index else 0 for value in y_pred]
        apple_precision, apple_recall, apple_f1, _ = prf(
            apple_true,
            apple_pred,
            average="binary",
            zero_division=0,
        )

    ips = len(y_true) / elapsed if elapsed else 0.0
    return (
        accuracy,
        precision,
        recall,
        f1,
        apple_precision,
        apple_recall,
        apple_f1,
        elapsed,
        ips,
    )


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

    (
        train_ds,
        val_ds,
        test_ds,
        detailed_names,
        generic_names,
        image_size,
        apple_index,
    ) = make_datasets(tf, train_dir, test_dir)

    print(
        f"[2/6] {len(detailed_names)} detailed classes remapped to "
        f"{len(generic_names)} generic classes BEFORE training.",
        flush=True,
    )
    if apple_index >= 0:
        print(
            f"[APPLE] Apple is generic class {apple_index}; "
            f"training weight={CFG['apple_train_weight']:.2f}.",
            flush=True,
        )

    num_classes = len(generic_names)
    cnn_epochs = 1 if quick else CFG["cnn_epochs"]
    transfer_epochs = 1 if quick else CFG["transfer_epochs"]
    finetune_epochs = 1 if quick else CFG["finetune_epochs"]

    trained = {}
    paths = {
        "Custom_CNN": MODELS / "custom_cnn.keras",
        "MobileNetV2": MODELS / "mobilenetv2.keras",
        "ResNet50": MODELS / "resnet50.keras",
    }

    print("[3/6] Training Custom CNN on generic labels...", flush=True)
    model = build_custom_cnn(tf, num_classes, image_size)
    fit_model(tf, model, train_ds, val_ds, cnn_epochs, quick)
    model.save(paths["Custom_CNN"])
    trained["Custom_CNN"] = model

    for name, unfreeze in (("MobileNetV2", 35), ("ResNet50", 30)):
        print(f"[3/6] Training {name} on generic labels...", flush=True)
        model, base = build_transfer(tf, name, num_classes, image_size)
        fit_model(tf, model, train_ds, val_ds, transfer_epochs, quick)

        base.trainable = True
        for layer in base.layers[:-unfreeze]:
            layer.trainable = False
        compile_model(tf, model, 1e-5)
        fit_model(tf, model, train_ds, val_ds, finetune_epochs, quick)

        model.save(paths[name])
        trained[name] = model

    print("[4/6] Evaluating generic models and Apple performance...", flush=True)
    rows = []
    for name, model in trained.items():
        (
            accuracy,
            precision,
            recall,
            f1,
            apple_precision,
            apple_recall,
            apple_f1,
            seconds,
            ips,
        ) = evaluate_model(
            tf,
            accuracy_score,
            prf,
            model,
            test_ds,
            apple_index,
            quick,
        )

        # Keep overall quality dominant while giving Apple performance a role
        # in deployment-model selection.
        selection_score = 0.75 * float(f1) + 0.25 * float(apple_f1)

        rows.append(
            {
                "Model": name,
                "Accuracy": float(accuracy),
                "Macro Precision": float(precision),
                "Macro Recall": float(recall),
                "Macro F1": float(f1),
                "Apple Precision": float(apple_precision),
                "Apple Recall": float(apple_recall),
                "Apple F1": float(apple_f1),
                "Selection Score": selection_score,
                "Inference seconds": float(seconds),
                "Images / second": float(ips),
                "Model MB": paths[name].stat().st_size / (1024 * 1024),
            }
        )

    print("[5/6] Saving comparison...", flush=True)
    results = pd.DataFrame(rows).sort_values(
        ["Selection Score", "Macro F1"],
        ascending=False,
    ).reset_index(drop=True)
    results.to_csv(CSV, index=False)

    chart_cols = [
        "Accuracy",
        "Macro F1",
        "Apple Precision",
        "Apple Recall",
        "Apple F1",
    ]
    ax = results.set_index("Model")[chart_cols].plot(
        kind="bar",
        figsize=(11, 5),
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Fruits-360 Generic Model + Apple Performance")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PNG, dpi=160)
    plt.close()

    deployable = results[results["Model MB"] <= CFG["deployment_max_mb"]]
    selected = deployable.iloc[0] if not deployable.empty else results.iloc[0]
    best_name = str(selected["Model"])
    shutil.copy2(paths[best_name], MODEL)

    # New models output generic classes directly.
    CLASSES.write_text(
        json.dumps(generic_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    META.write_text(
        json.dumps(
            {
                "dataset_handle": CFG["dataset_handle"],
                "best_model": best_name,
                "label_mode": "generic",
                "num_detailed_source_classes": len(detailed_names),
                "num_classes": len(generic_names),
                "test_accuracy": float(selected["Accuracy"]),
                "macro_f1": float(selected["Macro F1"]),
                "apple_precision": float(selected["Apple Precision"]),
                "apple_recall": float(selected["Apple Recall"]),
                "apple_f1": float(selected["Apple F1"]),
                "apple_train_weight": float(CFG["apple_train_weight"]),
                "quick_run": bool(quick),
                "tensorflow_version": tf.__version__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[6/6] Exported {best_name}; Apple F1="
        f"{float(selected['Apple F1']) * 100:.2f}%.",
        flush=True,
    )


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


def square_crop(image: Image.Image) -> Image.Image:
    """Remove landscape stretching while keeping the centered fruit."""
    image = image.convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = max(0, (width - side) // 2)
    top = max(0, (height - side) // 2)
    return image.crop((left, top, left + side, top + side))


st.title("🍎 Fruits-360 All-in-One AI System")
st.caption(
    "Apple-focused generic training: varieties are merged before training, "
    "not only after prediction."
)

meta = load_metadata()
ready = model_ready()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Python", sys.version.split()[0])
c2.metric("Model", "Ready ✅" if ready else "Not trained")
c3.metric("Generic classes", meta.get("num_classes", "—"))
c4.metric(
    "Overall accuracy",
    f"{meta['test_accuracy'] * 100:.2f}%" if "test_accuracy" in meta else "—",
)
c5.metric(
    "Apple F1",
    f"{meta['apple_f1'] * 100:.2f}%" if "apple_f1" in meta else "—",
)

train_tab, results_tab, detect_tab = st.tabs(
    ["1️⃣ Train Model", "2️⃣ Results", "3️⃣ Detect Fruit"]
)

with train_tab:
    st.subheader("Train Generic Fruits-360 Models")
    st.success(
        "New training method: all Apple varieties are mapped to one Apple label "
        "before the neural network is trained."
    )
    st.caption(
        "Apple examples receive a small 1.30× training weight. This improves "
        "Apple learning without blindly changing every green fruit into Apple."
    )

    token = secret("KAGGLE_API_TOKEN")
    entered = "" if token else st.text_input("Kaggle API token", type="password")

    mode = st.radio(
        "Training mode",
        ["Quick pipeline test", "Full assignment training"],
        horizontal=True,
    )

    if mode == "Quick pipeline test":
        st.warning(
            "Quick Test checks code only. It is blocked from final recognition."
        )
    else:
        st.info(
            "Full training is required to obtain the improved Apple model. "
            "An old saved model will not gain the new accuracy automatically."
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
            "Apple Precision",
            "Apple Recall",
            "Apple F1",
            "Selection Score",
        ):
            if column in display.columns:
                display[column] = display[column].map(
                    lambda value: f"{float(value) * 100:.2f}%"
                )
        st.dataframe(display, use_container_width=True, hide_index=True)
        if PNG.exists():
            st.image(str(PNG), use_container_width=True)

        if "Apple F1" in df.columns:
            best_apple = df.sort_values("Apple F1", ascending=False).iloc[0]
            st.success(
                f"Best Apple model in this run: {best_apple['Model']} — "
                f"Apple F1 {float(best_apple['Apple F1']) * 100:.2f}%"
            )
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
            saved_names = json.loads(CLASSES.read_text(encoding="utf-8"))
            image_size = (
                int(model.input_shape[2]),
                int(model.input_shape[1]),
            )

            if meta.get("label_mode") == "generic":
                return model, saved_names, None, image_size, "generic"

            # Compatibility with an older detailed-class model already present
            # in a running Streamlit instance.
            generic_names, generic_groups = build_groups(saved_names)
            return (
                model,
                generic_names,
                generic_groups,
                image_size,
                "legacy-detailed",
            )

        model, output_names, legacy_groups, image_size, label_mode = runtime()

        def predict(image: Image.Image, top_k: int = 5):
            prepared = square_crop(image)
            resized = prepared.resize(
                image_size,
                Image.Resampling.BILINEAR,
            )
            array = np.asarray(resized, dtype=np.float32)[None, ...]
            logits = model(array, training=False)[0]
            raw_probs = tf.nn.softmax(logits, axis=-1).numpy().astype(np.float32)

            if label_mode == "generic":
                probs = raw_probs
            else:
                probs = np.array(
                    [
                        float(raw_probs[indexes].sum())
                        for indexes in legacy_groups
                    ],
                    dtype=np.float32,
                )
                total = float(probs.sum())
                if total > 0:
                    probs /= total

            order = np.argsort(probs)[::-1][:top_k]
            return [
                (output_names[int(index)], float(probs[int(index)]))
                for index in order
            ], prepared

        if meta.get("label_mode") != "generic":
            st.warning(
                "This running app still has an older detailed-class model. "
                "Run Full assignment training once to activate the new "
                "Apple-focused generic model."
            )
        else:
            st.success(
                "New generic model active. Apple varieties were merged before training."
            )

        camera_tab, upload_tab = st.tabs(
            ["📸 Take Photo", "🖼️ Upload Image"]
        )

        def show_result(image: Image.Image):
            results, prepared = predict(image)
            best_name, best_conf = results[0]

            left, right = st.columns(2)
            with left:
                st.image(
                    prepared,
                    caption="Square crop used by the model",
                    use_container_width=True,
                )
            with right:
                if best_conf >= 0.42:
                    st.success(f"Detected: **{best_name}**")
                else:
                    st.warning(f"Low confidence: **{best_name}**")

                st.metric("Confidence", f"{best_conf * 100:.2f}%")
                st.subheader("Top predictions")
                for name, confidence in results:
                    st.write(f"**{name}** — {confidence * 100:.2f}%")
                    st.progress(float(min(max(confidence, 0.0), 1.0)))

        with camera_tab:
            st.info(
                "For best Apple accuracy: put one apple near the center, "
                "fill most of the frame, and use a plain/light background."
            )
            captured = st.camera_input(
                "Take a photo of ONE fruit",
                key="fruit-camera-apple-v2",
            )
            if captured is not None:
                show_result(Image.open(captured).convert("RGB"))

        with upload_tab:
            uploaded = st.file_uploader(
                "Upload JPG, JPEG, PNG or WEBP",
                type=["jpg", "jpeg", "png", "webp"],
                key="fruit-upload-apple-v2",
            )
            if uploaded is not None:
                show_result(Image.open(uploaded).convert("RGB"))

st.divider()
st.caption(
    "Apple accuracy build: generic-label training + Apple-weighted learning + "
    "camera-style augmentation + square-crop inference."
)

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import kagglehub
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

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "training_config.json"
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Fruits-360, train three classifiers, evaluate them, and export the Streamlit model."
    )
    parser.add_argument("--quick", action="store_true", help="Use 1/1/1 epochs for a pipeline test.")
    parser.add_argument("--force-download", action="store_true", help="Force KaggleHub to redownload the dataset.")
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--skip-mobilenet", action="store_true")
    parser.add_argument("--skip-resnet", action="store_true")
    return parser.parse_args()


def class_folder_count(path: Path) -> int:
    try:
        return sum(1 for item in path.iterdir() if item.is_dir())
    except OSError:
        return 0


def find_split(root: Path, split_name: str) -> Path:
    candidates = [p for p in root.rglob(split_name) if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No {split_name} directory found below {root}")

    candidates.sort(
        key=lambda p: (
            "100x100" in str(p).lower(),
            class_folder_count(p),
        ),
        reverse=True,
    )
    print(f"\nSelected {split_name}: {candidates[0]}")
    print(f"Class folders: {class_folder_count(candidates[0])}")
    return candidates[0]


def download_dataset(config: dict, force_download: bool) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading Kaggle dataset: {config['dataset_handle']}")
    try:
        return Path(
            kagglehub.dataset_download(
                config["dataset_handle"],
                output_dir=str(DATA_DIR),
                force_download=force_download,
            )
        )
    except Exception as exc:
        print(f"Kaggle download needs authentication: {exc}")
        print("Opening KaggleHub login...")
        kagglehub.login()
        return Path(
            kagglehub.dataset_download(
                config["dataset_handle"],
                output_dir=str(DATA_DIR),
                force_download=force_download,
            )
        )


def make_datasets(train_dir: Path, test_dir: Path, config: dict):
    image_size = (int(config["image_size"]), int(config["image_size"]))
    batch_size = int(config["batch_size"])
    seed = int(config["seed"])
    val_split = float(config["validation_split"])

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


def augmentation_layer() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.12),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )


def build_custom_cnn(num_classes: int, image_size: tuple[int, int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=image_size + (3,))
    x = augmentation_layer()(inputs)
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
    compile_model(model, 1e-3)
    return model


def build_transfer_model(name: str, num_classes: int, image_size: tuple[int, int]):
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
    x = augmentation_layer()(inputs)
    x = preprocess(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)

    model = tf.keras.Model(inputs, outputs, name=name)
    compile_model(model, 1e-3)
    return model, base


def callbacks(config: dict):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(config["early_stopping_patience"]),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=1,
            min_lr=1e-7,
        ),
    ]


def train_models(train_ds, val_ds, class_names, image_size, config, args):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    num_classes = len(class_names)
    cnn_epochs = 1 if args.quick else int(config["cnn_epochs"])
    transfer_epochs = 1 if args.quick else int(config["transfer_epochs"])
    finetune_epochs = 1 if args.quick else int(config["finetune_epochs"])

    trained: dict[str, tf.keras.Model] = {}

    if not args.skip_cnn:
        print("\n=== Training Custom CNN ===")
        model = build_custom_cnn(num_classes, image_size)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cnn_epochs,
            callbacks=callbacks(config),
        )
        model.save(MODEL_DIR / "custom_cnn.keras")
        trained["Custom_CNN"] = model

    if not args.skip_mobilenet:
        print("\n=== Training MobileNetV2 ===")
        model, base = build_transfer_model("MobileNetV2", num_classes, image_size)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=transfer_epochs,
            callbacks=callbacks(config),
        )
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        compile_model(model, 1e-5)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=finetune_epochs,
            callbacks=callbacks(config),
        )
        model.save(MODEL_DIR / "mobilenetv2.keras")
        trained["MobileNetV2"] = model

    if not args.skip_resnet:
        print("\n=== Training ResNet50 ===")
        model, base = build_transfer_model("ResNet50", num_classes, image_size)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=transfer_epochs,
            callbacks=callbacks(config),
        )
        base.trainable = True
        for layer in base.layers[:-25]:
            layer.trainable = False
        compile_model(model, 1e-5)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=finetune_epochs,
            callbacks=callbacks(config),
        )
        model.save(MODEL_DIR / "resnet50.keras")
        trained["ResNet50"] = model

    if not trained:
        raise RuntimeError("All models were skipped. Train at least one model.")
    return trained


def evaluate_model(model, test_ds, class_names, name: str) -> dict:
    y_true: list[int] = []
    y_pred: list[int] = []

    start = time.perf_counter()
    for images, labels in test_ds:
        logits = model.predict(images, verbose=0)
        predictions = np.argmax(logits, axis=1)
        y_true.extend(labels.numpy().astype(int).tolist())
        y_pred.extend(predictions.astype(int).tolist())
    elapsed = time.perf_counter() - start

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    report = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report.to_csv(OUTPUT_DIR / f"{name}_classification_report.csv")

    cm = confusion_matrix(y_true, y_pred)
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


def save_comparison(results: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    metrics = results.set_index("Model")[["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]]
    ax = metrics.plot(kind="bar", figsize=(10, 5))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Fruits-360 Model Comparison")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=180)
    plt.close()


def select_deployment_model(
    results: pd.DataFrame,
    trained: dict[str, tf.keras.Model],
    class_names: list[str],
    image_size: tuple[int, int],
    config: dict,
) -> None:
    ranked = results.sort_values("Macro F1", ascending=False).reset_index(drop=True)
    best_overall = ranked.iloc[0]

    max_mb = float(config["deployment_max_mb"])
    deployable = ranked[ranked["Model MB"] <= max_mb]
    deployment_row = deployable.iloc[0] if not deployable.empty else best_overall

    deployment_name = str(deployment_row["Model"])
    source_path = {
        "Custom_CNN": MODEL_DIR / "custom_cnn.keras",
        "MobileNetV2": MODEL_DIR / "mobilenetv2.keras",
        "ResNet50": MODEL_DIR / "resnet50.keras",
    }[deployment_name]

    shutil.copy2(source_path, MODEL_DIR / "best_fruit_model.keras")
    (MODEL_DIR / "class_names.json").write_text(
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
        "deployment_limit_mb": max_mb,
        "tensorflow_version": tf.__version__,
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n=== Deployment export ===")
    print(json.dumps(metadata, indent=2))
    print(f"\nStreamlit model: {MODEL_DIR / 'best_fruit_model.keras'}")


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"Use Python 3.12 for this project. Current Python: {sys.version.split()[0]}"
        )

    args = parse_args()
    config = load_config()

    tf.keras.utils.set_random_seed(int(config["seed"]))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("TensorFlow:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    dataset_root = download_dataset(config, args.force_download)
    train_dir = find_split(dataset_root, "Training")
    test_dir = find_split(dataset_root, "Test")

    train_ds, val_ds, test_ds, class_names, image_size = make_datasets(
        train_dir,
        test_dir,
        config,
    )
    print(f"\nDetected classes: {len(class_names)}")

    trained = train_models(
        train_ds,
        val_ds,
        class_names,
        image_size,
        config,
        args,
    )

    rows = []
    for name, model in trained.items():
        print(f"\n=== Evaluating {name} ===")
        rows.append(evaluate_model(model, test_ds, class_names, name))

    results = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)
    print("\n=== Final comparison ===")
    print(results.to_string(index=False))
    save_comparison(results)
    select_deployment_model(results, trained, class_names, image_size, config)

    print("\nTraining completed.")
    print("Generated:")
    print("  models/best_fruit_model.keras")
    print("  models/class_names.json")
    print("  models/model_metadata.json")
    print("  outputs/model_comparison.csv")
    print("  outputs/model_comparison.png")


if __name__ == "__main__":
    main()

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras import callbacks


IMAGE_SIZE = (100, 100)
BATCH_SIZE = 32
SEED = 123
EPOCHS = 10


def build_model(num_classes: int) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(100, 100, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(100, 100, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.resnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)  # logits, like the notebook

    model = tf.keras.Model(inputs, outputs, name="fruit_resnet50")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def load_datasets(train_dir: Path, test_dir: Path):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
    )

    # Same seed is intentionally used so training and validation form a correct split.
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=False,
    )

    class_names = train_ds.class_names

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names


def evaluate_and_save_report(model, test_ds, class_names, output_dir: Path):
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        logits = model.predict(images, verbose=0)
        predictions = np.argmax(logits, axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predictions.tolist())

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_path = output_dir / "classification_report.csv"
    report_df.to_csv(report_path)

    print("\nOverall test results")
    print(f"Accuracy:        {report.get('accuracy', 0):.4f}")
    print(f"Macro precision: {report['macro avg']['precision']:.4f}")
    print(f"Macro recall:    {report['macro avg']['recall']:.4f}")
    print(f"Macro F1-score:  {report['macro avg']['f1-score']:.4f}")
    print(f"Saved report to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Fruits-360 ResNet50 model for the Streamlit fruit classifier."
    )
    parser.add_argument(
        "--train-dir",
        default="dataset/fruits-360_dataset/fruits-360/Training",
        help="Path to Fruits-360 Training folder.",
    )
    parser.add_argument(
        "--test-dir",
        default="dataset/fruits-360_dataset/fruits-360/Test",
        help="Path to Fruits-360 Test folder.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Maximum training epochs.",
    )
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_dir.exists():
        raise FileNotFoundError(f"Training folder not found: {train_dir.resolve()}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test folder not found: {test_dir.resolve()}")

    train_ds, val_ds, test_ds, class_names = load_datasets(train_dir, test_dir)
    print(f"Number of classes: {len(class_names)}")

    with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)

    model = build_model(len(class_names))
    model.summary()

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    checkpoint = callbacks.ModelCheckpoint(
        filepath=output_dir / "fruit_resnet50.keras",
        monitor="val_accuracy",
        save_best_only=True,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[early_stopping, checkpoint],
    )

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    # Reload the best validation model before test evaluation.
    best_model = tf.keras.models.load_model(output_dir / "fruit_resnet50.keras")
    test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    evaluate_and_save_report(best_model, test_ds, class_names, output_dir)

    print("\nTraining complete.")
    print("Run the web app with: streamlit run app.py")


if __name__ == "__main__":
    main()

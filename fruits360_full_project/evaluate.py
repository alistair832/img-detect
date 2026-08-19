from __future__ import annotations

import sys

import pandas as pd
import tensorflow as tf

from train import (
    MODEL_DIR,
    OUTPUT_DIR,
    download_dataset,
    evaluate_model,
    find_split,
    load_config,
    make_datasets,
    save_comparison,
)

MODEL_FILES = {
    "Custom_CNN": MODEL_DIR / "custom_cnn.keras",
    "MobileNetV2": MODEL_DIR / "mobilenetv2.keras",
    "ResNet50": MODEL_DIR / "resnet50.keras",
}


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"Use Python 3.12. Current Python: {sys.version.split()[0]}"
        )

    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_root = download_dataset(config, force_download=False)
    train_dir = find_split(dataset_root, "Training")
    test_dir = find_split(dataset_root, "Test")
    _, _, test_ds, class_names, _ = make_datasets(train_dir, test_dir, config)

    rows = []
    for name, path in MODEL_FILES.items():
        if not path.exists():
            print(f"Skipping {name}: {path.name} not found")
            continue

        print(f"\nLoading {name}: {path}")
        model = tf.keras.models.load_model(path, compile=False)
        rows.append(evaluate_model(model, test_ds, class_names, name))

    if not rows:
        raise FileNotFoundError(
            "No trained model files were found. Run `python train.py` first."
        )

    results = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)
    save_comparison(results)
    print("\nEvaluation results:")
    print(results.to_string(index=False))
    print(f"\nSaved: {OUTPUT_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()

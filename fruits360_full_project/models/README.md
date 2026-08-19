# Generated model files

Run `python ../train.py` from this folder's parent directory, or run `python train.py` from `fruits360_full_project/`.

The training script generates:

- `best_fruit_model.keras` — model loaded by Streamlit.
- `class_names.json` — class index to class-name mapping.
- `model_metadata.json` — selected model, test metrics, image size and TensorFlow version.

It also creates individual training models (`custom_cnn.keras`, `mobilenetv2.keras`, `resnet50.keras`). Those individual files are ignored by Git; only the selected deployment model is intended to be committed for Streamlit.

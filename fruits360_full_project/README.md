# Fruits-360 All-in-One Streamlit Assignment

This project now uses **one Python program only**:

- `app.py` — training, evaluation, results, live camera and image upload.

Streamlit only needs to run:

```text
fruits360_full_project/app.py
```

## What the one app does

### 1. Train Model
- Downloads Kaggle `moltean/fruits` (Fruits-360)
- Finds the Training and Test folders automatically
- Creates an 80/20 training-validation split
- Resizes images to 128×128
- Applies augmentation
- Trains Custom CNN
- Trains MobileNetV2
- Trains ResNet50
- Evaluates Accuracy, Precision, Recall and F1
- Selects a deployment model
- Creates `models/best_fruit_model.keras`
- Creates `models/class_names.json`
- Creates `models/model_metadata.json`

### 2. Results
Displays the model-comparison table and chart and allows the CSV/chart to be downloaded.

### 3. Detect Fruit
Provides:
- smooth front-camera recognition
- image upload
- class prediction
- confidence percentage
- Top-5 predictions

## Streamlit Community Cloud settings

Use:

- Repository: `alistair832/img-detect`
- Branch: `main`
- Main file: `fruits360_full_project/app.py`
- Python: `3.12`

## Kaggle authentication

Recommended Streamlit secret:

```toml
KAGGLE_API_TOKEN = "YOUR_KAGGLE_API_TOKEN"
```

Add it in Streamlit **Manage app → Settings → Secrets**. The app can also accept a token in the Train Model tab, but the Secrets setting is preferred.

## Training modes

**Quick pipeline test**
- one epoch per training stage
- limited evaluation batches
- used only to confirm the workflow works

**Full assignment training**
- uses the configured full epoch counts
- produces the metrics intended for the assignment report

## Important

The large Kaggle dataset and generated training files live in the Streamlit runtime filesystem and are not automatically committed back to GitHub. A Streamlit reboot/redeploy can remove runtime-generated files. For a permanent deployment model, keep a copy of the generated `best_fruit_model.keras`, `class_names.json`, and `model_metadata.json` outside the temporary runtime and add them to the repository when ready.

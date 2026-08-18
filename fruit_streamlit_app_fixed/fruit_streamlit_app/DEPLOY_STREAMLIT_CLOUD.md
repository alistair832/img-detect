# Streamlit Community Cloud Deployment

## Recommended Python version

Use **Python 3.12** in Streamlit Community Cloud.

When creating the app:

1. Select your GitHub repository.
2. Set the entrypoint to `app.py`.
3. Open **Advanced settings**.
4. Select **Python 3.12**.
5. Deploy.

If the app was already deployed using another Python version, delete that Streamlit app and redeploy it with Python 3.12.

## Deployment dependencies

`requirements.txt` contains only packages needed by the web app:

```text
streamlit==1.49.1
tensorflow-cpu==2.20.0
numpy==2.2.6
pandas==2.2.3
Pillow==11.3.0
```

The CPU TensorFlow build is used because Community Cloud does not need GPU TensorFlow for prediction and the package is smaller than the full Linux TensorFlow distribution.

## Training dependencies

For local model training, install:

```bash
pip install -r requirements-training.txt
```

This adds scikit-learn for precision, recall, F1-score, and the classification report.

## Required model files

Before deployment, your repository must include:

```text
models/fruit_resnet50.keras
models/class_names.json
```

If these files are missing, the dependency installation can succeed but the app will stop with a "Model not found" message.

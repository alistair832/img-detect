# Fruits-360 Full Assignment Project

This folder contains a complete assignment workflow:

- `Fruits360_Complete_Assignment_Training.ipynb` — Kaggle download, preprocessing, 3-model training, evaluation and model export.
- `ASSIGNMENT_REPORT.md` — report text aligned with the notebook.
- `app.py` — Streamlit live-camera + image-upload app using the trained model.
- `requirements-training.txt` — Jupyter training environment.
- `requirements.txt` — Streamlit deployment environment.
- `models/` — generated model, labels and metadata.
- `outputs/` — generated metrics, plots, reports and confusion matrices.

## Recommended workflow

1. Use Python 3.12 or 3.13 (recommended: 3.12).
2. Open Jupyter from the repository root.
3. Run `Fruits360_Complete_Assignment_Training.ipynb`.
4. Wait for all three algorithms to finish.
5. Confirm these generated files exist:
   - `models/best_fruit_model.keras`
   - `models/class_names.json`
   - `models/model_metadata.json`
6. Run:
   `streamlit run fruits360_full_project/app.py`

## Streamlit Community Cloud

Deploy with:
- Entry point: `fruits360_full_project/app.py`
- Python: 3.12 (or 3.13)

The `requirements.txt` located beside this app takes precedence over the root requirements file on Streamlit Community Cloud.

## Important

The full 8GB-class Kaggle dataset is downloaded by the notebook and should NOT be committed to GitHub.
The trained model is also generated after training. If it exceeds GitHub's normal file-size limit, use Git LFS or run Streamlit locally for the assignment demonstration.

## Streamlit dependency troubleshooting

The full TensorFlow deployment must not use Python 3.14. TensorFlow 2.21 supports Python through 3.13. If Streamlit Cloud shows an installer failure or the app reports that TensorFlow is unavailable, delete that Streamlit deployment and create it again with Python 3.12 or 3.13 in Advanced settings.

# Fruits-360 Full Assignment Project

This folder is the GitHub-first version of the fruit image classification assignment.

## Main workflow

```text
GitHub source code
      ↓
python train.py
      ↓
Kaggle Fruits-360 download
      ↓
Preprocessing + augmentation
      ↓
Custom CNN + MobileNetV2 + ResNet50
      ↓
Accuracy + Precision + Recall + F1 + speed
      ↓
Select a deployable model
      ↓
models/best_fruit_model.keras
models/class_names.json
models/model_metadata.json
      ↓
Push generated deployment files to GitHub
      ↓
Streamlit Cloud loads the model from this repository
```

## Files

- `train.py` — primary training command. Downloads Fruits-360, trains all selected models, evaluates them and exports the Streamlit model.
- `evaluate.py` — reevaluates already-trained model files without retraining.
- `training_config.json` — batch size, image size, epochs, seed and deployment file-size setting.
- `app.py` — Streamlit front-camera and uploaded-image prediction interface.
- `ASSIGNMENT_REPORT.md` — assignment write-up aligned with the implementation.
- `Fruits360_Complete_Assignment_Training.ipynb` — optional notebook version. `train.py` is now the recommended path.
- `requirements-training.txt` — packages required for model training.
- `requirements.txt` — packages required by Streamlit Cloud.

## 1. Training environment

Use Python **3.12** so the training and Streamlit TensorFlow environments match.

From this folder:

```bash
python -m pip install -r requirements-training.txt
```

The first Kaggle download may request authentication. `train.py` automatically calls `kagglehub.login()` if the initial download cannot proceed.

## 2. Quick pipeline test

Before a long training run:

```bash
python train.py --quick
```

`--quick` uses one epoch for each training stage. It tests the complete pipeline but its model is not intended as the final assignment result.

You can also skip expensive models while testing:

```bash
python train.py --quick --skip-resnet
```

## 3. Full assignment training

Run:

```bash
python train.py
```

The script automatically:

1. downloads Kaggle dataset `moltean/fruits`;
2. finds the best `Training` and `Test` folders, preferring the 100x100 branch;
3. discovers the class names automatically;
4. creates an 80/20 training-validation split;
5. resizes images to the configured size;
6. applies flip, rotation, zoom and contrast augmentation;
7. trains Custom CNN;
8. trains and fine-tunes MobileNetV2;
9. trains and fine-tunes ResNet50;
10. evaluates Accuracy, Macro Precision, Macro Recall, Macro F1 and inference speed;
11. saves class-level reports and confusion matrices;
12. selects the strongest model that also fits the configured deployment-size target;
13. exports the exact files used by Streamlit.

## 4. Generated results

After successful training:

```text
models/
├── custom_cnn.keras
├── mobilenetv2.keras
├── resnet50.keras
├── best_fruit_model.keras
├── class_names.json
└── model_metadata.json

outputs/
├── model_comparison.csv
├── model_comparison.png
├── Custom_CNN_classification_report.csv
├── MobileNetV2_classification_report.csv
├── ResNet50_classification_report.csv
└── *_confusion_matrix.npy
```

The individual `.keras` training models remain ignored by Git. The selected `best_fruit_model.keras` is explicitly allowed so it can be committed for Streamlit deployment.

## 5. Re-evaluate without retraining

After models have been trained:

```bash
python evaluate.py
```

This recreates `outputs/model_comparison.csv` and the comparison chart from the existing model files.

## 6. Push the final Streamlit files to GitHub

After `train.py` finishes:

```bash
git add models/best_fruit_model.keras \
        models/class_names.json \
        models/model_metadata.json \
        outputs/model_comparison.csv \
        outputs/model_comparison.png

git commit -m "Add trained Fruits-360 deployment model"
git push
```

Do **not** commit the downloaded multi-gigabyte dataset. The `data/` folder is ignored.

## 7. Streamlit Community Cloud

Use:

```text
Repository: alistair832/img-detect
Branch: main
Main file: fruits360_full_project/app.py
Python: 3.12
```

Once these files exist in GitHub:

```text
models/best_fruit_model.keras
models/class_names.json
models/model_metadata.json
```

`app.py` loads them and exposes:

- live front-camera classification;
- image upload classification;
- prediction confidence;
- Top-5 predictions;
- test accuracy and selected model metadata.

## Training configuration

Edit `training_config.json` rather than changing the Python source for normal tuning.

Default configuration:

```json
{
  "image_size": 128,
  "batch_size": 32,
  "validation_split": 0.2,
  "cnn_epochs": 8,
  "transfer_epochs": 8,
  "finetune_epochs": 4,
  "deployment_max_mb": 90
}
```

## Important distinction

GitHub is the source repository for the complete code and the final generated deployment model. Streamlit Cloud is the inference/output layer. The large Fruits-360 training dataset is downloaded by `train.py` when training is executed and is not stored in the repository.

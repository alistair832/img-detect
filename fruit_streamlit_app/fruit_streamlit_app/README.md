# Fruit Image Detection - Streamlit + ResNet50

This project converts the uploaded Fruits-360 ResNet notebook into a Streamlit fruit image classification demo.

## Model used

- Dataset: Fruits-360
- Task: Multi-class fruit image classification
- Input: 100 x 100 RGB image
- Model: ResNet50 transfer learning using ImageNet weights
- Augmentation: horizontal flip + random rotation
- Optimizer: Adam (learning rate 0.0001)
- Loss: SparseCategoricalCrossentropy(from_logits=True)
- Metrics produced after training: accuracy, precision, recall and F1-score

The uploaded notebook reports 131 classes and reached validation accuracy of approximately 99.99% in its final epoch. Real-world camera images may perform worse because Fruits-360 images are highly controlled.

## Project structure

```text
fruit_streamlit_app/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── dataset/
│   └── fruits-360_dataset/
│       └── fruits-360/
│           ├── Training/
│           └── Test/
└── models/
    ├── fruit_resnet50.keras       # created after training
    ├── class_names.json           # created after training
    ├── training_history.csv       # created after training
    └── classification_report.csv  # created after training
```

## 1. Install packages

Open Command Prompt / PowerShell inside the project folder:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Then:

```bash
pip install -r requirements.txt
```

## 2. Add the Fruits-360 dataset

Place the dataset so these folders exist:

```text
dataset/fruits-360_dataset/fruits-360/Training
dataset/fruits-360_dataset/fruits-360/Test
```

Alternatively, pass your own paths:

```bash
python train_model.py --train-dir "YOUR_TRAINING_PATH" --test-dir "YOUR_TEST_PATH"
```

## 3. Train and save the model

```bash
python train_model.py
```

The trained model will be saved as:

```text
models/fruit_resnet50.keras
```

## 4. Run Streamlit

```bash
streamlit run app.py
```

Streamlit will show a local URL, normally:

```text
http://localhost:8501
```

## If you already trained the uploaded notebook

At the end of the notebook, run:

```python
import json
from pathlib import Path

Path("models").mkdir(exist_ok=True)
model.save("models/fruit_resnet50.keras")

with open("models/class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)
```

Then copy the `models` folder beside `app.py` and run:

```bash
streamlit run app.py
```

## Important correction from the notebook

The original notebook's prediction function uses `images[i]` instead of its `img` argument and treats the largest raw logit as a confidence percentage. The Streamlit version fixes this by using the actual uploaded image and applying softmax to the output logits.

# Fruit Image Detection / Classification — Assignment Report

## 1.0 Introduction

Image processing and computer vision enable computers to extract useful information from digital images. Common applications include optical character recognition, face recognition, medical-image analysis, image classification, object detection and image segmentation. This project focuses on **fruit image classification** using the Fruits-360 dataset. The system receives an image containing one main object and predicts the corresponding fruit, vegetable, nut or seed class.

The practical objective is to create an end-to-end artificial-intelligence workflow that can be demonstrated in a Streamlit application. The workflow covers dataset acquisition, preprocessing, data augmentation, model training, evaluation, model comparison and deployment.

## 1.1 Problem Statement

Manual recognition is easy for familiar fruit but becomes more difficult when a dataset contains many varieties with similar appearance. Apples, pears, tomatoes, peppers and other classes may have similar colour, shape or texture. The computer-vision problem is therefore to learn discriminative visual features that allow an image to be assigned to the correct class.

This assignment implements **multi-class image classification**. It should not be confused with object detection. Classification predicts the class of the image, while object detection additionally predicts the location of objects with bounding boxes.

## 1.2 Objectives

1. Download the Fruits-360 dataset directly from Kaggle.
2. Explore the dataset and automatically identify the current class count.
3. Resize and preprocess the images for neural-network training.
4. Apply data augmentation to improve robustness.
5. Train and compare a Custom CNN, MobileNetV2 and ResNet50.
6. Measure Accuracy, Precision, Recall and F1-score.
7. Discuss the strengths and weaknesses of each algorithm.
8. Save the selected model and class labels.
9. Demonstrate the trained system using Streamlit with a live camera and image upload.

---

## 2.0 Background Study

### 2.1 Convolutional Neural Networks

A Convolutional Neural Network (CNN) is designed to learn visual patterns directly from image pixels. Convolutional filters capture local structures such as edges, colours, shapes and textures. Pooling reduces spatial dimensions, while deeper convolutional layers learn increasingly abstract features. In this project, a custom CNN is used as a baseline because its complete feature representation is learned from the Fruits-360 training images.

### 2.2 Transfer Learning

Transfer learning reuses visual features learned by a neural network on a large source dataset. Instead of learning all filters from random initialization, the project loads ImageNet-pretrained networks, replaces the original classification head and trains a new head for the Fruits-360 classes. A second stage can unfreeze selected upper layers for fine-tuning. TensorFlow describes both feature extraction and fine-tuning as standard transfer-learning strategies.

### 2.3 MobileNetV2

MobileNetV2 is an efficient convolutional architecture developed for lower-computation environments. Its main value in this assignment is the balance between recognition performance and inference efficiency. A lighter model is useful for Streamlit because live webcam recognition must repeatedly process frames.

### 2.4 ResNet50

ResNet50 is a deeper residual neural network. Residual connections allow information and gradients to pass through shortcut paths, helping deep networks train effectively. ResNet50 provides a high-capacity transfer-learning comparison. The supplied reference Fruits-360 notebook also used ResNet50, so it provides continuity with the earlier experiment while the new notebook adds a cleaner evaluation pipeline.

### 2.5 YOLO and Mask R-CNN

YOLO is designed mainly for real-time object detection. It predicts object classes and bounding boxes and is appropriate when several objects may appear in one image. Mask R-CNN performs instance segmentation by adding a pixel-level object mask to detection. These methods are important for advanced computer vision, but the main Fruits-360 classification branch is organized into class folders containing one primary object per image. Therefore, this project uses classification models for its main experiment.

---

## 3.0 Dataset

The project uses the **Fruits-360** dataset from Kaggle (`moltean/fruits`). The dataset is actively updated and contains multiple branches, including a standardized 100×100 image branch and an original-size branch. Because the number of classes can change as the dataset is updated, the notebook does not hard-code a fixed class count. It searches for the Training and Test directories and reads their class subdirectories dynamically.

The selected classification branch provides separate Training and Test folders. The Training directory is further divided by the notebook into training and validation subsets. This creates three evaluation stages:

- Training data — used to optimize model weights.
- Validation data — used during model development and early stopping.
- Test data — reserved for final performance measurement.

---

## 4.0 Data Preprocessing

### 4.1 Resizing

All images are resized to **128×128 RGB**. The original standardized branch contains smaller images, but 128×128 provides a consistent input shape that is still computationally manageable for MobileNetV2 and ResNet50.

### 4.2 Label Encoding

Class-folder names are automatically converted into integer labels using `tf.keras.utils.image_dataset_from_directory()`. A `class_names.json` file is exported after training so Streamlit can convert a predicted class index back into a readable class name.

### 4.3 Validation Split

Twenty percent of the Training directory is reserved for validation. The same random seed is used for both training and validation subset generation so the partitions remain reproducible and non-overlapping.

### 4.4 Data Augmentation

The project applies:
- horizontal flipping;
- random rotation;
- random zoom;
- random contrast.

Augmentation introduces controlled image variation and helps reduce excessive dependence on the exact training orientation or lighting.

### 4.5 Normalization

The custom CNN uses pixel rescaling to the range 0–1. MobileNetV2 and ResNet50 use their respective TensorFlow preprocessing functions so input values match the conventions used during ImageNet pretraining.

---

## 5.0 Model Development

### 5.1 Algorithm 1 — Custom CNN

The custom CNN contains convolution, batch-normalization and pooling blocks followed by global average pooling, dropout and a dense classification layer. This model provides a baseline without external pretrained image features.

**Advantages**
- Easy to understand and explain.
- Fully customized to the project dataset.
- Useful baseline for comparing transfer learning.

**Disadvantages**
- Starts with random feature weights.
- Requires more training to learn general visual features.
- Can generalize less effectively than pretrained networks.

### 5.2 Algorithm 2 — MobileNetV2

MobileNetV2 is initialized with ImageNet weights. The base network is first frozen while a new classification head is trained. The upper layers are then fine-tuned using a lower learning rate.

**Advantages**
- Efficient inference.
- Lower model size than ResNet50.
- Strong candidate for live Streamlit recognition.
- Benefits from ImageNet transfer learning.

**Disadvantages**
- Lower capacity than large convolutional networks.
- Fine-tuning can reduce performance if the learning rate is too high.

### 5.3 Algorithm 3 — ResNet50

ResNet50 is also initialized using ImageNet weights. The transfer-learning and fine-tuning process is similar to MobileNetV2.

**Advantages**
- High feature-representation capacity.
- Residual architecture supports deep learning.
- Strong image-classification performance.

**Disadvantages**
- Larger model.
- Higher memory requirement.
- Slower inference for live camera processing.

---

## 6.0 Evaluation Method

The models are evaluated using the independent Test folder.

### Accuracy

Accuracy measures the proportion of all test images that are classified correctly.

### Precision

Precision measures how often a predicted class is correct.

### Recall

Recall measures the model's ability to find examples belonging to each actual class.

### F1-score

F1-score combines Precision and Recall. Macro F1 is used so every class contributes equally to the final score.

### Inference Speed

The notebook also measures test-set prediction time and images per second. This provides practical information for Streamlit deployment.

### Final Results Table

**Do not invent values here. Run the notebook and copy the generated `outputs/model_comparison.csv` results into this table.**

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Images/Second |
|---|---:|---:|---:|---:|---:|
| Custom CNN | Generated by notebook | Generated | Generated | Generated | Generated |
| MobileNetV2 | Generated by notebook | Generated | Generated | Generated | Generated |
| ResNet50 | Generated by notebook | Generated | Generated | Generated | Generated |

The notebook automatically saves class-level classification reports and confusion matrices for deeper analysis.

---

## 7.0 Model Comparison and Interpretation

The expected comparison should consider both predictive performance and deployment cost. A model with the highest test accuracy is not automatically the best practical choice if its model file is very large or live inference is slow. MobileNetV2 may be selected for deployment even if ResNet50 achieves slightly stronger accuracy, because MobileNetV2 is designed for efficient inference.

The Custom CNN demonstrates how much performance can be obtained without pretrained features. If MobileNetV2 and ResNet50 outperform the Custom CNN, this provides evidence that transfer learning is useful for this image-classification problem.

---

## 8.0 Real-Life Demonstration Using Streamlit

The Streamlit application provides two modes.

### Live Front Camera

The application starts a front-facing webcam stream. A green guide box indicates the region used for classification. To reduce CPU usage, the application processes selected frames rather than every frame. Several recent predictions are combined to improve visual stability.

### Upload Picture

The user can upload JPG, JPEG, PNG or WEBP images. The system displays:
- predicted class;
- confidence percentage;
- Top-5 predictions.

This mode is useful when a physical fruit is not available during presentation.

---

## 9.0 Limitations

The Fruits-360 dataset contains many images captured in controlled conditions with a single prominent object. A network can therefore achieve very high dataset accuracy while performing less reliably on cluttered real-world scenes. Background complexity, lighting, scale, occlusion and multiple objects may reduce Streamlit camera accuracy.

A future extension could use YOLO to automatically detect and localize multiple fruits before classification, or train directly on more diverse real-world images.

---

## 10.0 Conclusion

This project develops a complete computer-vision pipeline for fruit image classification. It automatically downloads Fruits-360, preprocesses the dataset, trains three algorithms, evaluates Accuracy/Precision/Recall/F1, exports the selected model and deploys it in a Streamlit interface.

The final conclusion should be completed using the actual generated metrics. State which algorithm achieved the highest test performance and explain whether it was also the most appropriate model for live deployment. The comparison should demonstrate the trade-off between model accuracy, model size and inference speed.

---

## References

- Kaggle. *Fruits-360 dataset* (`moltean/fruits`).
- Kaggle. *kagglehub: Python library to access Kaggle resources*.
- TensorFlow. *Transfer learning and fine-tuning*.
- TensorFlow. *Load and preprocess images*.
- TensorFlow. *tf.keras.utils.image_dataset_from_directory*.
- Muresan, H., & Oltean, M. (2018). *Fruit recognition from images using deep learning*. Acta Universitatis Sapientiae, Informatica, 10(1), 26–42.

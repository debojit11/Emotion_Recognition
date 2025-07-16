# Facial Emotions Detection

A real-time facial emotion recognition system using ResNet18 and the Balanced RAF-DB dataset, built with PyTorch, OpenCV, and Streamlit. This project detects 7 basic emotions from live webcam input.

---

## 🔍 Overview

This project implements a deep learning-based pipeline to recognize human emotions in real time using grayscale facial images. The model is trained on a class-balanced, preprocessed version of the RAF-DB dataset, achieving **92.48% test accuracy** in just 15 epochs. The application runs in your browser using Streamlit and captures webcam video to detect emotions on-the-fly.

---

## 🚀 Tech Stack

- PyTorch  
- Torchvision  
- OpenCV  
- Streamlit  
- ResNet18 (pretrained)

---

## 📊 Model Summary

- Architecture: ResNet18 (pretrained)
- Input Shape: 1×75×75 (grayscale)
- Output Classes: 7 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- Best Accuracy: Val: 92.32%, Test: 92.48% (Epoch 15)
- Final Loss: 0.2333

---

## 🗂 Dataset: Balanced RAF-DB

https://www.kaggle.com/datasets/dollyprajapati182/balanced-raf-db-dataset-7575-grayscale

The Balanced RAF-DB Dataset is an augmented and class-equalized version of the Real-world Affective Faces Database (RAF-DB), tailored for robust facial emotion recognition (FER) tasks. This version ensures uniform distribution across the seven primary emotion classes with standardized image preprocessing, making it ideal for machine learning and deep learning applications.

🎯 Purpose  
This dataset aims to address class imbalance and variability in the original RAF-DB dataset. It provides a machine-learning-ready, grayscale, and uniformly preprocessed dataset for researchers and developers building models for facial emotion detection.

🧾 Dataset Characteristics  
Source: Derived from the original RAF-DB dataset  
Image Size: 75 × 75 pixels  
Emotion Classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral  
Total Images: 62,916 (8,988 images per class)

⚙️ Preprocessing Pipeline  
- Converted all images to grayscale  
- Resized to 75×75 pixels  
- Applied data augmentation:  
  - Random rotations  
  - Horizontal flipping  
  - Brightness, contrast, and sharpness adjustment  
- Balanced each emotion class to contain exactly 8,988 images  
- Split into:  
  - Training: 80% (6,472 images/class)  
  - Validation: 20% (1,618 images/class)  
  - Test: 10% (898 images/class)

✅ Advantages  
- Fixes the class imbalance present in the original RAF-DB dataset  
- Enables better generalization and fairness in training models  
- Preprocessed and formatted for quick deployment in ML/DL pipelines  
- Consistent grayscale format and resolution simplifies input standardization

---

## 🖥 How to Run the App

### 1️⃣ Create Conda Environment

```
conda create -n emotion_env python=3.10 -y
conda activate emotion_env
```

### 2️⃣ Install Dependencies

```
pip install torch torchvision opencv-python streamlit
```

### 3️⃣ Run the App

```
streamlit run app.py
```

---

## 📈 Training Performance

```
Epoch 15: Loss=0.2333, Val Acc=92.32%, Test Acc=92.48%
```

---

## 📂 Project Structure

```
Emotion_Recognition/
├── app.py
├── best_model.pt
├── notebook.ipynb
├── README.md
```

---

## ⚠️ Disclaimer:
This project was developed during an internship. It is intended for personal portfolio purposes only.
All rights reserved. Please do not reuse or redistribute without permission.
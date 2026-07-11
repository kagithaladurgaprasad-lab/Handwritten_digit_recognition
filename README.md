# ✍️ Air Handwritten Digit Recognition using CNN & MediaPipe

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Project Overview

This project is a **real-time AI application** that recognizes handwritten digits written in the air using a webcam.

The application combines **Computer Vision**, **Deep Learning**, and **MediaPipe Hand Tracking** to capture finger movements, convert them into digital images, and classify the handwritten digit using a trained Convolutional Neural Network (CNN).

---

# 🎯 Objective

Build an end-to-end AI application capable of:

- Detecting hand landmarks using MediaPipe
- Tracking the index fingertip
- Drawing digits in the air
- Preprocessing the drawing
- Predicting the handwritten digit using a CNN model
- Displaying prediction confidence in real time

---

# 🚀 Features

✅ Real-time hand tracking

✅ Air writing using index finger

✅ Gesture-based controls

✅ Virtual drawing canvas

✅ Image preprocessing

✅ CNN digit prediction

✅ Real-time confidence score

✅ Modular project architecture

---

# 🛠 Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn

---

# 📂 Project Structure

```
handwritten_project/
│
├── app.py
├── hand_tracker.py
├── gesture_controller.py
├── virtual_canvas.py
├── image_processor.py
├── digit_predictor.py
│
├── handwritten_digit_cnn.keras
├── hand_landmarker.task
├── requirements.txt
└── README.md
```

---

# 🔄 Project Workflow

```
Webcam
    │
    ▼
MediaPipe Hand Tracking
    │
    ▼
Index Finger Detection
    │
    ▼
Virtual Canvas
    │
    ▼
Image Preprocessing
    │
    ▼
CNN Model
    │
    ▼
Digit Prediction
    │
    ▼
Display Result
```

---

# 🧠 CNN Model Workflow

1. Import Dataset
2. Exploratory Data Analysis (EDA)
3. Image Preprocessing
4. Feature Extraction
5. Input–Output Separation
6. Train-Test Split
7. Baseline CNN Model
8. Hyperparameter Tuning
9. Best CNN Model
10. Model Evaluation
11. Model Saving
12. Model Loading

---

# ✋ Gesture Controls

| Gesture | Action |
|----------|--------|
| ☝️ Index Finger | Draw |
| ✌️ Index + Middle | Stop Drawing |
| ✊ Fist | Clear Canvas |
| 🖐 Open Palm | Predict Digit |

---

# 📊 Model Performance

| Metric | Score |
|---------|--------|
| Training Accuracy | 99%+ |
| Validation Accuracy | 99%+ |
| Test Accuracy | 99%+ |

> Performance may vary depending on air-written digit quality and lighting conditions.

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/air-handwritten-digit-recognition.git
```

Go to the project folder

```bash
cd air-handwritten-digit-recognition
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

---


# 📈 Future Improvements

- Better gesture recognition
- Smooth handwriting using Bézier curves
- Multi-digit recognition
- Word recognition
- Custom CNN training
- Streamlit Web Application
- Deployment using Hugging Face Spaces

---

# 💡 Skills Demonstrated

- Deep Learning
- CNN
- Computer Vision
- Image Processing
- OpenCV
- MediaPipe
- TensorFlow
- Real-Time AI Applications
- Python Programming
- Model Deployment

---

# 👨‍💻 Developed By

**Kagithala Durga Prasad**

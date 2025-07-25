# Sign Language to Text Translator

A real-time hand gesture recognition system using Python, OpenCV, and a CNN (Keras/TensorFlow) to translate American Sign Language (ASL) alphabet (A–Y, excluding J and Z) into text.

---

## 📌 Project Overview

This project captures hand gestures from your webcam, uses a trained deep learning model to classify the gesture as an ASL letter, and displays the predicted letter on the screen in real time. It is designed for accessibility, learning, and as a demonstration of computer vision and deep learning techniques.

---

## ✨ Features
- Real-time webcam-based gesture recognition
- CNN model for high accuracy
- Uses the public Sign Language MNIST dataset (no manual data collection required)
- Simple OpenCV-based interface (no GUI, no text-to-speech)
- Error handling for missing model or camera
- **Live, animated feedback:** The OpenCV window updates instantly as you move your hand, and the prediction text changes in real time as you sign different letters.

---

## 🗂️ Folder Structure
```
SignLang AI/
├── dataset/
│   ├── sign_mnist_train.csv
│   └── sign_mnist_test.csv
├── model/
│   ├── train.py
│   └── sign_model.h5
├── src/
│   ├── predict.py
│   └── collect_data.py  # (not used, for manual data collection)
├── requirements.txt
├── README.md
└── app.py  # (entry point, not required for basic usage)
```

---

## 🧰 Requirements
- Python 3.8+
- OpenCV
- TensorFlow
- Keras
- NumPy
- scikit-learn
- pandas
- Pillow

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## 🖼️ Dataset
- Uses the [Sign Language MNIST dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- Place `sign_mnist_train.csv` and `sign_mnist_test.csv` in the `dataset/` folder
- **Note:** Only 24 letters are supported (A–Y, excluding J and Z)

---

## 🚀 Setup & Usage

### 1. Clone the Repository
```bash
git clone <repo-url>
cd SignLang\ AI
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On Linux/Mac
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Download and Place Dataset
- Download `sign_mnist_train.csv` and `sign_mnist_test.csv` from Kaggle
- Place them in the `dataset/` folder

### 5. Train the Model
```bash
python model/train.py
```
- This will create `model/sign_model.h5`

### 6. Run Real-Time Prediction
```bash
python src/predict.py
```
- An OpenCV window will open showing your webcam feed and the predicted letter
- Show your hand in the ROI box, mimicking the MNIST dataset’s style (single hand, palm facing camera, plain background)
- Press `q` to quit

---

## 📸 Example Screenshots & Real-Time Demo

Below are example screenshots from the working project. The OpenCV window provides **live, animated feedback** as you sign different letters:

![Real-time prediction example: Q](amer_sign2.png)
*The system predicts the letter 'Q' as you show the corresponding sign in the ROI box.*

![Real-time prediction example: T](amer_sign3.png)
*The system predicts the letter 'T' as you show the corresponding sign in the ROI box.*

- The left side of the window shows your webcam feed with a blue rectangle marking the region of interest (ROI) for your hand.
- The predicted letter (e.g., "Prediction: Q") is displayed in large green text at the top left.
- As you move your hand and change the sign, the prediction updates instantly, providing a real-time, animated experience.

> **Tip:** You can add your own screenshots by saving images from the OpenCV window and placing them in the project folder. Update the README to include your images if desired.

---

## 🛠️ Error Handling
- If the model file is missing, you will see an error: "Model file not found..."
- If the webcam is not found, you will see an error: "Webcam not found..."
- The script will exit gracefully in these cases

---

## 🐍 Python Version & OS Compatibility
- **Python:** 3.8 or higher recommended
- **OS:** Works on Windows, Linux, macOS (webcam required)

---

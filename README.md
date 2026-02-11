# Skin Type Detection

A deep learning–based system that detects facial skin type from a selfie image using computer vision and Convolutional Neural Networks (CNN), and provides personalized skincare product recommendations based on the detected skin type and user budget.

---

## 📌 Overview

Choosing suitable skincare products requires understanding one's skin type. This project automates skin type detection using image preprocessing and deep learning techniques. 

The system analyzes facial texture, brightness, and surface characteristics from a selfie image to classify the skin type. Based on the prediction, it recommends suitable skincare products and filters them according to the user's budget.

This reduces guesswork and supports more personalized skincare selection.

---

## 🎯 Objectives

- Detect skin type from a selfie image
- Use CNN-based deep learning for classification
- Apply image preprocessing techniques
- Develop a personalized product recommendation module
- Enable budget-based filtering of skincare products

---

## Skin Types Classified

The model classifies skin into:

- Oily
- Dry
- Normal
- Combination
- Sensitive

---

## Product Recommendation Module

After detecting the skin type, the system:

- Filters products suitable for the predicted skin type
- Allows users to specify a budget range
- Returns relevant and affordable skincare products
- Can be extended to include ingredient-based filtering

---

## System Workflow

User Image  
   ↓  
Image Preprocessing  
   ↓  
CNN Model  
   ↓  
Skin Type Prediction  
   ↓  
Product Recommendation Engine  
   ↓  
Budget-Based Filtering  
   ↓  
Final Output  

---

## Dataset

- Skin type image dataset (Oily, Dry, Normal, Combination, Sensitive)
- Skincare product dataset (Product name, category, suitable skin type, price)
- Dataset split into training, validation, and testing sets

---

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/skin-type-detection.git
cd skin-type-detection
```

### 2️⃣ Create Virtual Environment (Optional)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Train the Model

```bash
python train.py
```

### Predict Skin Type

```bash
python predict.py
```

### Run Full System (Detection + Recommendation)

```bash
python app.py
```

---

## Model Evaluation

- Accuracy
- Loss curve
- Confusion matrix
- Precision & Recall

(Replace with actual results after training)

---

## Future Enhancements

- Real-time webcam skin detection
- Microscopic texture analysis
- Ingredient-based product recommendation
- Weather-aware skincare suggestions
- Web deployment using Streamlit or Flask

---

## 👨‍💻 Team

- Allan Nizar
- Anirudh Raj V
- Vishnu N
- Yedhun P A

Guide: Anupa Tom  
Department of Artificial Intelligence & Data Science
Muthoot Institute of Technology and Science

---

## License

This project is developed for academic purposes as part of a Mini Project.


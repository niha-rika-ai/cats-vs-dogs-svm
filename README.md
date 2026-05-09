# 🐶🐱 Cats vs Dogs Image Classification using SVM

## 📌 Project Overview
This project implements a **Support Vector Machine (SVM)** model to classify images of cats and dogs using the Kaggle Dogs vs Cats dataset.

The project focuses on classical machine learning techniques for computer vision, including image preprocessing, feature extraction, model training, evaluation, and deployment using Streamlit.

---

## 🚀 Features
- 🖼️ Image preprocessing using OpenCV
- 🧠 HOG (Histogram of Oriented Gradients) feature extraction
- 📊 SVM image classification
- 📈 Model evaluation using confusion matrix
- 🌐 Interactive Streamlit web application
- 📷 Upload and classify custom images
- 🔥 Confidence score prediction

---

## 📂 Dataset
Dataset Used:
- Kaggle Dogs vs Cats Dataset

The original dataset contained approximately **25,000 images**.  
To optimize SVM training and reduce computational load, the dataset was reduced and organized into a smaller balanced dataset.

---

## 🛠️ Technologies Used
- Python
- OpenCV
- Scikit-learn
- Streamlit
- NumPy
- Matplotlib
- scikit-image

---

## 🧠 Model Details
- Algorithm: Support Vector Machine (SVM)
- Feature Extraction: HOG (Histogram of Oriented Gradients)
- Image Size: 64×64
- Classes:
  - Cat 🐱
  - Dog 🐶

---

## 📊 Model Performance
- Final Accuracy: **75.75%**

Using HOG features significantly improved the model performance compared to using raw image pixels.

---

## ▶️ How to Run the Project
1️⃣ Install Dependencies
 - pip install -r requirements.txt
2️⃣ Preprocess Images
 - python src/preprocess.py
3️⃣ Train the SVM Model
 - python src/train_model.py
4️⃣ Evaluate the Model
 - python src/evaluate_model.py
5️⃣ Run the Streamlit App
 - python -m streamlit run app/app.py

---
## 📸 Screenshots
<img width="583" height="284" alt="Screenshot 2026-05-09 151447" src="https://github.com/user-attachments/assets/646f2078-e5c4-4558-b132-ee39d2842ce7" />
<img width="1612" height="535" alt="1" src="https://github.com/user-attachments/assets/b2eac1a9-0456-4bfb-b40e-2d8cebe27662" />
<img width="1580" height="922" alt="2" src="https://github.com/user-attachments/assets/7165e478-0654-4a93-a073-c9b6d21a2414" />
<img width="1025" height="865" alt="3" src="https://github.com/user-attachments/assets/b80195a2-0870-4465-af21-1f3aa6cf6962" />
<img width="991" height="495" alt="4" src="https://github.com/user-attachments/assets/87e91f55-7dfc-469e-a07a-06f0fda59d39" />

---

## 💡 Key Learnings

Through this project, I learned:

Image preprocessing techniques
Feature extraction using HOG
Working with large image datasets
Dataset optimization for machine learning
Building ML-powered web applications using Streamlit

---

import streamlit as st
import numpy as np
import cv2
import joblib

from PIL import Image
from skimage.feature import hog

# ---------------- LOAD MODEL ----------------
model = joblib.load("model/svm_model.pkl")

IMG_SIZE = 64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐶",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

h1 {
    color: white;
    text-align: center;
}

.result-box {
    background-color: #1e5631;
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    width: 100%;
    height: 3em;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🐶🐱 Cats vs Dogs Classification using SVM")

st.write(
    "Upload an image and the SVM model will predict whether it is a Cat or Dog."
)

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------
if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to grayscale
    img_array = np.array(image.convert('L'))

    # Resize image
    resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    # Extract HOG features
    features = hog(
        resized,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    # Reshape for prediction
    features = features.reshape(1, -1)

    # Prediction
    prediction = model.predict(features)
    confidence = abs(model.decision_function(features)[0])

    # Label
    label = "Dog 🐶" if prediction[0] == 1 else "Cat 🐱"

    # Result box
    st.markdown(f"""
    <div class='result-box'>
        Prediction: {label}<br><br>
        Confidence Score: {confidence:.2f}
    </div>
    """, unsafe_allow_html=True)

    # Model insights
    st.subheader("📊 Model Insights")

    st.write("""
    This model uses:
    - HOG (Histogram of Oriented Gradients) feature extraction
    - Support Vector Machine (SVM) classification

    HOG helps detect shapes and edges in images, while SVM separates cats and dogs using learned patterns.
    """)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 About Project")

st.sidebar.write("""
This project classifies images of Cats and Dogs using:
- OpenCV
- HOG Features
- SVM Classifier
- Streamlit UI
""")

st.sidebar.write("Model Accuracy: ~75.75%")
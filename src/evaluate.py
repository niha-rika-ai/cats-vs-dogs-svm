import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Load data
X = np.load("data/X.npy")
y = np.load("data/y.npy")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load model
model = joblib.load("model/svm_model.pkl")

# Predictions
predictions = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, predictions)

# Plot
fig, ax = plt.subplots(figsize=(6, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Cat", "Dog"]
)

disp.plot(ax=ax)

plt.title("Confusion Matrix")

plt.savefig("assets/confusion_matrix.png")

print("Confusion matrix saved!")
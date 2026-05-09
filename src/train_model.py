import numpy as np
import joblib
import os

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# Load processed data
X = np.load("data/X.npy")
y = np.load("data/y.npy")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM model
model = SVC(kernel='rbf')

print("Training SVM model...")

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/svm_model.pkl")

print(classification_report(y_test, predictions))
print("Model saved successfully!")
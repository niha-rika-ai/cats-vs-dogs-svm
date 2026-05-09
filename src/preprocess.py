import os
import cv2
import numpy as np

from skimage.feature import hog

IMG_SIZE = 64

categories = ["cat", "dog"]

training_data = []

# Process images
def create_training_data():

    for category in categories:

        path = os.path.join("data/train", category)

        class_num = categories.index(category)

        for img in os.listdir(path):

            try:
                img_path = os.path.join(path, img)

                # Read image
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # Resize
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                # HOG features
                features = hog(
                    resized_array,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    feature_vector=True
                )

                training_data.append([features, class_num])

            except Exception as e:
                pass


create_training_data()

print("Images processed:", len(training_data))

# Split features and labels
X = []
y = []

for features, label in training_data:

    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Save arrays
np.save("data/X.npy", X)
np.save("data/y.npy", y)

print("Preprocessing completed!")
import joblib
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import cv2
import os

# Set the path to the image folder
image_folder = "D:/PycharmProjects/mP2/photos"
image_path = ""
model = joblib.load('D:/PycharmProjects/mst/trained_model_hue.pkl')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
class_names = ["Cool", "Warm"]
# Function to crop face, resize, and preprocess for model prediction
def preprocess_image_for_prediction(image_path, target_size=(150, 150)):
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Assuming the first detected face
        face = img[y:y + h, x:x + w]

        # Resize the face to the target size
        face = cv2.resize(face, target_size)

        # Preprocess the image for model prediction
        face = face.astype(np.float32) / 255.0  # Rescale pixel values to [0, 1]
        face = np.expand_dims(face, axis=0)  # Add batch dimension

        return face
    else:
        return None

# Function to process a folder of images
def process_image_folder(image_folder, class_name):
    for filename in os.listdir(image_folder):
        if filename.endswith((".jpg",".jpeg",".png", ".jfif")):
            image_path = os.path.join(image_folder, filename)
            preprocessed_image = preprocess_image_for_prediction(image_path)
            if preprocessed_image is not None:
                predictions = model.predict(preprocessed_image)
                predicted_class = class_names[np.argmax(predictions)]

                print(f'Image: {filename}, Predicted class: {predicted_class}')


def preprocess_iamge(image_path, class_name):
    preprocessed_image = preprocess_image_for_prediction(image_path)
    if preprocessed_image is not None:
        predictions = model.predict(preprocessed_image)
        predicted_class = class_names[np.argmax(predictions)]

        print(f'Image: {filename}, Predicted class: {predicted_class}')

# Call the function to process the folder
process_image_folder(image_folder, class_name)
# preprocess_image(image_path, class_name
# or
import joblib
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
# Set the path to the image
image_path = "D:/PycharmProjects/mastersProject/ja - Copy.png"
model = joblib.load("D:/PycharmProjects/mst/trained_model_hue_hsvyuv5.pkl")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to crop face, resize, and preprocess for model prediction
def preprocess_image_for_prediction(image_path, target_size=(150, 150)):
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Assuming the first detected face
        face = img[y:y + h, x:x + w]

        # Resize the face to the target size
        face = cv2.resize(face, target_size)

        # Preprocess the image for model prediction
        face = face.astype(np.float32) / 255.0  # Rescale pixel values to [0, 1]
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        face = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2YUV)
        # cv2.imwrite(save_path, face)
        plt.imshow(face)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('D:/PycharmProjects/mastersProject/ja - Copy.png', bbox_inches='tight', pad_inches=0)
        # Show the image
        plt.close()
        return face
    else:
        return None

# Function to process a single image
def process_single_image(image_path):

    preprocessed_image = cv2.imread(image_path)
    predictions = model.predict(preprocessed_image)
    class_names = ['Cool', 'Warm']  # Replace with your class names
    predicted_class = class_names[np.argmax(predictions)]

    print(f'Image: {image_path}, Predicted class: {predicted_class}')

# Call the function to process the single image
preprocess_image_for_prediction(image_path)
process_single_image(image_path)

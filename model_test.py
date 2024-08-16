import os
import cv2
import numpy as np
import datatset_preparation as dp
import pickle
import json
import matplotlib.pyplot as plt

config = dp.get_config()

# Load the face detection model (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the three models and their corresponding class names
models = []
class_names_list = []

# Model 1: Load and append to models and class_names_list
with open('D:/PycharmProjects/mst/Hue_MODEL.pkl', 'rb') as file:
    models.append(pickle.load(file))
class_names_list.append(config['Hue'])

# Model 2: Load and append to models and class_names_list
with open('D:/PycharmProjects/mst/Saturation_MODEL.pkl', 'rb') as file:
    models.append(pickle.load(file))
class_names_list.append(config['Saturation'])

# Model 3: Load and append to models and class_names_list
with open('D:/PycharmProjects/mst/Value_MODEL.pkl', 'rb') as file:
    models.append(pickle.load(file))
class_names_list.append(config['Value'])

# Define suffixes for each model
suffixes = ['_hue', '_sat', '_val']

def crop_face(image_path):
    """
    Detect and crop the face from an image.

    Parameters:
    - image_path (str): The path to the input image.

    Returns:
    - face (numpy.ndarray or None): The cropped face image if detected, otherwise None.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Assuming the first detected face
        face = img[y:y + h, x:x + w]
        return face
    else:
        return None

def preprocess_image(image_path, model_index, destination_path=None):
    """
    Preprocess an image for a specific model and save the preprocessed image if a destination path is provided.

    Parameters:
    - image_path (str): The path to the input image.
    - model_index (int): The index of the model to preprocess for.
    - destination_path (str, optional): The path to save the preprocessed image. Default is None.

    Returns:
    - face (numpy.ndarray): The preprocessed image ready for model prediction.
    """
    # Ensure the destination path exists
    if destination_path and not os.path.exists(destination_path):
        os.makedirs(destination_path)
        print(f"Created destination directory: {destination_path}")

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(destination_path, f"{base_name}{suffixes[model_index]}.png")

    # Check if the preprocessed image already exists
    if destination_path and os.path.exists(save_path):
        print(f"Preprocessed image found: {save_path}")
        preprocessed_image = cv2.imread(save_path)
        preprocessed_image = cv2.resize(preprocessed_image, (150, 150))
        preprocessed_image = preprocessed_image / 255.0
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        return preprocessed_image

    print(f"Processing image for model {model_index}: {image_path}")

    # Crop the face from the image
    face = crop_face(image_path)
    if face is None:
        raise ValueError("No face detected in the image.")

    # Model-specific preprocessing
    if model_index == 0:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2YUV)
    elif model_index == 1:
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.convertScaleAbs(s, alpha=1.5, beta=0)
        enhanced_hsv = cv2.merge([h, s, v])
        face = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    elif model_index == 2:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = np.expand_dims(face, axis=-1)  # Shape: (height, width, 1)
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)  # Convert grayscale back to 3 channels for consistency

    # Resize the face image to the expected input size
    face = cv2.resize(face, (150, 150))
    face = face / 255.0  # Normalize the image
    face = np.expand_dims(face, axis=0)  # Add batch dimension

    # Save the preprocessed image
    if destination_path:
        print(f"Saving preprocessed image at: {save_path}")
        if model_index == 0:
            plt.imshow(face[0])
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        else:
            cv2.imwrite(save_path, (face[0] * 255).astype(np.uint8))

    return face

def process_image_folder(image_folder, output_json, destination_path=None):
    """
    Process all images in a folder using predefined models and save the predictions to a JSON file.

    Parameters:
    - image_folder (str): The folder containing images to process.
    - output_json (str): The path to save the JSON output containing predictions.
    - destination_path (str, optional): The path to save preprocessed images. Default is None.

    Returns:
    - None
    """
    results = []

    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, image_file)
            try:
                image_result = {'file_name': image_file, 'predictions': []}

                for i, model in enumerate(models):
                    preprocessed_image = preprocess_image(image_path, i, destination_path)
                    predictions = model.predict(preprocessed_image)
                    predicted_index = np.argmax(predictions)
                    predicted_class = class_names_list[i][predicted_index]
                    confidence_score = predictions[0][predicted_index]

                    image_result['predictions'].append({
                        'model': f'model_{i + 1}',
                        'predicted_class': predicted_class,
                        'confidence_score': float(confidence_score)
                    })

                results.append(image_result)
            except ValueError as e:
                print(f"Skipping {image_file}: {e}")

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)

# Example usage
process_image_folder(config['TEST_DATA'], config['JSON_OUTPUT'], 'D:/PycharmProjects/mst/dest_test')

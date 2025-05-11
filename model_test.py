import os
import cv2
import numpy as np
import dataset_preparation as dp
import pickle
import json
import matplotlib.pyplot as plt

config = dp.get_config()

# Load the face detection model (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load models and class names
models = []
class_names_list = []

model_files = [
    ('D:/PycharmProjects/MST/Models/Hue_MODEL.pkl', config['Hue']),
    ('D:/PycharmProjects/MST/Models/Saturation_MODEL.pkl', config['Saturation']),
    ('D:/PycharmProjects/MST/Models/Value_MODEL.pkl', config['Value'])
]

suffixes = ['_hue', '_sat', '_val']

for model_file, class_names in model_files:
    with open(model_file, 'rb') as file:
        models.append(pickle.load(file))
    class_names_list.append(class_names)


def crop_face(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Assuming the first detected face
        face = img[y:y + h, x:x + w]
        return face
    return None


def preprocess_image(image_path, model_index, destination_path=None):
    if destination_path and not os.path.exists(destination_path):
        os.makedirs(destination_path)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(destination_path, f"{base_name}{suffixes[model_index]}.png") if destination_path else None

    if save_path and os.path.exists(save_path):
        preprocessed_image = cv2.imread(save_path)
        preprocessed_image = cv2.resize(preprocessed_image, (150, 150)) / 255.0
        return np.expand_dims(preprocessed_image, axis=0)

    face = cv2.imread(image_path)
    # face = crop_face(image_path)
    # if face is None:
    #     raise ValueError("No face detected in the image.")

    if model_index == 0:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2YUV)
    elif model_index == 1:
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.convertScaleAbs(s, alpha=1.5, beta=0)
        face = cv2.merge([h, s, v])
        face = cv2.cvtColor(face, cv2.COLOR_HSV2BGR)
    elif model_index == 2:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = np.expand_dims(face, axis=-1)
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

    face = cv2.resize(face, (150, 150)) / 255.0
    face = np.expand_dims(face, axis=0)

    if destination_path:
        if model_index == 0:
            plt.imshow(face[0])
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        else:
            cv2.imwrite(save_path, (face[0] * 255).astype(np.uint8))

    return face


def process_image_folder(image_folder, output_json, destination_path=None):
    # Load existing results if file exists
    if os.path.exists(output_json):
        with open(output_json, 'r') as f:
            results = json.load(f)
    else:
        results = []

    # Get already processed filenames to avoid duplication
    processed_files = {entry['file_name'] for entry in results}

    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')) and image_file not in processed_files:
            image_path = os.path.join(image_folder, image_file)
            base_name = os.path.splitext(image_file)[0]

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

                # Save updated results after every image
                with open(output_json, 'w') as f:
                    json.dump(results, f, indent=4)

                print(f"Processed: {image_file}")

            except ValueError as e:
                print(f"Skipping {image_file}: {e}")
        else:
            print(f"Skipping already processed image: {image_file}")



# Example usage
process_image_folder(config['TEST_DATA'], config['JSON_OUTPUT'], 'D:/PycharmProjects/MST/Train_destination_2')

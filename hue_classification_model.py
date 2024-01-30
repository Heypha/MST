import math
import pandas as pd
from tensorflow import keras
import cv2
import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("D:/PycharmProjects/mP2/hue_dataset.csv")
image_files = data['Image Path']
labels = data['Label']

# Split the data into training, validation, and test sets
train_files, test_files, train_labels, test_labels = train_test_split(image_files, labels, test_size=0.3, random_state=42)
train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size=0.15, random_state=42)

# Destination directory
destination_directory = 'D:/PycharmProjects/mP2/destination_hue'

def remove_images_with_suffix(root_dir):
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith('.png'):
                file_path = os.path.join(root, filename)
                os.remove(file_path)
sets=['/train_set','/test_set','/val_set', '/train_set_cropped','/test_set_cropped','/val_set_cropped']
for set in sets:
    remove_images_with_suffix(destination_directory + set)

def folder_create(old_dir,sufix):
    class_names = ['Cool', 'Warm']

    # Folder Create Segment
    new_dir = old_dir + sufix
    os.makedirs(new_dir, exist_ok=True)
    for class_name in class_names:
        class_train_dir = os.path.join(new_dir, class_name)
        os.makedirs(class_train_dir, exist_ok=True)

    return new_dir

train_dir = folder_create(destination_directory, '/train_set')
test_dir = folder_create(destination_directory, '/test_set')
val_dir = folder_create(destination_directory, '/val_set')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_face(image_path):
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

def save_photo(destination_path,photo_paths_list,classificator,ifFace):
    if ifFace:
        for image_path, class_name in zip(photo_paths_list, classificator):
            face = crop_face(image_path)
            if face is not None:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
                face = cv2.cvtColor(face, cv2.COLOR_BGR2YUV)
                save_path = os.path.join(destination_path, class_name, os.path.basename(image_path))
                # cv2.imwrite(save_path, face)
                plt.imshow(face)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                # Show the image
                plt.close()
    else:
        for image_path, class_name in zip(photo_paths_list, classificator):
            save_path = os.path.join(destination_path, class_name, os.path.basename(image_path))
            shutil.copy(image_path, save_path)


sufixes = ["train", "test", "val"]

for sufix in sufixes:
    dir = sufix + "_dir"
    files = sufix +"_files"
    labels = sufix + "_labels"
    save_photo(locals()[dir], locals()[files], locals()[labels], False)



cropped_train_dir = folder_create(train_dir, '_cropped')
cropped_test_dir = folder_create(test_dir, '_cropped')
cropped_val_dir = folder_create(val_dir, '_cropped')


save_photo(cropped_train_dir, train_files, train_labels, True)
save_photo(cropped_test_dir, test_files, test_labels, True)
save_photo(cropped_val_dir, val_files, val_labels, True)

# cropped_train_dir = "D:/PycharmProjects/mP2/destination_hue/train_set_cropped"
# cropped_test_dir= "D:/PycharmProjects/mP2/destination_hue/test_set_cropped"
# cropped_val_dir= "D:/PycharmProjects/mP2/destination_hue/val_set_cropped"
# cropped_train_dir+"/Warm"
# os.listdir(cropped_train_dir+"/Warm")

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 120 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        cropped_train_dir,  # This is the source directory for training images
        classes = ['Cool', 'Warm'],
        target_size=(150,150),  # All images will be resized to 200x200
        batch_size=32,
        # Use binary labels
        class_mode='categorical',  # For classification tasks
    shuffle=True)
# Flow validation images in batches of 19 using valid_datagen generator
test_generator = test_datagen.flow_from_directory(
        cropped_test_dir,  # This is the source directory for training images
        classes = ['Cool', 'Warm'],
        target_size=(150,150),  # All images will be resized to 200x200
        batch_size=32,
        # Use binary labels
        class_mode='categorical',
        shuffle=False)
# Flow validation images in batches of 19 using valid_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        cropped_val_dir,  # This is the source directory for training images
        classes = ['Cool', 'Warm'],
        target_size=(150,150),  # All images will be resized to 200x200
        batch_size=32,
        # Use binary labels
        class_mode='categorical',
        shuffle=True)


model = keras.Sequential()

# Convolutional layer and maxpool layer 1
model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 2
model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 3
model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 4
model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

# This layer flattens the resulting image array to 1D array
model.add(keras.layers.Flatten())

# Hidden layer with 512 neurons and Rectified Linear Unit activation function
model.add(keras.layers.Dense(512,activation='relu'))

# Output layer with single neuron which gives 0 for Cat or 1 for Dog
#Here we use sigmoid activation function which makes our model output to lie between 0 and 1
model.add(keras.layers.Dense(2,activation='sigmoid'))
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_generator,
      steps_per_epoch=len(train_generator),
      epochs=25,
      validation_data = validation_generator,
      validation_steps=len(validation_generator))

loss, accuracy = model.evaluate(validation_generator)
print("Test loss:", loss)
print("Test accuracy:", accuracy)




import pickle

# Save data
with open('trained_model_hue_hsvyuv2.pkl', 'wb') as file:
    pickle.dump(model, file)
with open('trained_model_hue_hsvyuv2.pkl', 'rb') as file:
    model = pickle.load(file)

test_predictions = model.predict(test_generator)
test_class_indices = test_generator.class_indices
test_class_labels = {v: k for k, v in test_class_indices.items()}
predicted_labels = [test_class_labels[i] for i in test_predictions.argmax(axis=1)]
image_paths = test_generator.filepaths
class_names = labels[:len(test_class_indices)]
test_confidence_scores = [test_predictions[i].max() for i in range(len(test_predictions))]
results_df = pd.DataFrame({'File': image_paths, 'Predicted_Label': predicted_labels, 'Confidence_Score': test_confidence_scores})
results_json = results_df.to_json(orient='records')
pre_label = test_predictions.argmax(axis=1)
true_labels = test_generator.classes
precision = precision_score(true_labels, pre_label)
print(precision)
with open('test_results_hue_hsvyuv2.json', 'w') as file:
    file.write(results_json)

del model

######################
## Dodatkowo niech wycina background ze zdjecia
#Proposition: To do % about groups :Hue(Warm/Cool), Value(Soft/Bright), Chroma(Light/Dark) and then based on the % classify them to seasons
######################

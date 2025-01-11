# # import os
# # import shutil
# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # import cv2
# #
# # data = pd.read_csv("D:/PycharmProjects/mP2/image_dataset.csv")
# # image_files = data['Image Path']
# # labels = data['Label']
# #
# # # Split the data into training, validation, and test sets
# # train_files, test_files, train_labels, test_labels = train_test_split(image_files, labels, test_size=0.3, random_state=42)
# # train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size=0.15, random_state=42)
# #
# # # Destination directory
# # destination_directory = 'D:/PycharmProjects/mst'
# #
# # def folder_create(old_dir,sufix):
# #     class_names = ["BrightSpring", "BrightWinter", "DarkAutumn", "DarkWinter", "LightSpring",
# #                   "LightSummer", "SoftAutumn", "SoftSummer", "TrueAutumn", "TrueSpring",
# #                   "TrueSummer", "TrueWinter"]
# #
# #     # Folder Create Segment
# #     new_dir = old_dir + sufix
# #     os.makedirs(new_dir, exist_ok=True)
# #     for class_name in class_names:
# #         class_train_dir = os.path.join(new_dir, class_name)
# #         os.makedirs(class_train_dir, exist_ok=True)
# #
# #     return new_dir
# #
# # train_dir = folder_create(destination_directory, '/train_set')
# # test_dir = folder_create(destination_directory, '/test_set')
# # val_dir = folder_create(destination_directory, '/val_set')
# #
# # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# #
# # def crop_face(image_path):
# #     img = cv2.imread(image_path)
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #
# #     # Detect faces in the image
# #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
# #
# #     if len(faces) > 0:
# #         (x, y, w, h) = faces[0]  # Assuming the first detected face
# #         face = img[y:y + h, x:x + w]
# #         return face
# #     else:
# #         return None
# #
# # def save_photo(destination_path,photo_paths_list,classificator,ifFace):
# #     if ifFace:
# #         for image_path, class_name in zip(photo_paths_list, classificator):
# #             face = crop_face(image_path)
# #             if face is not None:
# #                 save_path = os.path.join(destination_path, class_name, os.path.basename(image_path))
# #                 cv2.imwrite(save_path, face)
# #     else:
# #         for image_path, class_name in zip(photo_paths_list, classificator):
# #             save_path = os.path.join(destination_path, class_name, os.path.basename(image_path))
# #             shutil.copy(image_path, save_path)
# #
# #
# # sufixes = ["train", "test", "val"]
# #
# # for sufix in sufixes:
# #     dir = sufix + "_dir"
# #     files = sufix +"_files"
# #     labels = sufix + "_labels"
# #     save_photo(locals()[dir], locals()[files], locals()[labels], False)
# import PIL
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import os
#
# from PIL import ImageOps, ImageEnhance
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,f1_score,recall_score
# # image = cv2.imread("D:/PycharmProjects/mP2/destination_hue/test_set_cropped/Warm/Warm_015.png")
# # image = cv2.imread("D:/PycharmProjects/mP2/destination_hue/test_set_cropped/Cool/Cool_037.png")
# # image = cv2.imread("D:/PycharmProjects/mP2/destination_hue/test_set_cropped/Warm/Warm_027.png")
# # Convert to CIELab color space
# folder_path = "D:/PycharmProjects/mP2/Hue_Data"
#
#
#
# # # Define thresholds for warm and cool undertones
# # warm_threshold = 174  # Adjust as needed
# # cool_threshold = -100  # Adjust as needed
# #
# # # Set a threshold for the dominant undertone
# # threshold_ratio = 1.5  # Adjust as needed
# # true_labels = []  # Store the true labels
# # predicted_labels = []  # Store the predicted labels
# # for filename in os.listdir(folder_path):
# #     if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
# #         # Extract the true label from the first four characters of the file name
# #         true_label = filename[:4]
# #
# #         # Load the image
# #         image_path = os.path.join(folder_path, filename)
# #         image = cv2.imread(image_path)
# #
# #         # Convert to CIELab color space
# #         lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
# #
# #         # Create binary masks for warm and cool undertones
# #         warm_mask = lab_image[:, :, 1] > warm_threshold
# #         cool_mask = lab_image[:, :, 1] < cool_threshold
# #
# #         # Convert the masks to 8-bit data type
# #         warm_mask = np.uint8(warm_mask)
# #         cool_mask = np.uint8(cool_mask)
# #
# #         # Count the number of warm and cool pixels
# #         num_warm_pixels = cv2.countNonZero(warm_mask)
# #         num_cool_pixels = cv2.countNonZero(cool_mask)
# #
# #         # Determine the dominant undertone
# #         if num_warm_pixels > num_cool_pixels * threshold_ratio:
# #             undertone = "Warm"
# #         else:
# #             undertone = "Cool"
# #
# #         # Print the result along with the file name
# #         print(f"File: {filename}, Undertone: {undertone}")
# #         true_labels.append(true_label)
# #         predicted_labels.append(undertone)
# #
# # # Create a confusion matrix and calculate accuracy
# # confusion = confusion_matrix(true_labels, predicted_labels)
# # accuracy = accuracy_score(true_labels, predicted_labels)
#
# # print("Confusion Matrix:")
# # print(confusion)
# # print(f"Accuracy: {accuracy * 100:.2f}%")
#
# #params:
# # Confusion Matrix:
# # [[22 40]
# #  [14 44]]
# # Accuracy: 55.00%
# # warm_threshold = 168
# # cool_threshold = -10
# # threshold_ratio = 1.5
#
# # Confusion Matrix:
# # [[34 28]
# #  [25 33]]
# # Accuracy: 55.83%
# # warm_threshold = 174
# # cool_threshold = -100
# # threshold_ratio = 1.5
#
# import cv2
# import matplotlib.pyplot as plt
# # Wa_004.png Co_011.png
# # img = cv2.imread("D:/PycharmProjects/mP2/destination_hue/test_set_cropped/Cool/Cool_037.png")
# # img = cv2.imread("D:/PycharmProjects/mP2/Hue_Data/Wa_004.png")
# # plt.imshow(img)
# # plt.show()
# #
# # #  Luv,YUV, HSV, LAB, HLS
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# #
# # # create a copy of the original image
# # img_rgb = img.copy()
# #
# # # extract blue channel of the rgb image
# # b = img_rgb[:,:,2]
# #
# # # increase the pixel values by 100
# # b = b + 100
# #
# # # if pixel values become > 255, subtract 255
# # cond = b[:, :] > 255
# # b[cond] = b[cond] - 255
# #
# # # assign the modified channel to image
# # img_rgb[:,:,2] = b
# #
# # plt.imshow(img_rgb)
# # plt.show()
#
# #
# # def convert_image_to_yuv(input_folder, output_folder):
# #     # Create the output folder if it doesn't exist
# #     os.makedirs(output_folder, exist_ok=True)
# #
# #     # List all files in the input folder
# #     image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
# #
# #     for image_file in image_files:
# #         # Read the image
# #         image_path = os.path.join(input_folder, image_file)
# #         image = cv2.imread(image_path)
# #
# #         if image is not None:
# #             # Convert the image to YUV
# #             yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
# #
# #             # Save the YUV image to the output folder
# #             output_path = os.path.join(output_folder, image_file)
# #             cv2.imwrite(output_path, yuv_image)
# #
# # if __name__ == "__main__":
# #     input_folder = "D:/PycharmProjects/mP2/Hue_Data" # Replace with the path to your input folder
# #     output_folder = "D:/PycharmProjects/mP2/HueS"  # Replace with the path to your output folder
# #
# #     convert_image_to_yuv(input_folder, output_folder)
#
# import cv2
# import os
# import shutil
# import pandas as pd
# # Wa_008 Co_011 Co_013 Wa_004 Co_001 Wa_014
# # data = pd.read_csv("D:/PycharmProjects/mP2/hue_dataset.csv")
# # image_files = data['Image Path']
# # labels = data['Label']
# # destination_path = "D:/PycharmProjects/mP2/Hues"
# # image_path = "D:/PycharmProjects/mP2/Hue_Data/Co_011.png"
# def crop_face(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Detect faces in the image
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     if len(faces) > 0:
#         (x, y, w, h) = faces[0]  # Assuming the first detected face
#         face = img[y:y + h, x:x + w]
#         return face
#     else:
#         return None
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# # for image_path, class_name in zip(image_files, labels):
# #     face = crop_face(image_path)
# #     if face is not None:
# #         face = cv2.cvtColor(face, cv2.COLOR_BGR2Luv)
# #         face = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
# #         save_path = os.path.join(destination_path, class_name, os.path.basename(image_path))
# #         cv2.imwrite(save_path, face)
#
# # face = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
# # face = cv2.cvtColor(face, cv2.COLOR_BGR2YUV)
# import numpy as np
# import cv2
# import numpy as np
# # Define augmentation parameters
# # angle = 45  # Rotation angle in degrees
# # scale = 1.5  # Scaling factor
# # tx, ty = 20, 30  # Translation in x and y directions
# # rotation_matrix = cv2.getRotationMatrix2D((face.shape[1] / 2, face.shape[0] / 2), angle, scale)
# # rotated_image = cv2.warpAffine(face, rotation_matrix, (face.shape[1], face.shape[0]))
# #
# from PIL import Image
# # # 2. Translation
# # translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
# # Br_005 Br_020 Br_015 Mu_008 Mu_013 Mu_014
# # translated_image = cv2.warpAffine(rotated_image, translation_matrix, (face.shape[1], face.shape[0]))
# # image_path1 = "D:/PycharmProjects/mP2/Saturation_Data/Br_030.png"
# # image_path2 = "D:/PycharmProjects/mP2/Saturation_Data/Mu_049.png"
# import cv2
# import numpy as np
# image_path1 = "D:/PycharmProjects/mP2/Saturation_Data/Br_035.png"
# image_path2 = "D:/PycharmProjects/mP2/Saturation_Data/Mu_054.png"
#
# # Load the image
# image1 = crop_face(image_path1)
# image2 = crop_face(image_path2)
# # solution 1
# # img1 = cv2.convertScaleAbs(image1, beta=50)
# # img2 = cv2.convertScaleAbs(img1, beta=-40)
# # img3 = cv2.convertScaleAbs(image2, beta=50)
# # img4 = cv2.convertScaleAbs(img3, beta=-40)
# # solution 2
# # img = np.array(image1, dtype=np.float64)
# # img = cv2.transform(img, np.matrix([[0.272,0.534,0.131], [0.349,0.686,0.168],[0.393,0.769,0.189]]))
# # img[np.where(img>255)] = 255
# # img = np.array(img, dtype=np.uint8)
# #
# # cv2.imshow("Bright", img4)
# # cv2.imshow("Muted", img2)
# #
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# # cv2.imshow("Bright", hsv)
# # cv2.imshow("Muted Image", hsv2)
#
# img = PIL.Image.open(image_path1)
# converter = PIL.ImageEnhance.Color(img)
# img2 = converter.enhance(1.3)
# plt.imshow(img2)
# plt.axis('off')
# plt.show()
# # img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
# # img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# # img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
# # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
# # plt.imshow(img)
# # plt.imshow(img2)
# # plt.show()
# # img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# # img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
# # img1 = cv2.convertScaleAbs(img1, beta=50)
# # img1 = cv2.convertScaleAbs(img1, beta=-30)
# # img2 = cv2.convertScaleAbs(img2, beta=50)
# # img2 = cv2.convertScaleAbs(img2, beta=-30)
# # cv2.imshow("Bright", img1)
# # cv2.imshow("Muted", img2)
# #
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# # # Duplicate the image
# # darkened_image = face.copy()
# #
# # # Define a factor to adjust the shadow intensity
# # shadow_intensity = 0.4  # Adjust this value to control the shadow depth
# #
# # # Darken the image by reducing the brightness
# # darkened_image = darkened_image * shadow_intensity
# #
# # # Ensure pixel values are within the valid range (0-255)
# # darkened_image = np.clip(darkened_image, 0, 255).astype(np.uint8)
# # # Display the darkened image
# # cv2.imshow('Normal Image', face)
# # cv2.imshow('Darkened Image', darkened_image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
#
# # plt.imshow(darkened_image)
# # plt.show()
#
#
# # Br_005 Br_020 Br_015 Mu_008 Mu_013 Mu_014
# # translated_image = cv2.warpAffine(rotated_image, translation_matrix, (face.shape[1], face.shape[0]))
#
#
# # # Image paths
# # image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
# #
# # # Load images
# # images = load_images(image_paths)
#
#
# # # Classify images based on saturation
# # for i, image in enumerate(face):
# #     saturation = calculate_saturation(image)
# #     if saturation > saturation_threshold:
# #         print(f"Image {i+1} is high saturation.")
# #     else:
# #         print(f"Image {i+1} is low saturation.")
#
#
#
#
#
# # img = Image.open(image_path)
# # img_data = img.getdata()
# #
# # lst=[]
# # for i in img_data:
# #
# #     lst.append(i[0]*0.06+i[1]*0.5+i[2]*0.02) ### Rec. 609-7 weights
# #     # lst.append(i[0]*0.2125+i[1]*0.7174+i[2]*0.0721) ### Rec. 709-6 weights
# #
# # new_img = Image.new("L", img.size)
# # new_img.putdata(lst)
# # new_img.show()
# # new_img.save(output_file_path)
# # Display the image using Matplotlib
# # Display the image using Matplotlib
# # plt.imshow(gray_image)
# # plt.show()
# # plt.axis('off')
# # plt.tight_layout()
# # plt.savefig('D:/PycharmProjects/mP2/Hues/output_image.jpg', bbox_inches='tight', pad_inches=0)
#
# # Show the image
# # plt.close()
# # Save the image in the RGB color space
# # cv2.imwrite('D:/PycharmProjects/mP2/Hues/output_image4.jpg', face)
#
# # Br_012 Br_020 Br_015 Mu_008 Mu_013 Mu_014
# # image_path = "D:/PycharmProjects/mP2/Saturation_Data/Br_015.png"
# # face, x, y, w, h = crop_face(image_path)
# # gray_image = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
# #
# # plt.imshow(gray_image)
# # plt.show()
#
# import os
#
# # Specify the folder path
# folder_path = 'D:/PycharmProjects/MST/Seasons/BrightSpring'
#
# # Iterate through all files in the folder
# for filename in os.listdir(folder_path):
#     # Check if it's a file (not a directory)
#     if os.path.isfile(os.path.join(folder_path, filename)):
#         # Extract the file extension
#         _, extension = os.path.splitext(filename)
#         print(extension)
import os
# def rename_images_in_folder(root_folder):
#     for root, dirs, files in os.walk(root_folder):
#         for dir_name in dirs:
#             folder_path = os.path.join(root, dir_name)
#             folder_name = os.path.basename(folder_path)
#             words = folder_name.split()
#             first_letters = ''.join([word[:2] for word in words])
#             image_files = [f for f in os.listdir(folder_path) if
#                            f.lower().endswith(('.com', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.jfif', '.webp'))]
#
#             for i, image_file in enumerate(image_files, start=1):
#                 ext = os.path.splitext(image_file)[1]
#                 new_name = f"{first_letters}_{i:03d}{ext}"
#                 new_path = os.path.join(folder_path, new_name)
#                 os.rename(os.path.join(folder_path, image_file), new_path)
#                 print(f"Renamed: {image_file} to {new_name}")
#
# rename_images_in_folder("D:/PycharmProjects/MST/Data/Seasons/Seasons")
import os
import shutil
import csv
import json
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

dest = "D:/PycharmProjects/MST/Data/Test_Data"
for root, dirs, files in os.walk(dest):
    for filename in files:
        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".jfif", ".webp", ".com")):
            image_path = os.path.join(root, filename)
            img = Image.open(image_path)
            new_filename = os.path.splitext(filename)[0] + ".png"
            img.save(os.path.join(dest, new_filename))
            print(f"Converted: {filename} to {new_filename}")
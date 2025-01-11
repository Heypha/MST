import os
import shutil
import csv
import json
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

def get_config():
    """
    Retrieve configuration settings from a JSON file.

    Returns:
    - dict: A dictionary containing the configuration settings.
    """

    file_path = os.path.dirname(os.path.abspath(__file__))
    config_file_name = "conifg.json"
    config_file = os.path.join(file_path, config_file_name)

    with open(config_file) as config_file:
        return json.load(config_file)

config = get_config()


class ImageProcessor:
    def __init__(self):
        self.model = config['MODEL']
        self.root_folder = os.path.join(config['ROOT_FOLDER'], config['MODEL'])
        self.destination_folder = os.path.join(config['ROOT_FOLDER'], config['MODEL'] + '_Data')
        self.csv_output_file = os.path.join(config['ROOT_FOLDER'], config['MODEL'] + '_dataset.csv')

    @staticmethod
    def rename_images_in_folder(folder_path):
        folder_name = os.path.basename(folder_path)
        words = folder_name.split()
        first_letters = ''.join([word[:2] for word in words])
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.com', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.jfif'))]

        for i, image_file in enumerate(image_files, start=1):
            ext = os.path.splitext(image_file)[1]
            new_name = f"{first_letters}_{i:03d}{ext}"
            new_path = os.path.join(folder_path, new_name)
            os.rename(os.path.join(folder_path, image_file), new_path)
            print(f"Renamed: {image_file} to {new_name}")

    def train_images_preprocessing(self, face, save_path):
        if self.model == 'Saturation':
            hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = cv2.convertScaleAbs(s, alpha=1.5, beta=0)
            enhanced_hsv = cv2.merge([h, s, v])
            face = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(save_path, face)
        elif self.model == 'Value':
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(save_path, face)
        elif self.model == 'Hue':
            face = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2YUV)
            plt.imshow(face)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            cv2.imwrite(save_path, face)

    def convert_images_to_png(self):
        if os.path.exists(self.destination_folder):
            shutil.rmtree(self.destination_folder)
        os.makedirs(self.destination_folder)

        for root, dirs, files in os.walk(self.root_folder):
            for filename in files:
                if filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".jfif", ".webp", ".com")):
                    image_path = os.path.join(root, filename)
                    img = Image.open(image_path)
                    new_filename = os.path.splitext(filename)[0] + ".png"
                    img.save(os.path.join(self.destination_folder, new_filename))
                    print(f"Converted: {filename} to {new_filename}")

    def create_csv_file(self, fileList, labels):
        if os.path.exists(self.csv_output_file):
            os.remove(self.csv_output_file)

        with open(self.csv_output_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Image Path', 'Label'])
            for image_path, label in zip(fileList, labels):
                csvwriter.writerow([image_path, label])

    def create_file_list(self, format='.png'):
        fileList, labels, names = [], [], []
        if config['MODEL'] == 'Saturation':
            keywords = {
                "Br": "Bright",
                "Mu": "Muted"
            }
        elif config['MODEL'] == 'Hue':
            keywords = {
                "Wa": "Warm",
                "Co": "Cool"
            }
        elif config['MODEL'] == 'Value':
            keywords = {
                "Da": "Dark",
                "Li": "Light"
            }
        elif config['MODEL'] == 'Seasons':
            keywords = {
                "Wi": "Winter",
                "Sp": "Spring",
                "Su": "Summer",
                "Au": "Autumn"
            }
        else:
            keywords = {
                "BrWi": "Bright Winter",
                "TrWi": "True Winter",
                "DaWi": "Dark Winter",
                "BrSp": "Bright Spring",
                "TrSp": "True Spring",
                "LiSp": "Light Spring",
                "LiSu": "Light Summer",
                "TrSu": "True Summer",
                "SoSu": "Soft Summer",
                "SoAu": "Soft Autumn",
                "TrAu": "True Autumn",
                "DaAu": "Dark Autumn"
            }

        for root, dirs, files in os.walk(self.destination_folder, topdown=True):
            for name in files:
                if name.endswith(format):
                    fullName = os.path.join(root, name)
                    fileList.append(fullName)
                    label = None
                    for keyword, season in keywords.items():
                        if keyword in name:
                            label = season
                            break
                    if label is not None:
                        labels.append(label)
                    else:
                        labels.append("Unknown")
                    names.append(name)
                    print(fileList, label)

        self.create_csv_file(fileList, labels)
        print(fileList)

    def process_all_folders(self):
        for root, dirs, files in os.walk(self.root_folder):
            for dir_name in dirs:
                folder_path = os.path.join(root, dir_name)
                self.rename_images_in_folder(folder_path)


if __name__ == "__main__":
    processor = ImageProcessor()
    processor.process_all_folders()
    processor.convert_images_to_png()
    processor.create_file_list()

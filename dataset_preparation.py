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
        self.root_folder = os.path.join(config['ROOT_FOLDER'], 'Seasons')
        self.season_folder =os.path.join(config['ROOT_FOLDER'], 'Seasons')
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
            # face = cv2.medianBlur(face, 15)
            # hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            # h, s, v = cv2.split(hsv)
            # s = cv2.convertScaleAbs(s, alpha=1.5, beta=0)
            # enhanced_hsv = cv2.merge([h, s, v])
            # face = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
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

        # Mapa sezonów do kategorii modelu
        model_map = {
            'Saturation': {
                'Bright': ['Bright Spring', 'Bright Winter'],
                'Muted': ['Soft Autumn', 'Soft Summer']
            },
            'Hue': {
                'Warm': ['True Spring', 'True Autumn'],
                'Cool': ['True Winter', 'True Summer']
            },
            'Value': {
                'Dark': ['Dark Autumn', 'Dark Winter'],
                'Light': ['Light Spring', 'Light Summer']
            },
            'Seasons': {
                'Winter': ['Bright Winter', 'True Winter', 'Dark Winter'],
                'Spring': ['Bright Spring', 'True Spring', 'Light Spring'],
                'Summer': ['Light Summer', 'True Summer', 'Soft Summer'],
                'Autumn': ['Soft Autumn', 'True Autumn', 'Dark Autumn']
            }
        }

        for season_dir in os.listdir(self.root_folder):
            season_path = os.path.join(self.root_folder, season_dir)
            if not os.path.isdir(season_path):
                continue

            for file in os.listdir(season_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.jfif', '.webp', '.com')):
                    file_path = os.path.join(season_path, file)
                    img = cv2.imread(file_path)

                    # Znajdź kategorię docelową
                    for category, folders in model_map[self.model].items():
                        if season_dir in folders:
                            dest_cat_path = os.path.join(self.destination_folder, category)
                            os.makedirs(dest_cat_path, exist_ok=True)
                            save_path = os.path.join(dest_cat_path, file)
                            cv2.imwrite(save_path, img)
                            print(f"Saved {file} to {category}")

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
                "So": "Muted"
            }
        elif config['MODEL'] == 'Hue':
            keywords = {
                "TrSp": "Warm",
                "TrAu": "Warm",
                "TrSu": "Cool",
                "TrWi": "Cool"
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


    def convert_all_to_png_first(self):
        source_root = self.season_folder
        output_root = self.root_folder

        if os.path.exists(output_root):
            shutil.rmtree(output_root)
        os.makedirs(output_root, exist_ok=True)

        for season_dir in os.listdir(source_root):
            season_path = os.path.join(source_root, season_dir)
            if not os.path.isdir(season_path):
                continue

            output_season_dir = os.path.join(output_root, season_dir)
            os.makedirs(output_season_dir, exist_ok=True)

            for file in os.listdir(season_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.bmp', '.gif', '.jfif', '.webp', '.com', '.png')):
                    try:
                        input_path = os.path.join(season_path, file)
                        img = Image.open(input_path).convert("RGB")  # Ensure proper format

                        filename_wo_ext = os.path.splitext(file)[0]
                        output_path = os.path.join(output_season_dir, filename_wo_ext + ".png")
                        img.save(output_path, format="PNG")
                        print(f"Converted {file} to PNG in {season_dir}")
                    except Exception as e:
                        print(f"Failed to convert {file} in {season_dir}: {e}")


if __name__ == "__main__":
    processor = ImageProcessor()
    # processor.convert_all_to_png_first()
    processor.convert_images_to_png()
    processor.create_file_list()

import os
import shutil
# I
# Function to rename images in a folder
def rename_images_in_folder(folder_path):
    # Get the two-word folder name
    folder_name = os.path.basename(folder_path)
    words = folder_name.split()


    # Get the first letter of each word
    first_letters = ''.join([word[:2] for word in words])

    # List image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.jfif'))]

    # Rename images
    for i, image_file in enumerate(image_files, start=1):
        ext = os.path.splitext(image_file)[1]
        new_name = f"{first_letters}_{i:03d}{ext}"
        new_path = os.path.join(folder_path, new_name)
        os.rename(os.path.join(folder_path, image_file), os.path.join(folder_path, new_name))

        # Optionally, resize the image
        # resize_image(new_path)

        print(f"Renamed: {image_file} to {new_name}")

# Function to resize the image (optional)
def resize_image(image_path, max_size=(1024, 1024)):
    # Implement image resizing logic here if needed
    pass

# Main function to process folders
def main(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            rename_images_in_folder(folder_path)

if __name__ == "__main__":
    root_folder = "D:/PycharmProjects/mP2/Saturation"
    main(root_folder)

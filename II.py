from PIL import Image
import os
# II
# Main folder containing subfolders with images
main_folder = "D:/PycharmProjects/mP2/Saturation"

# Create a destination folder for PNG images
destination_folder = "D:/PycharmProjects/mP2/Saturation_Data"
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Iterate over subfolders in the main folder
for root, dirs, files in os.walk(main_folder):
    for filename in files:
        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", '.jfif')):
            image_path = os.path.join(root, filename)
            img = Image.open(image_path)

            # Change the format to PNG and save it to the destination folder
            new_filename = os.path.splitext(filename)[0] + ".png"
            img.save(os.path.join(destination_folder, new_filename))

# Now all images from subfolders are converted to PNG and saved in the destination folder

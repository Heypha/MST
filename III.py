import os
import csv
# III
def create_csv_file(fileList, labels, output_file):
    # Create and open the CSV file for writing
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header row to the CSV file
        csvwriter.writerow(['Image Path', 'Label'])
        # Write the image paths and labels to the CSV file
        for image_path, label in zip(fileList, labels):
            csvwriter.writerow([image_path, label])

def createFileList(myDir, format='.png', output_file="D:/PycharmProjects/mP2/saturation_dataset.csv"):
    fileList, labels, names = [], [], []
    keywords = {
        "Br": "Bright",
        "Mu": "Muted"
    }

    for root, dirs, files in os.walk(myDir, topdown=True):
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
    # Create the CSV file with image paths and labels
    create_csv_file(fileList, labels, output_file)
    print(fileList)
# Usage:
myDir = "D:/PycharmProjects/mP2/Saturation_Data"
createFileList(myDir)

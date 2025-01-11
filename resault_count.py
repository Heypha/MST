import json
import pandas as pd
from collections import defaultdict

# Load JSON data
with open('Hue_RESULTS.json', 'r') as file:
    data = json.load(file)

# Initialize dictionaries to store counts and total confidence scores
label_counts = defaultdict(int)
label_confidence_scores = defaultdict(float)
correct_predictions = defaultdict(int)

# Process each entry in the JSON data
for entry in data:
    # Extract true label and predicted label
    file_path = entry['File']
    predicted_label = entry['Predicted_Label']

    # True label is the part of the file path after 'test_set_cropped\\'
    true_label = file_path.split('test_set_cropped\\')[-1].split('\\')[0]

    # Increment the count for the true label
    label_counts[true_label] += 1

    # Add the confidence score for the true label
    label_confidence_scores[true_label] += entry['Confidence_Score']

    # Check if the predicted label matches the true label
    if predicted_label == true_label:
        correct_predictions[true_label] += 1

# Calculate mean confidence scores and percentages
results = []
for label in label_counts:
    mean_confidence = label_confidence_scores[label] / label_counts[label]
    total_count = label_counts[label]
    correct_count = correct_predictions[label]
    accuracy_percentage = (correct_count / total_count) * 100 if total_count > 0 else 0

    results.append({
        'Label': label,
        'Count': total_count,
        'Mean Confidence Score': mean_confidence,
        'Correct Predictions': correct_count,
        'Accuracy Percentage': accuracy_percentage
    })

# Create a DataFrame for the results
results_df = pd.DataFrame(results)

# Export to Excel
results_df.to_excel('Hue_label_results.xlsx', index=False)

print("Results exported to label_results.xlsx")

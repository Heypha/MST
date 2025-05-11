import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the trained model
model = joblib.load('final_model.pkl')

# Load and preprocess the new test data
new_df = pd.read_csv('test.csv')

# Drop non-feature columns
X_new = new_df.drop(columns=['file_name', 'true_label'])  # Adjust as needed

# Ensure the new data has the same feature columns as the training data
# You might need to align columns if they are not in the same order

# Make predictions on the new data
new_predictions_encoded = model.predict(X_new)

# Assuming you have the class names from training
class_names = [
    "Bright Winter", "True Winter", "Dark Winter", "Bright Spring", "True Spring",
    "Light Spring", "Light Summer", "True Summer", "Soft Summer", "Soft Autumn",
    "True Autumn", "Dark Autumn"
]

# Fit LabelEncoder on class names
label_encoder = LabelEncoder()
label_encoder.fit(class_names)  # Ensure label encoder is fitted to class names

# Convert encoded predictions to original labels
new_predictions_labels = label_encoder.inverse_transform(new_predictions_encoded)

# Add predictions to the new DataFrame
new_df['predicted_label'] = new_predictions_labels

# Save or inspect the results
new_df.to_csv('new_test_predictions.csv', index=False)
print(new_df[['file_name', 'predicted_label']])

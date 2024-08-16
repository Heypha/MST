import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle  # Import shuffle function

# Load the processed data
df = pd.read_csv('train.csv')

# Shuffle the DataFrame
df = shuffle(df, random_state=42)

# Prepare features (X) and target (y)
X = df.drop(columns=['file', 'True_Label'])  # Drop non-feature columns
y = df['True_Label']  # Target variable

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Optionally, save the trained model for future use
import joblib
joblib.dump(model, 'logistic_regression_model.pkl')

# Predicting on new data
new_predictions = model.predict(X_test)  # Example using X_test
new_predictions_labels = label_encoder.inverse_transform(new_predictions)
print(new_predictions_labels)

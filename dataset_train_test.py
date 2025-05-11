import pandas as pd
import json

# Define the keyword mapping for True_Label
KEYWORDS = {
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

# Define the labels for each model
MODEL_LABELS = {
    "model_1": ["Cool", "Warm"],
    "model_2": ["Bright", "Muted"],
    "model_3": ["Dark", "Light"]
}


def match_true_label(file_name, keywords=KEYWORDS):
    prefix = file_name[:4]
    return keywords.get(prefix, 'Unknown')


def process_json_data(json_file, include_true_labels=True):
    with open(json_file) as f:
        json_data = json.load(f)

    # Initialize a dictionary to store structured results
    structured_data = {}

    # Initialize column names for each model
    column_names = []
    for model, labels in MODEL_LABELS.items():
        for label in labels:
            column_names.append(f'{model}_{label}')

    # Process the JSON data
    for item in json_data:
        file_name = item['file_name']

        if file_name not in structured_data:
            structured_data[file_name] = {col: 0 for col in column_names}

        for prediction in item['predictions']:
            model_name = prediction['model']
            label = prediction['predicted_class']
            score = round(prediction['confidence_score'], 3)

            column_name = f'{model_name}_{label}'
            if column_name in structured_data[file_name]:
                structured_data[file_name][column_name] = score

    # Convert structured data to a pandas DataFrame
    df = pd.DataFrame.from_dict(structured_data, orient='index').reset_index()
    df = df.rename(columns={'index': 'file_name'})

    if include_true_labels:
        df['true_label'] = df['file_name'].apply(match_true_label)

    return df


def prepare_train_dataset(json_file, output_csv=None):
    df_train = process_json_data(json_file, include_true_labels=True)
    if output_csv:
        df_train.to_csv(output_csv, index=False)
    return df_train


def prepare_test_dataset(json_file, output_csv=None):
    df_test = process_json_data(json_file, include_true_labels=False)
    if output_csv:
        df_test.to_csv(output_csv, index=False)
    return df_test


from sklearn.model_selection import train_test_split

def split_train_test(json_file, train_output='train.csv', test_output='test.csv'):
    # Process full dataset with true labels
    df = process_json_data(json_file, include_true_labels=True)

    # Split the dataset 50/50
    df_train, df_test = train_test_split(df, test_size=0.35, random_state=42, shuffle=True)

    # Save to CSV
    df_train.to_csv(train_output, index=False)
    df_test.to_csv(test_output, index=False)

    return df_train, df_test


# Call the function
train_df, test_df = split_train_test('train_output.json')


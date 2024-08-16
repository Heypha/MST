import pandas as pd
import json
from datatset_preparation import get_config
from sklearn.preprocessing import MinMaxScaler


def load_json_data(json_file_path):
    """Load JSON data from a file."""
    with open(json_file_path) as f:
        return json.load(f)


def initialize_data_structures():
    """Initialize and return data structures for storing results."""
    return {}, {}


def process_json_data(json_data, structured_data, model_scores):
    """Process JSON data to populate structured_data and model_scores."""
    for item in json_data:
        file_name = item['file_name']

        if file_name not in structured_data:
            structured_data[file_name] = {}

        for prediction in item['predictions']:
            model_name = prediction['model']
            label = prediction['predicted_class']
            score = prediction['confidence_score']

            if model_name not in model_scores:
                model_scores[model_name] = []

            model_scores[model_name].append({
                'file_name': file_name,
                'label': label,
                'score': score
            })


def normalize_scores(model_scores, structured_data):
    """Normalize scores for each model using Min-Max scaling."""
    scaler = MinMaxScaler()

    for model_name, scores in model_scores.items():
        df_scores = pd.DataFrame(scores)
        df_scores['score'] = scaler.fit_transform(df_scores[['score']]).round(3)

        for _, row in df_scores.iterrows():
            column_name = f'{model_name}_{row["label"]}'
            file_name = row['file_name']
            structured_data[file_name][column_name] = row['score']


def create_dataframe(structured_data):
    """Convert structured data to a pandas DataFrame and process columns."""
    df = pd.DataFrame.from_dict(structured_data, orient='index').reset_index()
    df = df.rename(columns={'index': 'file'})
    df = df.fillna(0)
    return df


def add_true_labels(df):
    """Map file names to true labels and add a 'True_Label' column."""
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
    df['True_Label'] = df['file'].str[:4].map(keywords).fillna('Unknown')


def main():
    config = get_config()
    json_data = load_json_data(config['JSON_OUTPUT'])

    structured_data, model_scores = initialize_data_structures()
    process_json_data(json_data, structured_data, model_scores)
    normalize_scores(model_scores, structured_data)

    df = create_dataframe(structured_data)
    add_true_labels(df)

    print(df.head())
    df.to_csv('train.csv', index=False)


if __name__ == "__main__":
    main()

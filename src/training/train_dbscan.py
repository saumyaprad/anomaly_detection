import argparse
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import yaml

def load_data(path):
    return pd.read_csv(path)

def train_model(data, eps, min_samples):
    scaled_data = StandardScaler().fit_transform(data)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(scaled_data)
    return model

def evaluate_model(model, validation_data):
    anomaly_columns = [col for col in validation_data.columns if col.startswith("anomaly_")]
    combined_scores = []

    for anomaly_col in anomaly_columns:
        X_val = validation_data.drop(columns=anomaly_columns)
        y_val = validation_data[anomaly_col]

        scaled_X_val = StandardScaler().fit_transform(X_val)
        y_pred = model.fit_predict(scaled_X_val)
        y_pred = [1 if x == -1 else 0 for x in y_pred]

        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        combined_scores.append((anomaly_col, f1, precision, recall))

    return combined_scores

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data = load_data(config["data"]["training_data_path"])

    train_data, validation_data = train_test_split(data, test_size=config["dbscan"]["test_size"], random_state=42)

    model = train_model(train_data, config["dbscan"]["eps"], config["dbscan"]["min_samples"])

    scores = evaluate_model(model, validation_data)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name(config["mlflow"]["experiment_name"]).experiment_id):
        mlflow.log_params({
            "eps": config["dbscan"]["eps"],
            "min_samples": config["dbscan"]["min_samples"],
            "test_size": config["dbscan"]["test_size"]
        })

        for anomaly_col, f1, precision, recall in scores:
            print(f"Results for {anomaly_col}: F1 Score: {f1}, Precision: {precision}, Recall: {recall}")
            mlflow.log_metrics({
                f"{anomaly_col}_f1": f1,
                f"{anomaly_col}_precision": precision,
                f"{anomaly_col}_recall": recall
            })

        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config/config_dbscan.yaml")
    args = parser.parse_args()
    main(args.config_path)

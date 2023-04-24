import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import yaml

def load_data(path):
    return pd.read_csv(path)

def plot_anomalies(model, data, idx, anomaly_column, save_plot=False):
    X = data.drop(columns=[col for col in data.columns if col.startswith("anomaly_")])
    y = data[anomaly_column]

    y_pred = model.predict(X)
    y_pred = [1 if x == -1 else 0 for x in y_pred]

    fig, ax = plt.subplots()
    ax.plot(X.loc[idx], color='blue', label='Time Series')

    anomalies = np.where(y_pred == 1)[0]
    for anomaly in anomalies:
        ax.plot(anomaly, X.loc[idx, anomaly], 'ro', markerfacecolor='red', markersize=8, label='Anomaly' if anomaly == anomalies[0] else "")

    ax.legend()
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title(f'Time Series: {idx} with Anomalies')

    if save_plot:
        plt.savefig(f'plots/anomalies_{idx}.png')
    else:
        plt.show()

def main(
        run_id, 
        data_path, 
        idx, 
        anomaly_column, 
        config_path,
        save_plot=False):

    print("Config path", config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(config)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    model = mlflow.sklearn.load_model(f'runs:/{run_id}/model')
    print(f"Loaded model with run ID: {run_id}")
    data = load_data(data_path)
    print(data.head())

    plot_anomalies(model, data, idx, anomaly_column, save_plot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True, help="The run ID of the saved Isolation Forest model in MLflow.")
    parser.add_argument("--data_path", required=True, help="Path to the validation dataset.")
    parser.add_argument("--idx", type=int, required=True, help="Index of the time series to plot.")
    parser.add_argument("--anomaly_column", required=True, help="Anomaly column in the validation dataset.")
    parser.add_argument("--config_path", required=True, help="Path to the config file.")
    parser.add_argument("--save_plot", action="store_true", help="Save the plot as an image instead of displaying it.")
    args = parser.parse_args()

    main(args.run_id, args.data_path, args.idx, args.anomaly_column, args.config_path, args.save_plot)

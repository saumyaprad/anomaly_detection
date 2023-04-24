import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SyntheticTimeSeries:
    def __init__(self, n, m, seasonality, noise_std, anomaly_pct, dataset_type='training'):
        self.n = n
        self.m = m
        self.seasonality = seasonality
        self.noise_std = noise_std
        self.anomaly_pct = anomaly_pct
        self.dataset_type = dataset_type

    def _generate_seasonal_component(self, t):
        seasonal_component = np.sin(2 * np.pi * t / self.seasonality)
        return seasonal_component

    def _generate_noise(self):
        noise = np.random.normal(0, self.noise_std, self.m)
        return noise

    def _generate_anomalies(self, data):
        num_anomalies = int(self.anomaly_pct * self.m)
        anomaly_indices = np.random.choice(self.m, num_anomalies, replace=False)
        anomalies = np.random.normal(data.mean(), 2 * data.std(), num_anomalies)
        data[anomaly_indices] = anomalies
        return data, anomaly_indices

    def generate_time_series(self):
        time_series_data = pd.DataFrame(index=range(self.n), columns=range(self.m))

        if self.dataset_type == 'validation':
            anomaly_labels = np.zeros((self.n, self.m))

        for i in range(self.n):
            t = np.arange(self.m)
            seasonal_component = self._generate_seasonal_component(t)
            noise = self._generate_noise()
            data = seasonal_component + noise
            data, anomaly_indices = self._generate_anomalies(data)

            if self.dataset_type == 'validation':
                anomaly_labels[i, anomaly_indices] = 1

            time_series_data.loc[i] = data

        if self.dataset_type == 'validation':
            anomaly_labels_df = pd.DataFrame(anomaly_labels, columns=[f'anomaly_{i}' for i in range(self.m)])
            time_series_data = pd.concat([time_series_data, anomaly_labels_df], axis=1)

        return time_series_data


if __name__ == '__main__':
    # Configuration
    n = 10  # number of time steps
    m = 1000  # number of time series
    seasonality = 50  # length of the seasonal pattern
    noise_std = 0.5  # standard deviation of the noise
    anomaly_pct = 0.01  # percentage of anomalies in the time series

    # Generate and save synthetic time series data
    for dataset_type in ['training', 'validation']:
        synthetic_ts = SyntheticTimeSeries(n, m, seasonality, noise_std, anomaly_pct, dataset_type)
        time_series_data = synthetic_ts.generate_time_series()
        time_series_data.T.to_csv(f'data/raw/{dataset_type}.csv')

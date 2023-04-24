import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'x': [0.688241, 0.641996, 0.246953, -0.145688, 0.389717, 0.618756, 0.371015, 0.779209, 1.177866, 1.342337, 1.813000, 2.011349, 1.323510, 1.208639, 0.375063, 1.508291, 1.056712, 1.006944, 1.011624, 0.601969],
    'anomalies': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
})

def plot_anomalies(data, window_size, width=10, height=6):
    x = data['x']
    anomalies = data['anomalies']
    rolling_mean = x.rolling(window=window_size).mean()
    rolling_std = x.rolling(window=window_size).std()
    upper_std = rolling_mean + rolling_std
    lower_std = rolling_mean - rolling_std

    fig, ax = plt.subplots(figsize=(width, height))
    ax.plot(x, color='blue', label='Time Series')
    ax.plot(rolling_mean, color='green', label=f'Rolling Mean (Window: {window_size})')
    ax.plot(upper_std, color='orange', linestyle='--', label=f'+1 Std Dev (Window: {window_size})')
    ax.plot(lower_std, color='orange', linestyle='--', label=f'-1 Std Dev (Window: {window_size})')

    ax.fill_between(data.index, lower_std, upper_std, color='orange', alpha=0.1)

    anomaly_indices = anomalies[anomalies == -1].index
    for idx in anomaly_indices:
        ax.plot(idx, x[idx], 'ro', markerfacecolor='red', markersize=8, label='Anomaly' if idx == anomaly_indices[0] else "")

    ax.legend()
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Time Series with Anomalies, Rolling Mean, and +/- Std Dev')
    
    return fig

st.title("Time Series Anomaly Detection")
window_size = st.slider("Window size for rolling mean and std dev:", min_value=1, max_value=len(data)-1, value=5, step=1)
fig = plot_anomalies(data, window_size)
st.pyplot(fig)

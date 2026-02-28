#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fault Detection and Diagnosis Machine Learning System
This script simulates sensor signals, injects various fault types, 
and uses Isolation Forest and SVC for detection and classification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set seed for reproducibility
np.random.seed(42)

def healthy_signal(A, omega, t):
    """Generates an ideal sinusoidal signal."""
    return A * np.sin(omega * t)

def bias(signal, t, bias_value, bias_start_time, bias_end_time):
    """Adds a constant bias to the signal within a specific time range."""
    faulty_signal = np.copy(signal)
    start_index = np.where(t >= bias_start_time)
    end_index = np.where(t >= bias_end_time)
    faulty_signal[start_index[0][0]:end_index[0][0]] += bias_value
    return faulty_signal

def drift(signal, t, drift_rate, drift_start_time, drift_end_time):
    """Adds a linear drift to the signal over time."""
    faulty_signal = np.copy(signal)
    start_index = np.where(t >= drift_start_time)
    end_index = np.where(t >= drift_end_time)
    for i in range(start_index[0][0], end_index[0][0] + 1):
        faulty_signal[i] += drift_rate * (t[i] - drift_start_time)
    return faulty_signal

def spike(signal, t, spike_magnitude, spike_trigger_time):
    """Adds a random-direction spike at a specific timestamp."""
    faulty_signal = np.copy(signal)
    spike_index = np.where(t >= spike_trigger_time)
    faulty_signal[spike_index[0][0]] += spike_magnitude * np.random.choice([-1, 1], replace=False)
    return faulty_signal

def stuck(signal, t, stuck_value, stuck_start_time, stuck_end_time):
    """Forces the signal to a constant value within a range."""
    faulty_signal = np.copy(signal)
    start_index = np.where(t >= stuck_start_time)
    end_index = np.where(t >= stuck_end_time)
    faulty_signal[start_index[0][0]:end_index[0][0]] = stuck_value
    return faulty_signal

def burst(signal, t, burst_magnitude, burst_start_time, burst_end_time):
    """Adds random Gaussian noise to the signal within a range."""
    faulty_signal = np.copy(signal)
    start_index = np.where(t >= burst_start_time)
    end_index = np.where(t >= burst_end_time)
    noise = np.random.normal(0, burst_magnitude, end_index[0][0] - start_index[0][0])
    faulty_signal[start_index[0][0]:end_index[0][0]] += noise
    return faulty_signal

def run_fault_detection_system():
    # Initializing values
    A = 10 
    omega = 2 * np.pi / 10 
    duration = 50
    sampling_rate = 100
    t = np.arange(0, duration + (1 / sampling_rate), 1 / sampling_rate)
    window_size = 1200
    
    # Time indices and magnitudes
    time_lists = [200, 250, 375, 450, 600, 1200, 1325, 1900, 2050, 
                  2900, 2975, 3275, 3375, 3925, 4150, 4275, 4800, 4900]
    bias_val, drift_r, spike_mag, stuck_val, burst_mag = -4, 3, 20, 8, 6

    # Signal Generation
    ideal_signal = healthy_signal(A, omega, t)
    faulty_signal = np.copy(ideal_signal)
    fault_type = np.zeros_like(t)

    for i in range(2):
        base = i * 9
        faulty_signal = bias(faulty_signal, t, bias_val, t[time_lists[base+0]], t[time_lists[base+1]])
        faulty_signal = drift(faulty_signal, t, drift_r, t[time_lists[base+2]], t[time_lists[base+3]])
        faulty_signal = spike(faulty_signal, t, spike_mag, t[time_lists[base+4]])
        faulty_signal = stuck(faulty_signal, t, stuck_val, t[time_lists[base+5]], t[time_lists[base+6]])
        faulty_signal = burst(faulty_signal, t, burst_mag, t[time_lists[base+7]], t[time_lists[base+8]])
        
        # Labeling
        fault_type[time_lists[base+0]:time_lists[base+1]] = 1
        fault_type[time_lists[base+2]:time_lists[base+3]] = 2
        fault_type[time_lists[base+4]] = 3
        fault_type[time_lists[base+5]:time_lists[base+6]] = 4
        fault_type[time_lists[base+7]:time_lists[base+8]] = 5

    # Data Preparation and Feature Engineering
    df = pd.DataFrame({'Time': t, 'Ideal signal': ideal_signal, 
                       'Faulty signal': faulty_signal, 'Actual fault': fault_type.astype(int)})
    
    df['rolling_mean'] = df['Faulty signal'].rolling(window=window_size).mean()
    df['rolling_std'] = df['Faulty signal'].rolling(window=window_size).std()
    
    # Impute missing rolling values
    for i in range(window_size):
        df.iloc[i, 4] = df.iloc[window_size:, 4].mean()
        df.iloc[i, 5] = df.iloc[window_size:, 5].mean()
        
    df['rolling_zscore'] = (df['Faulty signal'] - df['rolling_mean']) / df['rolling_std']
    df['rolling_gradient'] = df['Faulty signal'].diff().fillna(0)

    # Anomaly Detection: Isolation Forest
    features_iso = ['rolling_mean', 'rolling_std', 'rolling_zscore', 'rolling_gradient']
    contamination = (time_lists[1] - time_lists[0] + time_lists[3] - time_lists[2] + time_lists[6] - time_lists[5] + 
                     time_lists[8] - time_lists[7] + time_lists[10] - time_lists[9] + time_lists[12] - time_lists[11] + 
                     time_lists[15] - time_lists[14] + time_lists[17] - time_lists[16])
    
    iso_model = IsolationForest(contamination=contamination/5001, random_state=42)
    iso_model.fit(df[features_iso])
    df['Observed fault'] = (iso_model.predict(df[features_iso]) == -1).astype(int)
    
    acc_iso = round(((df['Actual fault'] > 0) == df['Observed fault']).mean() * 100, 1)
    print(f"Prediction accuracy using Isolation Forest = {acc_iso} %")

    # Fault Classification: SVC
    features_svc = ['Ideal signal', 'Faulty signal', 'rolling_mean', 'rolling_std', 'rolling_zscore', 'rolling_gradient']
    X, y = df[features_svc], df['Actual fault']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    pipeline = Pipeline([
        ('scale', ColumnTransformer([('scaler', StandardScaler(), slice(0, 6))])),
        ('svc', SVC(random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    df['Predicted fault'] = pipeline.predict(X)
    
    acc_svc = round(accuracy_score(y, df['Predicted fault']) * 100, 1)
    cv_svc = round(cross_val_score(pipeline, X_train, y_train, cv=2).mean() * 100, 1)
    print(f"Prediction accuracy using SVC = {acc_svc} %")
    print(f"Cross validation accuracy using SVC = {cv_svc} %")

    # Visualizations
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 20))
    
    ax1.plot(t, ideal_signal, color = 'gray', linestyle = '--', label = 'Ideal signal')
    ax1.plot(t, faulty_signal, color = 'blue', linestyle = '-', label = 'Faulty signal')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Theta (rad)')
    ax1.set_title('Ideal and faulty signal')
    ax1.legend()
    ax2.plot(t, ideal_signal, color = 'gray', linestyle = '--', label = 'Ideal signal')
    ax2.scatter(df.index[df['Actual fault'] > 0] * 0.01, df['Faulty signal'][df['Actual fault'] > 0],
    marker = 'x', color = 'green', s = 72, label = 'Actual fault')
    ax2.scatter(df.index[df['Observed fault'] == 1] * 0.01, df['Faulty signal'][df['Observed fault'] == 1],
    marker = '.', color = 'red', label = 'Observed fault')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Theta (rad)')
    ax2.set_title('Anomaly detection with unsupervised ML using Isolation Forest')
    ax2.legend()
    colors_list_actual = ['blue', 'orange', 'green', 'red', 'black']
    colors_list_predicted = ['orange', 'blue', 'black', 'green', 'red']
    fault_types = ['bias', 'drift', 'spike', 'stuck', 'burst']
    ax3.plot(t, ideal_signal, color = 'gray', linestyle = '--', label = 'Ideal signal')
    for i in range(1, 6):
        ax3.scatter(df.index[df['Actual fault'] == i] * 0.01, df['Faulty signal'][df['Actual fault'] == i],
        marker = 'x', color = colors_list_actual[i - 1], s = 72, label = f"Actual {fault_types[i - 1]}")
        ax3.scatter(df.index[df['Predicted fault'] == i] * 0.01, df['Faulty signal'][df['Predicted fault'] == i],
        marker = '.', color = colors_list_predicted[i - 1], label = f"Predicted {fault_types[i - 1]}")
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Theta (rad)')
    ax3.set_title('Fault classification with supervised ML using Support Vector Classifier (SVC)')
    ax3.legend()
    
    plt.tight_layout()
    fig.savefig('plots.png')

if __name__ == "__main__":
    run_fault_detection_system()
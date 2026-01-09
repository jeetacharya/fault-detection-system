#!/usr/bin/env python
# coding: utf-8

# In[5]:


'''
Fault Detection and Diagnosis System
'''


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

np.random.seed(42)


# Defining function for ideal sinusoidal signal
def healthy_signal(A, omega, t):
    signal = A * np.sin(omega * t)    # defining ideal sinusoidal signal
    return signal

# Defining functions for introducing realistic faults into the ideal signal
def bias(signal, t, bias_value, bias_start_time, bias_end_time):    # introducing constant offset
    faulty_signal = np.copy(signal)
    start_index = np.where(t >= bias_start_time)[0][0]
    end_index = np.where(t >= bias_end_time)[0][0]
    faulty_signal[start_index:end_index] += bias_value
    return faulty_signal

def drift(signal, t, drift_rate, drift_start_time, drift_end_time):    # introducing drift noise
    faulty_signal = np.copy(signal)
    start_index = np.where(t >= drift_start_time)[0][0]
    end_index = np.where(t >= drift_end_time)[0][0]
    for i in range(start_index, end_index + 1):
        faulty_signal[i] += drift_rate * (t[i] - drift_start_time)
    return faulty_signal

def spike(signal, t, spike_magnitude, spike_trigger_times):    # introducing sudden outlier
    faulty_signal = np.copy(signal)
    spike_index = np.where(t >= spike_trigger_times)[0][0]
    faulty_signal[spike_index] += spike_magnitude * np.random.choice([-1, 1], replace = False)
    return faulty_signal

def stuck(signal, t, stuck_value, stuck_start_time, stuck_end_time):    # introducing fault with constant sensor value
    faulty_signal = np.copy(signal)
    start_index = np.where(t >= stuck_start_time)[0][0]
    end_index = np.where(t >= stuck_end_time)[0][0]
    faulty_signal[start_index:end_index] = stuck_value
    return faulty_signal

def burst(signal, t, burst_magnitude, burst_start_time, burst_end_time):    # introducing normally distributed noise
    faulty_signal = np.copy(signal)
    start_index = np.where(t >= burst_start_time)[0][0]
    end_index = np.where(t >= burst_end_time)[0][0]
    faulty_signal[start_index:end_index] += np.random.normal(0, burst_magnitude, end_index - start_index)
    return faulty_signal


# Initializing values
A = 10                      # amplitude
omega = 2 * np.pi / 10      # angular frequency
duration = 50
sampling_rate = 100
t = np.arange(0, duration + (1 / sampling_rate), 1 / sampling_rate)    # time series is [0, 0.01, 0.02, ..., ..., 49.99, 50]
window_size = 1200          # used for rolling feature calculations

fault_type = np.zeros_like(t)    # labelling fault types
time_lists = [200, 250, 375, 450, 600, 1200, 1325, 1900, 2050, 2900, 2975, 3275, 3375, 3925, 4150, 4275, 4800, 4900]    # timestamps for fault triggers


# Fault magnitude values
bias_value = -4
drift_rate = 3
spike_magnitude = 20
stuck_value = 8
burst_magnitude = 6


# Calling functions for calculating ideal signal and faulty signal
ideal_signal = healthy_signal(A, omega, t)
faulty_signal = np.copy(ideal_signal)

for i in range(2):    
    faulty_signal = bias(faulty_signal, t, bias_value, t[time_lists[(i * 9) + 0]], t[time_lists[(i * 9) + 1]])
    faulty_signal = drift(faulty_signal, t, drift_rate, t[time_lists[(i * 9) + 2]], t[time_lists[(i * 9) + 3]])
    faulty_signal = spike(faulty_signal, t, spike_magnitude, t[time_lists[(i * 9) + 4]])
    faulty_signal = stuck(faulty_signal, t, stuck_value, t[time_lists[(i * 9) + 5]], t[time_lists[(i * 9) + 6]])
    faulty_signal = burst(faulty_signal, t, burst_magnitude, t[time_lists[(i * 9) + 7]], t[time_lists[(i * 9) + 8]])
    
    fault_type[time_lists[(i * 9) + 0] : time_lists[(i * 9) + 1]] = 1    # bias corresponds to fault type = 1
    fault_type[time_lists[(i * 9) + 2] : time_lists[(i * 9) + 3]] = 2    # drift corresponds to fault type = 2
    fault_type[time_lists[(i * 9) + 4]] = 3                              # spike corresponds to fault type = 3
    fault_type[time_lists[(i * 9) + 5] : time_lists[(i * 9) + 6]] = 4    # stuck corresponds to fault type = 4
    fault_type[time_lists[(i * 9) + 7] : time_lists[(i * 9) + 8]] = 5    # burst corresponds to fault type = 5
    

# Preparing a pandas DataFrame
table_signal = pd.DataFrame(list(zip(t, ideal_signal, faulty_signal, fault_type)), 
                            columns = ['Time', 'Ideal signal', 'Faulty signal', 'Actual fault'])
table_signal['Actual fault'] = table_signal['Actual fault'].astype(int)


# Calculating features to be used for anomaly detection
table_signal['rolling_mean'] = table_signal['Faulty signal'].rolling(window = window_size).mean()    # rolling mean
table_signal['rolling_std'] = table_signal['Faulty signal'].rolling(window = window_size).std()      # rolling standard deviation
for i in range(window_size):
    rolling_mean_sum = 0
    rolling_mean_std_dev = 0
    for j in range(1, int(len(t) / window_size)):
        rolling_mean_sum += table_signal.iloc[(j * window_size) + i - 1, 4]
        rolling_mean_std_dev += table_signal.iloc[(j * window_size) + i - 1, 5]
    table_signal.iloc[i, 4] = rolling_mean_sum / (int(len(t) / window_size) - 1)           # imputing missing values
    table_signal.iloc[i, 5] = rolling_mean_std_dev / (int(len(t) / window_size) - 1)       # imputing missing values

table_signal['rolling_zscore'] = (table_signal['Faulty signal'] - table_signal['rolling_mean']) / table_signal['rolling_std']    # rolling zscore
table_signal['rolling_gradient'] = table_signal['Faulty signal'].diff().fillna(0)    # rolling gradient


# Detecting anomalies with unsupervised ML using Isolation Forest
features = ['rolling_mean', 'rolling_std', 'rolling_zscore', 'rolling_gradient']    # selecting features required for ML model

contamination_value = ((time_lists[1] - time_lists[0]) + (time_lists[3] - time_lists[2]) + (time_lists[6] - time_lists[5]) + 
                      (time_lists[8] - time_lists[7]) + (time_lists[10] - time_lists[9]) + (time_lists[12] - time_lists[11]) + 
                      (time_lists[15] - time_lists[14]) + (time_lists[17] - time_lists[16]))    # ratio of faulty data to total data
model = IsolationForest(contamination = contamination_value / 5001, random_state = 42)
model.fit(table_signal[features])

table_signal['Fault Score'] = model.decision_function(table_signal[features])
table_signal['prediction'] = model.predict(table_signal[features])
table_signal['Observed fault'] = table_signal['prediction'].apply(lambda x: 1 if x == -1 else 0)    # value = 1 for faulty signal prediction otherwise value = 0
table_signal = table_signal.drop('prediction', axis = 1)
accuracy_isolation_forest = ((table_signal['Actual fault'] > 0) == table_signal['Observed fault']).mean()    # calculating accuracy
accuracy_isolation_forest = round(accuracy_isolation_forest * 100, 1)
print(f"Prediction accuracy using Isolation Forest = {accuracy_isolation_forest} %")


# Classifying fault types with supervised ML using Support Vector Classifier (SVC)
features = ['Ideal signal', 'Faulty signal', 'rolling_mean', 'rolling_std', 'rolling_zscore', 'rolling_gradient']    # selecting features required for ML model

X = table_signal[features]
y = table_signal['Actual fault']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42)

trf1 = ColumnTransformer(transformers = [
    ('scale', StandardScaler(), slice(0, 6))    # transfrmer for scaling data in the required columns
])

trf2 = SVC(random_state = 42)    # transformer for implementing SVC

pipe = Pipeline([
    ('trf1', trf1),
    ('trf2', trf2)
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X)

table_signal['Predicted fault'] = y_pred

accuracy_svc = accuracy_score(y, y_pred)
accuracy_svc = round(accuracy_svc * 100, 1)
print(f"Prediction accuracy using SVC = {accuracy_svc} %")    # accuracy of trained model on test data

average_accuracy_svc = cross_val_score(pipe, X_train, y_train, cv = 2, scoring = 'accuracy').mean()
average_accuracy_svc = round(average_accuracy_svc * 100, 1)
print(f"Cross validation accuracy of model using SVC = {average_accuracy_svc} %")    # cross validation accuracy of the model


# Plotting and saving graphs
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (16, 20))

ax1.plot(t, ideal_signal, color = 'gray', linestyle = '--', label = 'Ideal signal')
ax1.plot(t, faulty_signal, color = 'blue', linestyle = '-', label = 'Faulty signal')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Theta (rad)')
ax1.set_title('Ideal and faulty signal')
ax1.legend()

ax2.plot(np.arange(len(t)), ideal_signal, color = 'gray', linestyle = '--', label = 'Ideal signal')
ax2.scatter(table_signal.index[table_signal['Actual fault'] > 0], table_signal['Faulty signal'][table_signal['Actual fault'] > 0], 
            marker = 'x', color = 'green', s = 72, label = 'Actual fault')
ax2.scatter(table_signal.index[table_signal['Observed fault'] == 1], table_signal['Faulty signal'][table_signal['Observed fault'] == 1], 
            marker = '.', color = 'red', label = 'Observed fault')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Theta (rad)')
ax2.set_title('Anomaly detection with unsupervised ML using Isolation Forest')
ax2.legend()

colors_list_actual = ['blue', 'orange', 'green', 'red', 'black']
colors_list_predicted = ['orange', 'blue', 'black', 'green', 'red']
fault_types = ['bias', 'drift', 'spike', 'stuck', 'burst']
ax3.plot(np.arange(len(t)), ideal_signal, color = 'gray', linestyle = '--', label = 'Ideal signal')
for i in range(1, 6):
    ax3.scatter(table_signal.index[table_signal['Actual fault'] == i], table_signal['Faulty signal'][table_signal['Actual fault'] == i], 
                marker = 'x', color = colors_list_actual[i - 1], s = 72, label = f"Actual {fault_types[i - 1]}")
    ax3.scatter(table_signal.index[table_signal['Predicted fault'] == i], table_signal['Faulty signal'][table_signal['Predicted fault'] == i], 
                marker = '.', color = colors_list_predicted[i - 1], label = f"Predicted {fault_types[i - 1]}")
ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('Theta (rad)')
ax3.set_title('Fault classification with supervised ML using Support Vector Classifier (SVC)')
ax3.legend()

plt.legend()
fig1.savefig('plots.png')
plt.show()


# In[ ]:


get_ipython().system('jupyter nbconvert --to script fault_detection_system.ipynb')


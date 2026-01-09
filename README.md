# fault-detection-system
Fault Detection and Diagnosis System

## Problem:
1. Medical device sensor readings can have practical errors in the real world.
2. These errors arise due to various reasons like encoder miscalibration, temperature-induced drift, loose connections, ADC failures, electrostatic discharges, etc.
3. Since the sensor data is used as feedback to inform the medical device about its parameters in real time, these sensor-induced errors in the obtained sensor data may overcompensate or undercompensate the resulting desired movement.
4. This leads to incorrect sensor data being sent to the device resulting in potential miscalculations for the device’s actuators to compensate for the errors.
5. Hence, these sensor-induced errors need to be detected and classified so that they can be compensated before providing any input to the device’s actuators.

## Approach:
1. When sensor-induced errors (faults) are to be detected and classified from sensor data, a healthy (ideal) signal is required for comparison.
2. We can then inject various types of realistic faulty data that can appear in medical device sensors.
3. The types of faulty data include a constant offset (bias) caused by miscalibrated encoders, sensor error slowly increasing over time (drift) caused by temperature-induced drifts, sudden outliers (spike) caused by EMI or loose connections, fixed sensor value (stuck) caused due to broken wires or ADC failures and temporary high noise (burst) caused due to electrostatic discharge.
4. These faulty sensor readings are now included in the healthy signal at different times and for different durations.
5. The fault types for the specific durations where the faulty data is added are noted for testing accuracy of the models in the future.
6. Unsupervised machine learning techniques like Isolation Forest can be used on this faulty sensor signal to detect anomalies for every timestamp and this can be done by using various features for anomaly detection. The Isolation Forest ML algorithm is used because it efficiently isolates anomalous (inconsistent) data points by randomly splitting the data, requiring fewer splits to isolate anomalies than normal points for unsupervised (unlabelled) datasets.
7. Since the healthy data has intermittent fault types included in the signal, these features need to be captured over moving windows to capture the short term implications of the fault types and detect if a certain reading is an anomaly or not.
8. These features captured over rolling windows include rolling mean, rolling standard deviation, rolling z-score based outlier detection and rolling gradient for anomaly detection.
9. When these features are combined and used to train the Isolation Forest algorithm, it can then segregate anomalous data points from the normal data points.
10. The anomalous data points obtained from the model are then compared with the pre-labelled faulty data points in Step 5 for calculating model accuracy.
11. In order to classify the fault types of the anomalous data points, the Support Vector Classification (SVC) algorithm, which is a supervised ML algorithm, is used along with the same features calculated from rolling windows of the sensor data. The SVC algorithm is used for classifying the sensor’s fault types because it uses the Radial Basis Function to implicitly map complex, non-linearly related input features into a higher dimensional space where different fault types become linearly separable.
12. The original fault types calculated in Step 5 are used as labels for the training dataset. The accuracy is calculated by comparing the original fault types with the predicted labels obtained by running the ML model through the testing dataset.
13. The cross validation score is then calculated to obtain a robust and reliable estimate of the model’s true predictive power and generalizability.

## Results:
1. Predictions of the model trained with Isolation Forest unsupervised ML algorithm on the provided dataset yielded an accuracy of 80.8% which is the percentage of readings correctly predicted as faulty or not by running the model through the dataset.
2. Predictions of the model trained with SVC supervised ML algorithm on the provided dataset yielded an accuracy of 97.6% which is the percentage of readings that were correctly classified with the correct fault types (0 - 5) after running the model through the dataset.
3. Cross validation accuracy of the model trained with SVC supervised ML algorithm on the provided dataset yielded an average accuracy of 95.3%.
4. The prediction accuracy of the model using unsupervised ML algorithm was lower than the prediction accuracy of the model using supervised ML algorithm even with 50% test_size because the model was trained on the “ground truths” and then used for its prediction.
5. The first plot below shows the ideal (healthy) sensor signal and the faulty (real) sensor signal.
6. The second plot below shows the ideal (healthy) sensor signal, true faulty readings in the faulty signal and predicted faulty readings by the model trained with the Isolation Forest algorithm.
7. The overlap between the true and the predicted faulty readings graphically represents the accuracy of the Isolation Forest-trained model with respect to detecting anomalies.
8. The third plot below shows the ideal (healthy) sensor signal, individual true fault types and individual predicted fault types that were predicted by the model trained with the Support Vector Classification (SVC) algorithm.
9. The overlap between the individual true and the individual predicted fault types graphically represents the accuracy of the SVC-trained model with respect to classifying the fault types of the anomalous data points.
![plots](https://github.com/jeetacharya/fault-detection-system/blob/46b73bacd3be6fc8efe092b7fca206ad103f810e/plots.png)

## Industry practices:
1. Industry practices for sensor fault detection and classification use:
   * Model-based techniques for mathematical modeling and residual generation like observer-based method, parity space method and parameter estimation method [1].
   * Signal processing-based techniques like time-domain, frequency-domain and time-frequency domain methods.
   * Data-driven techniques like Linear Discriminant Analysis (LDA), Support Vector Machine (SVM), Random Forest (RF) and Artificial Neural Networks (ANN).
2. References:
   * [1] G. Jombo, E. Zhang, and N. Lu, "Sensor Fault Detection and Diagnosis: Methods and Challenges," 12th IFAC Symposium on Fault Detection, Supervision and Safety for Technical Processes (IFAC SAFEPROCESS 2024), June 2024.

## How to Run:
1. Prerequisites: Python software
2. Execution: Navigate to the directory in the Terminal or Command Prompt containing ‘fault_detection_system.py’ and execute this Python file by running one of the following commands:
   * python fault_detection_system.py
   * python3 fault_detection_system.py

## Custom functions:
1. Ideal signal:
```python
def healthy_signal(A, omega, t):
    # defining ideal sinusoidal signal
    return signal
```

2. Bias:
```python
def bias(signal, t, bias_value, bias_start_time, bias_end_time):
    # defining faulty signal after including bias
    return faulty_signal
```

3. Drift:
```python
def drift(signal, t, drift_rate, drift_start_time, drift_end_time):
    # defining faulty signal after including drift
    return faulty_signal
```

4. Spike:
```python
def spike(signal, t, spike_magnitude, spike_trigger_times):
    # defining faulty signal after including spike
    return faulty_signal
```

5. Stuck:
```python
def stuck(signal, t, stuck_value, stuck_start_time, stuck_end_time):
    # defining faulty signal after including stuck
    return faulty_signal
```

6. Burst:
```python
def burst(signal, t, burst_magnitude, burst_start_time, burst_end_time):
    # defining faulty signal after including burst
    return faulty_signal
```

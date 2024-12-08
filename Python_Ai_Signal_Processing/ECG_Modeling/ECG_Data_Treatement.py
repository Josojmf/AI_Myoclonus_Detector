import csv
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal as signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tkinter as tk
from tkinter import messagebox

# Load ECG sample from CSV file using pandas
dataset = pd.read_csv(r"..\..\More_Datasets\ECG-noise.csv")

y = dataset['hart'].values

# Number of sample points
N = len(y)
# Sampling frequency
Fs = 1000
# Sample spacing
T = 1.0 / Fs
# Time axis
x = np.linspace(0.0, N*T, N)

# Compute FFT of the original signal
yf = scipy.fftpack.fft(y)
# Frequency axis
xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))

# Plot original signal in time and frequency domains
fig_td = plt.figure()
fig_td.canvas.manager.set_window_title('Time Domain Signals')
fig_fd = plt.figure()
fig_fd.canvas.manager.set_window_title('Frequency Domain Signals')
ax1 = fig_td.add_subplot(211)
ax1.set_title('Before Filtering')
ax2 = fig_td.add_subplot(212)
ax2.set_title('After Filtering')
ax3 = fig_fd.add_subplot(211)
ax3.set_title('Before Filtering')
ax4 = fig_fd.add_subplot(212)
ax4.set_title('After Filtering')

# Plot non-filtered inputs
ax1.plot(x, y, color='r', linewidth=0.7)
ax3.plot(xf, 2.0/N * np.abs(yf[:N//2]), color='r', linewidth=0.7, label='raw')
ax3.set_ylim([0, 0.2])

# Design a Butterworth low-pass filter to eliminate 50Hz noise
b, a = signal.butter(4, 50/(Fs/2), 'low')
# Apply the filter
tempf = signal.filtfilt(b, a, y)

# Design a Kaiser window FIR filter to eliminate baseline drift noise
nyq_rate = Fs / 2.0
width = 5.0 / nyq_rate
ripple_db = 60.0
O, beta = signal.kaiserord(ripple_db, width)
cutoff_hz = 4.0
taps = signal.firwin(O, cutoff_hz/nyq_rate, window=('kaiser', beta), pass_zero=False)
# Apply the FIR filter
y_filt = signal.lfilter(taps, 1.0, tempf)

# Compute FFT of the filtered signal
yff = scipy.fftpack.fft(y_filt)

# Plot filtered outputs
ax4.plot(xf, 2.0/N * np.abs(yff[:N//2]), color='g', linewidth=0.7)
ax4.set_ylim([0, 0.2])
ax2.plot(x, y_filt, color='g', linewidth=0.7)

# Save filtered signal to a CSV file with an additional column 'noisy'
filtered_signal_df = pd.DataFrame({'Time': x, 'Filtered Signal': y_filt})
filtered_signal_df['noisy'] = 0  # Add the 'noisy' column with all values set to 0
filtered_signal_df.to_csv(".\ECG-filtered_signal.csv", index=False)

# Save frequency domain data to a CSV file
frequency_domain_df = pd.DataFrame({'Frequency': xf, 'Amplitude': 2.0/N * np.abs(yff[:N//2])})
frequency_domain_df.to_csv("ECG-frequency_domain.csv", index=False)

# Compute moving average to detect peaks
dataset['filt'] = y_filt
hrw = 1  # One-sided window size, as proportion of the sampling frequency
fs = 333  # The example dataset was recorded at 300Hz
mov_avg = dataset['filt'].rolling(int(hrw * fs)).mean()

# Impute where moving average function returns NaN
avg_hr = np.mean(dataset['filt'])
mov_avg = [avg_hr if math.isnan(v) else v for v in mov_avg]
mov_avg = [(0.5 + v) for v in mov_avg]
mov_avg = [v * 1.2 for v in mov_avg]  # Raise the average by 20% to prevent interference
dataset['filt_rollingmean'] = mov_avg

# Detect peaks
window = []
peaklist = []
listpos = 0
for datapoint in dataset['filt']:
    rollingmean = dataset['filt_rollingmean'][listpos]
    if (datapoint < rollingmean) and (len(window) < 1):
        listpos += 1
    elif (datapoint > rollingmean):
        window.append(datapoint)
        listpos += 1
    else:
        maximum = max(window)
        beatposition = listpos - len(window) + window.index(max(window))
        peaklist.append(beatposition)
        window = []
        listpos += 1

ybeat = [dataset['filt'][i] for i in peaklist]

# Plot detected peaks
fig_hr = plt.figure()
fig_hr.canvas.manager.set_window_title('Peak Detector')
ax5 = fig_hr.add_subplot(111)
ax5.set_title("Detected Peaks in Signal")
ax5.plot(dataset['filt'], alpha=0.5, color='blue')
ax5.plot(mov_avg, color='green')
ax5.scatter(peaklist, ybeat, color='red')

# Compute heart rate
RR_list = []
cnt = 0
while cnt < (len(peaklist) - 1):
    RR_interval = (peaklist[cnt + 1] - peaklist[cnt])
    ms_dist = ((RR_interval / fs) * 1000.0)
    RR_list.append(ms_dist)
    cnt += 1

bpm = 60000 / np.mean(RR_list)
print("\n\n\nAverage Heart Beat is: %.1f\n" % bpm)
print("Number of peaks in sample: %d" % len(peaklist))

# Save detected peaks to a CSV file
detected_peaks_df = pd.DataFrame({'Peak Index': peaklist, 'Peak Amplitude': ybeat})
detected_peaks_df.to_csv("ECG_detected_peaks.csv", index=False)

# Save original time-domain signal to a CSV file
original_signal_df = pd.DataFrame({'Time': x, 'Original Signal': y})
original_signal_df.to_csv("ECG_original_signal.csv", index=False)

# Load datasets for model training
file1 = pd.read_csv("ECG_filtered_signal.csv")
file2 = pd.read_csv("ECG_original_signal_labeled.csv")
data = pd.concat([file1, file2])
data = data.sample(frac=1).reset_index(drop=True)

# Prepare data for modeling
X = data[['Filtered Signal']].values
y = data['noisy'].values
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Define and train a Keras model
model = Sequential([
    Dense(16, input_dim=1, activation='relu'),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Save the model
model.save("signal_noise_classifier.h5")

# Popup to ask the user whether to show the plots
def ask_to_plot():
    root = tk.Tk()
    root.withdraw()
    response = messagebox.askyesno("Plot Data", "Do you want to display the plots?")
    return response

if ask_to_plot():
    plt.show()
else:
    print("Plots skipped.")

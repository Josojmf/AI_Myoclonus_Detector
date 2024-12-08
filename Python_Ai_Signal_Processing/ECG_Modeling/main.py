import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal as signal
import tkinter as tk
from tkinter import messagebox

# Function to visualize data (filtering and plotting)
def represent_data():
    dataset = pd.read_csv(r"..\..\More_Datasets\ECG-noise.csv")
    y = dataset['hart'].values

    # Time and frequency setup
    N = len(y)
    Fs = 1000
    T = 1.0 / Fs
    x = np.linspace(0.0, N * T, N)
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    # Filter design
    b, a = signal.butter(4, 50 / (Fs / 2), 'low')
    tempf = signal.filtfilt(b, a, y)
    nyq_rate = Fs / 2.0
    width = 5.0 / nyq_rate
    ripple_db = 60.0
    O, beta = signal.kaiserord(ripple_db, width)
    cutoff_hz = 4.0
    taps = signal.firwin(O, cutoff_hz / nyq_rate, window=('kaiser', beta), pass_zero=False)
    y_filt = signal.lfilter(taps, 1.0, tempf)

    # Visualize
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs[0].plot(x, y, label="Original Signal")
    axs[0].set_title("Original Signal")
    axs[1].plot(x, y_filt, label="Filtered Signal", color="green")
    axs[1].set_title("Filtered Signal")
    axs[2].plot(xf, 2.0 / N * np.abs(yf[:N // 2]), label="Frequency Spectrum", color="red")
    axs[2].set_title("Frequency Spectrum")
    plt.legend()
    plt.show()

# Main training function
def main():
    # Load datasets
    file1 = pd.read_csv("ECG_filtered_signal.csv")
    file2 = pd.read_csv("ECG_original_signal_labeled.csv")
    data = pd.concat([file1, file2]).sample(frac=1).reset_index(drop=True)

    # Prepare features and labels
    X = data[['Signal']].values
    y = data['noisy'].values
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

    # Define and train model
    model = Sequential([
        Dense(16, input_dim=1, activation='relu'),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

    # Evaluate and save model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f}")
    model.save("noisy_signal_classifier.h5")

    # Data visualization option
    root = tk.Tk()
    root.withdraw()
    if messagebox.askyesno("Representation", "Do you want to represent the data?"):
        represent_data()
    else:
        print("Visualization skipped.")

if __name__ == "__main__":
    main()

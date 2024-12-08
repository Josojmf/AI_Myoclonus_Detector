import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Main training function
def main():
    # Load datasets
    file1 = pd.read_csv("ECG_filtered_signal.csv")
    file2 = pd.read_csv("ECG_original_signal_labeled.csv")
    folder = "Emgs-formatted"
    filesArray = ["emg1.csv", "emg2.csv", "emg3.csv", "emg4.csv", "emg5.csv", "emg6.csv", "emg7.csv", "emg8.csv"]
    
    # Load all EMG files and combine them with the main data
    emg_data = []
    for file in filesArray:
        filepath = f"{folder}/{file}"
        emg_data.append(pd.read_csv(filepath))
    emg_combined = pd.concat(emg_data, axis=0)
    
    # Combine all datasets
    data = pd.concat([file1, file2, emg_combined]).sample(frac=1).reset_index(drop=True)

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
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

    # Evaluate and save model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f}")
    model.save("noisy_signal_classifier.h5")
    print("Model saved as 'noisy_signal_classifier.h5'")

if __name__ == "__main__":
    main()

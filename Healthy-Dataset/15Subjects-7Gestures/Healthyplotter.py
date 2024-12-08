import os
import sys
import signal
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\nProgram interrupted! Exiting gracefully...")
    sys.exit(0)

# Register the handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

def load_and_normalize_data(file_path):
    """
    Load data from CSV, skip the timestamp column, and normalize EMG data to a range between 0 and 1.
    
    :param file_path: Path to the CSV file
    :return: Normalized DataFrame containing only EMG data
    """
    # Load the data and skip the timestamp column
    data = pd.read_csv(file_path)
    emg_data = data.iloc[:, 1:]  # Skip the first column (timestamp)
    
    # Normalize the EMG data
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = pd.DataFrame(scaler.fit_transform(emg_data), columns=emg_data.columns)
    return normalized_data

def plot_emg_data(data, title="EMG Signal Data"):
    """
    Plot EMG data from a DataFrame, with each column representing a channel.
    
    :param data: DataFrame with EMG data
    :param title: Title of the plot
    """
    plt.figure(figsize=(12, 6))
    for column in data.columns:
        plt.plot(data[column], label=f"Channel {column}")
    
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Signal Amplitude")
    plt.legend()
    plt.show()

def main():
    # Define base folder where extracted dataset resides
    base_folder = os.path.join("..", "Healthy-Dataset", "15Subjects-7Gestures")

    # Prompt user to select a subject
    print("Available Subjects:")
    for i in range(15):
        print(f"S{i}")
    
    subject_code = input("Enter the subject code (e.g., 'S0', 'S1'): ").strip()
    subject_folder = os.path.join(base_folder, subject_code)

    if not os.path.exists(subject_folder):
        print(f"Subject folder '{subject_code}' does not exist.")
        return

    # Dynamically detect available gestures by scanning files in the subject folder
    gestures = {}
    for file_name in os.listdir(subject_folder):
        if file_name.endswith(".csv"):
            gesture_code = file_name.split("-")[1].replace(".csv", "")
            gestures[gesture_code] = gesture_code  # Set description as code by default
    if not gestures:
        print(f"No gesture files found for subject '{subject_code}'.")
        return

    # Display available gestures and prompt user for gesture code
    print("\nAvailable Gestures:")
    for code in gestures:
        print(f"{code}")
    
    gesture_code = input("Enter the gesture code to plot (e.g., 'fistdwn'): ").strip()

    # Search for the specified gesture file in the subject folder
    file_found = False
    for file_name in os.listdir(subject_folder):
        if gesture_code in file_name and file_name.endswith(".csv"):
            file_path = os.path.join(subject_folder, file_name)
            print(f"Loading data from: {file_path}")
            try:
                data = load_and_normalize_data(file_path)
                plot_emg_data(data, title=f"Subject: {subject_code} - Gesture: {gesture_code}")
                file_found = True
                break
            except Exception as e:
                print(f"Error loading data from {file_path}: {e}")
                continue

    if not file_found:
        print(f"No data found for gesture '{gesture_code}' in subject '{subject_code}'. Ensure the code and file exist.")

if __name__ == "__main__":
    main()

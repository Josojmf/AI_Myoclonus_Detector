import numpy as np
import os
import sys
import signal
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from Model.ModelTrainer import EMGModelTrainer
from DataTreatement.DataFormatter.DatasetLoader import DatasetLoader

# Parameters
WINDOW_SIZE = 127  # Actual row count of the dataset
TARGET_NUM_CHANNELS = 9
MODEL_PATH = "./Model/final_trained_model_spastic_vs_healthy.keras"
HEALTHY_FOLDER = "C:/INFORMATICA/TFG/Codigo/Healthy-Dataset"
SPASTIC_FOLDER = "C:/INFORMATICA/TFG/Codigo/Spasticity-Dataset"

# Signal handler for graceful interruption
def signal_handler(sig, frame):
    print("\nProgram interrupted! Exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Function to add a zero-filled column to each CSV file and ensure consistent columns
def add_zero_column_to_csv_files(folder_path, new_column_name="label", target_columns=TARGET_NUM_CHANNELS):
    """
    Recursively adds a column filled with zeros to the end of each CSV file in the specified folder
    and overwrites the original file to ensure all files have a consistent number of columns.
    """
    for root, _, files in os.walk(folder_path):  # Walk through directories
        for file_name in files:
            if file_name.endswith(".csv"):  # Only process CSV files
                file_path = os.path.join(root, file_name)
                df = pd.read_csv(file_path)

                # Add columns if missing
                missing_columns = target_columns - len(df.columns)
                if missing_columns > 0:
                    # Add missing columns filled with zeros
                    for i in range(missing_columns):
                        df[f"extra_col_{i}"] = 0

                # Trim columns if there are extra columns
                df = df.iloc[:, :target_columns]

                # Add the new column with zeros at the end if it doesn't already exist
                if new_column_name not in df.columns:
                    df[new_column_name] = 0

                # Ensure final column count matches target
                df = df.iloc[:, :target_columns]

                # Overwrite the original file with the modified DataFrame
                df.to_csv(file_path, index=False)
                print(f"File modified and saved: {file_path}")

def check_consistent_column_count(folder_path, target_columns):
    consistent = True
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".csv"):
                file_path = os.path.join(root, file_name)
                df = pd.read_csv(file_path)
                if df.shape[1] != target_columns:
                    print(f"Inconsistent column count in {file_path}: {df.shape[1]} columns (expected {target_columns})")
                    consistent = False
    return consistent

def main():
    # Ensure the model directory exists
    model_dir = os.path.dirname(MODEL_PATH)
    os.makedirs(model_dir, exist_ok=True)

    # Step 1: Ensure all files have consistent columns
    print("Ensuring consistent column structure for healthy data...")
    add_zero_column_to_csv_files(HEALTHY_FOLDER, target_columns=TARGET_NUM_CHANNELS)
    print("Ensuring consistent column structure for spastic data...")
    add_zero_column_to_csv_files(SPASTIC_FOLDER, target_columns=TARGET_NUM_CHANNELS)

    # Step 2: Load datasets with DatasetLoader
    dataset_loader = DatasetLoader(window_size=WINDOW_SIZE, target_num_channels=TARGET_NUM_CHANNELS)
    healthy_data, healthy_labels = dataset_loader.load_all_files_from_folder(HEALTHY_FOLDER, spastic_label=0)
    spastic_data, spastic_labels = dataset_loader.load_all_files_from_folder(SPASTIC_FOLDER, spastic_label=1)

    # Check for empty data
    if healthy_data.size == 0 or spastic_data.size == 0:
        print("Error: Missing data in one or both classes.")
        return

    # Step 3: Ensure consistent channel sizes
    min_num_channels = min(healthy_data.shape[2], spastic_data.shape[2])
    healthy_data = healthy_data[:, :, :min_num_channels]
    spastic_data = spastic_data[:, :, :min_num_channels]

    # Step 4: Balance dataset sizes
    min_len = min(len(healthy_data), len(healthy_labels))
    if len(spastic_data) > min_len:
        indices = np.random.choice(len(spastic_data), min_len, replace=False)
        spastic_data = spastic_data[indices]
        spastic_labels = spastic_labels[indices]
    else:
        healthy_data = healthy_data[:min_len]
        healthy_labels = healthy_labels[:min_len]

    # Step 5: Combine datasets for training
    combined_data = np.concatenate([healthy_data, spastic_data], axis=0)
    combined_labels = np.concatenate([healthy_labels, spastic_labels], axis=0)

    # Step 6: Train-test split
    X_train, X_val, Y_train, Y_val = train_test_split(
        combined_data, combined_labels, test_size=0.2, random_state=42, stratify=combined_labels
    )
    Y_train, Y_val = to_categorical(Y_train, num_classes=2), to_categorical(Y_val, num_classes=2)

    # Step 7: Model training and evaluation
    trainer = EMGModelTrainer(window_size=WINDOW_SIZE, num_channels=min_num_channels, num_classes=2, model_path=MODEL_PATH)
    trainer.train_and_evaluate(X_train, Y_train, X_val, Y_val, epochs=100)
    trainer.save_model()

if __name__ == "__main__":
    main()
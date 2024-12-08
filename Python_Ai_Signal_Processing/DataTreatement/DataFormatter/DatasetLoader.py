import numpy as np
import pandas as pd
import os

class DatasetLoader:
    def __init__(self, window_size=127, target_num_channels=9):
        """
        Initialize the DatasetLoader with specific parameters.

        Parameters:
            window_size (int): Number of rows per window in each CSV file.
            target_num_channels (int): Target number of channels (columns) in each data window.
        """
        self.window_size = window_size
        self.target_num_channels = target_num_channels

    def load_all_files_from_folder(self, folder_path, spastic_label=0):
        """
        Recursively load all CSV files from a folder and its subdirectories.

        Parameters:
            folder_path (str): Path to the main folder containing subdirectories of CSV files.
            spastic_label (int): Label to assign to the loaded data (0 for healthy, 1 for spastic).

        Returns:
            np.array: Combined data from all CSV files.
            np.array: Corresponding labels.
        """
        data = []
        labels = []
        for root, _, files in os.walk(folder_path):  # Recursively walk through directories
            for file_name in files:
                if file_name.endswith(".csv"):  # Only process CSV files
                    file_path = os.path.join(root, file_name)
                    print(f"Loading file: {file_path}")
                    try:
                        df = pd.read_csv(file_path)

                        # Ensure the DataFrame has the target number of channels
                        if df.shape[1] > self.target_num_channels:
                            df = df.iloc[:, :self.target_num_channels]
                        elif df.shape[1] < self.target_num_channels:
                            # Add missing columns filled with zeros if fewer than target channels
                            for i in range(self.target_num_channels - df.shape[1]):
                                df[f"extra_col_{i}"] = 0

                        # Convert the DataFrame into windows
                        num_windows = len(df) // self.window_size
                        for i in range(num_windows):
                            window_data = df.iloc[i * self.window_size:(i + 1) * self.window_size].values
                            data.append(window_data)
                            labels.append(spastic_label)
                    except Exception as e:
                        print(f"Error loading file {file_name}: {e}")

        if not data:
            print("No data loaded from the folder.")

        return np.array(data), np.array(labels) if data else (np.array([]), np.array([]))

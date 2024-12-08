import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional

class EMGPreprocessor:
    """
    A class for normalizing and segmenting EMG data.
    """
    def __init__(self, window_size: int = 128, normalize: bool = True):
        """
        Initialize the preprocessor with data parameters.

        :param window_size: Size of each data window for segmentation.
        :param normalize: Whether to normalize the data.
        """
        self.window_size = window_size
        self.normalize = normalize
        self.scaler = MinMaxScaler() if normalize else None
        self.data = None
        self.windows = None
        self.labels = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.

        :param file_path: Path to the data file.
        :return: DataFrame with the loaded data.
        """
        try:
            self.data = pd.read_csv(file_path)
            print("Data loaded successfully.")
            return self.data
        except FileNotFoundError:
            print("Data file not found. Please check the path.")
            return None

    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess data array directly (normalize if specified).

        :param data: Raw EMG data as a NumPy array.
        :return: Preprocessed (and optionally normalized) data.
        """
        if self.normalize:
            data = self.scaler.fit_transform(data)
            print("Data normalized.")
        return data

    def segment_data(self, data: np.ndarray) -> np.ndarray:
        """
        Segment data into fixed-size windows for model input.

        :param data: Preprocessed data as a NumPy array.
        :return: Data segmented into windows.
        """
        windows = []
        for i in range(0, len(data) - self.window_size + 1, self.window_size):
            windows.append(data[i:i + self.window_size])
        segmented_data = np.array(windows)
        print(f"Data segmented into {len(segmented_data)} windows.")
        return segmented_data

    def generate_labels(self, num_windows: int, label_type: int) -> np.ndarray:
        """
        Generate labels for each window segment.

        :param num_windows: Number of windows for which labels are generated.
        :param label_type: Label for each segment (e.g., 0 for Healthy, 1 for Spastic).
        :return: Array of labels.
        """
        labels = np.full(num_windows, label_type)
        print("Labels generated.")
        return labels

    def preprocess(self, data: Optional[np.ndarray] = None, file_path: Optional[str] = None, label_type: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the full preprocessing pipeline on the provided data or file path.

        :param data: Optional raw data as a NumPy array (if loading from a file is not required).
        :param file_path: Optional path to a data file.
        :param label_type: Label type for the data segments.
        :return: Tuple of preprocessed windows and labels.
        """
        # Load data from file if provided
        if file_path is not None:
            data = self.load_data(file_path).values  # Convert DataFrame to NumPy array

        if data is None:
            raise ValueError("No data provided. Please provide either data or file_path.")

        # Normalize and segment the data
        data = self.preprocess_data(data)
        segmented_data = self.segment_data(data)

        # Generate labels
        labels = self.generate_labels(len(segmented_data), label_type)
        return segmented_data, labels

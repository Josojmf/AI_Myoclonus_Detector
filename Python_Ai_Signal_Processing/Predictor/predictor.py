import sys
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class EMGPredictor:
    def __init__(self, model_path, data_path=None, window_size=51, num_channels=9):
        """
        Initializes the predictor with the specified model path, data path,
        and model parameters.
        """
        self.model_path = model_path
        self.data_path = data_path
        self.window_size = window_size
        self.num_channels = num_channels
        self.model = None
        self.predicted_classes = None
        self.class_labels = ["Healthy", "Spastic"]
        self.new_data = None

    def load_model(self):
        """Loads the pre-trained model."""
        self.model = load_model(self.model_path)
        print("Model loaded successfully.")

    def set_data_path(self, data_path):
        """Sets a new data path for the predictor."""
        self.data_path = data_path
        print(f"Data path set to: {self.data_path}")

    def preprocess_data(self):
        """Loads and preprocesses the new data from CSV to match the model's input shape."""
        if self.data_path is None:
            raise ValueError(
                "Data path is not set. Please provide a CSV file path using set_data_path().")

        try:
            # Load data
            new_data = pd.read_csv(self.data_path, header=None)

            # Ensure data has 9 columns
            num_channels = new_data.shape[1]
            print(f"Original data shape: {new_data.shape}")

            if num_channels < 9:
                new_data = pd.concat([new_data, pd.DataFrame(
                    0, index=new_data.index, columns=range(num_channels, 9))], axis=1)
            elif num_channels > 9:
                new_data = new_data.iloc[:, :9]

            # Normalize the data
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(new_data)

            # Reshape data into windows
            target_elements = (len(normalized_data) //
                               self.window_size) * self.window_size
            reshaped_data = normalized_data[:target_elements].reshape(
                -1, self.window_size * 9)

            # Pad each sample to have 1984 elements if necessary
            if reshaped_data.shape[1] < 1984:
                padding = np.zeros(
                    (reshaped_data.shape[0], 1984 - reshaped_data.shape[1]))
                reshaped_data = np.hstack((reshaped_data, padding))

            print("Final data shape after padding:", reshaped_data.shape)

            # Update self.new_data with the final reshaped and padded data
            self.new_data = reshaped_data

        except Exception as e:
            print(f"Error loading data from {self.data_path}: {e}")
            self.new_data = None

            """Loads and preprocesses the new data from CSV to match the model's input shape."""
            if self.data_path is None:
                raise ValueError(
                    "Data path is not set. Please provide a CSV file path using set_data_path().")

            try:
                # Load data
                new_data = pd.read_csv(self.data_path, header=None)

                # Check if the data has fewer than 9 columns (channels)
                num_channels = new_data.shape[1]
                print(f"Original data shape: {new_data.shape}")

                if num_channels < 9:
                    new_data = pd.concat([new_data, pd.DataFrame(
                        0, index=new_data.index, columns=range(num_channels, 9))], axis=1)
                elif num_channels > 9:
                    new_data = new_data.iloc[:, :9]

                # Normalize the data
                scaler = MinMaxScaler()
                normalized_data = scaler.fit_transform(new_data)

                # Reshape data into windows
                target_elements = (len(normalized_data) //
                                   self.window_size) * self.window_size
                reshaped_data = normalized_data[:target_elements].reshape(
                    -1, self.window_size, 9)

                # Debug prints
                print("Normalized and reshaped data shape:", reshaped_data.shape)
                print("Sample windows from reshaped data:",
                      reshaped_data[:5])  # Print first 5 windows

                # Update self.new_data
                self.new_data = reshaped_data

            except Exception as e:
                print(f"Error loading data from {self.data_path}: {e}")
                self.new_data = None

        def predict(self):
            """Performs predictions on the preprocessed data."""
            if self.model is None:
                raise ValueError(
                    "Model is not loaded. Please call load_model() first.")
            if self.new_data is None:
                raise ValueError(
                    "Data is not preprocessed. Please call preprocess_data() first.")

            # Perform predictions on each window and store results
            predictions = self.model.predict(self.new_data)
            self.predicted_classes = np.argmax(predictions, axis=1)

            # Debug print for each window's prediction
            print("Predicted class for each window:")
            for i, pred in enumerate(self.predicted_classes):
                print(
                    f"Window {i}: Predicted Class = {self.class_labels[pred]}")

            # Count occurrences of each class
            print("Count of predicted classes:",
                  Counter(self.predicted_classes))

        if self.model is None:
            raise ValueError(
                "Model is not loaded. Please call load_model() first.")
        if self.new_data is None:
            raise ValueError(
                "Data is not preprocessed. Please call preprocess_data() first.")

        # Perform predictions on each window and store results
        predictions = self.model.predict(self.new_data)
        self.predicted_classes = np.argmax(predictions, axis=1)

        # Debug print for each window's prediction
        print("Predicted class for each window:")
        for i, pred in enumerate(self.predicted_classes):
            print(
                f"Window {i}: Predicted Class = {self.class_labels[pred]}")

            # Count occurrences of each class
        print("Count of predicted classes:",
              Counter(self.predicted_classes))

    def show_results(self):
        """Displays the prediction results in a pop-up window with periodic check for SIGINT."""
        if self.predicted_classes is None:
            raise ValueError(
                "No predictions to show. Please call predict() first.")

        # Count occurrences of each class
        class_counts = Counter(self.predicted_classes)

        # Create a pop-up window
        result_window = tk.Tk()
        result_window.title("EMG Prediction Results")
        result_window.geometry("400x300")

        # Summary of predictions
        summary_text = f"Total Predictions: {len(self.predicted_classes)}\n"
        for cls, count in class_counts.items():
            summary_text += f"{self.class_labels[cls]}: {count}\n"

        summary_label = tk.Label(
            result_window, text=summary_text, font=("Arial", 12))
        summary_label.pack(pady=10)

        # Plot the results
        fig, ax = plt.subplots()
        ax.bar(self.class_labels, [class_counts.get(
            0, 0), class_counts.get(1, 0)], color=['blue', 'orange'])
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title("Prediction Results")

        # Display the plot in the pop-up window
        canvas = FigureCanvasTkAgg(fig, result_window)
        canvas.get_tk_widget().pack()
        canvas.draw()

        # Set up periodic check for SIGINT (Ctrl+C)
        def check_for_interrupt():
            try:
                # Re-run this check every 100 ms
                result_window.after(100, check_for_interrupt)
            except KeyboardInterrupt:
                print("Program interrupted! Exiting gracefully...")
                result_window.destroy()
                sys.exit(0)

        check_for_interrupt()  # Start the periodic check
        result_window.mainloop()

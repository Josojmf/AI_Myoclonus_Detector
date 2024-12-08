import pandas as pd
import matplotlib.pyplot as plt
import os

class EMGPlotter:
    def __init__(self):
        # Define the feature and label mappings
        self.feature_map = {
            "standard_deviation": 0,
            "root_mean_square": 8,
            "minimum": 16,
            "maximum": 24,
            "zero_crossings": 32,
            "average_amplitude_change": 40,
            "amplitude_first_burst": 48,
            "mean_absolute_value": 56,
            "wave_form_length": 64,
            "willison_amplitude": 72
        }
        self.labels_dict = {
            1: "index_finger",
            2: "middle_finger",
            3: "ring_finger",
            4: "little_finger",
            5: "thumb",
            6: "rest",
            7: "victory_gesture"
        }

    def load_data(self, file_path):
        """
        Loads EMG data from a specified file path.
        """
        self.df = pd.read_csv(file_path)
        self.labels = self.df.iloc[:, -1]  # The last column with gesture labels
        self.features = self.df.iloc[:, :-1]  # First 80 columns with features

    def plot_feature(self, feature_name="standard_deviation", electrode=1, normalize=False):
        """
        Plots a specific feature for a chosen electrode across time.
        
        Parameters:
        - feature_name (str): The name of the feature to plot. Options include:
          'standard_deviation', 'root_mean_square', 'minimum', 'maximum', 'zero_crossings',
          'average_amplitude_change', 'amplitude_first_burst', 'mean_absolute_value',
          'wave_form_length', 'willison_amplitude'
        - electrode (int): The electrode number (1 through 8).
        - normalize (bool): If True, normalize the feature values.
        """
        # Verify feature and electrode inputs
        if feature_name not in self.feature_map:
            print("Invalid feature name. Using 'standard_deviation' as default.")
            feature_name = "standard_deviation"
        if electrode < 1 or electrode > 8:
            print("Invalid electrode number. Using electrode 1 as default.")
            electrode = 1

        # Calculate column index based on feature and electrode
        col_index = self.feature_map[feature_name] + (electrode - 1)
        amplitude = self.features.iloc[:, col_index]

        # Normalize if needed
        if normalize:
            amplitude = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min())

        # Plotting time vs amplitude
        plt.figure(figsize=(12, 6))
        plt.plot(self.features.index, amplitude, label=f'{feature_name.capitalize()} (Electrode {electrode})')
        plt.xlabel("Time (Sample Index)")
        plt.ylabel("Amplitude" + (" (Normalized)" if normalize else ""))
        plt.title(f"Time vs {feature_name.capitalize()} for Electrode {electrode}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_all_labels(self, normalize=False):
    
        for label, label_name in self.labels_dict.items():
            # Filter rows for the current label
            label_data = self.features[self.labels == label]
            
            # Check if there is any data for this label
            if label_data.empty:
                print(f"No data available for label: {label_name}")
                continue
            
            print(f"Plotting data for {label_name}...")
            for feature_name, start_col in self.feature_map.items():
                plt.figure(figsize=(12, 8))
                
                for electrode in range(8):
                    col_index = start_col + electrode
                    
                    # Check that col_index is within bounds
                    if col_index >= label_data.shape[1]:
                        print(f"Warning: Column index {col_index} out-of-bounds for feature {feature_name}.")
                        continue
                    
                    # Extract amplitude values for the electrode and feature
                    amplitude = label_data.iloc[:, col_index]
                    
                    # Normalize if needed
                    if normalize:
                        amplitude = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min())
                    
                    plt.plot(
                        amplitude.values, 
                        label=f'Electrode {electrode + 1}'
                    )
                
                plt.xlabel("Time (Sample Index)")
                plt.ylabel("Amplitude" + (" (Normalized)" if normalize else ""))
                plt.title(f"{label_name.capitalize()} - {feature_name.capitalize()}")
                plt.legend()
                plt.grid(True)
                plt.show()


# Example usage:
# if __name__ == "__main__":
#     plotter = EMGPlotter()
#     plotter.load_data("/path/to/rawdataEMG.csv")
#     # Plot a single feature for a single electrode
#     plotter.plot_feature(feature_name="standard_deviation", electrode=1, normalize=True)
#     # Plot all features for each label
#     plotter.plot_all_labels(normalize=True)

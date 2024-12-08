import pandas as pd
import matplotlib.pyplot as plt
import re

class EMGDataPlotter:
    """
    A class to plot EMG data from various muscle groups and electrodes.
    """
    def __init__(self, file_path, muscle_name):
        """
        Initialize the plotter with the file path and muscle name.

        :param file_path: Path to the CSV file containing EMG data.
        :param muscle_name: Name of the muscle.
        """
        self.file_path = file_path
        self.muscle_name = self.parse_muscle_name(muscle_name)

    def parse_muscle_name(self, muscle_name):
        """
        Parse and format the muscle name.

        :param muscle_name: Raw muscle name.
        :return: Formatted muscle name.
        """
        muscle_name = muscle_name.split('-')[0]
        return re.sub(r"(\w)([A-Z])", r"\1 \2", muscle_name)

    def plot_columns(self, variables):
        """
        Plot each variable (column) in the EMG data on a grid.

        :param variables: Dictionary of data columns where keys are column names and values are data arrays.
        """
        num_cols = 5  # Display 5 columns per row
        num_rows = len(variables) // num_cols + (1 if len(variables) % num_cols > 0 else 0)

        plt.figure(figsize=(20, 4 * num_rows))
        plt.suptitle(f'EMG Data for {self.muscle_name}')

        for i, (var_name, data) in enumerate(variables.items(), start=1):
            plt.subplot(num_rows, num_cols, i)
            plt.plot(data)
            plt.title(f'Electrode {i} - {var_name}')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the suptitle
        plt.show()

import pandas as pd

# Load the raw data without headers
file_path = 'rawdataEMG.csv'  # Update with the correct path to your file
data = pd.read_csv(file_path, header=None)

# Manually set the column names
target_columns = ['timestamp', 'emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8']
data.columns = target_columns[:len(data.columns)]  # Assign only as many names as columns present in the data

# Print the loaded data to verify the structure
print("Data after renaming columns:")
print(data.head())

# Convert columns to numeric if needed, handling errors by setting non-numeric entries to NaN
data['timestamp'] = pd.to_numeric(data['timestamp'], errors='coerce')
emg_columns = target_columns[1:len(data.columns)]
data[emg_columns] = data[emg_columns].apply(pd.to_numeric, errors='coerce')

# Fill any NaN values with zero if necessary
data.fillna(0, inplace=True)

# Save the adjusted DataFrame to a new CSV file
output_path = 'formatted_rawdataEMG.csv'
data.to_csv(output_path, index=False)

print(f"Data formatted and saved to {output_path}")

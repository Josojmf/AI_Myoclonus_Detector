import os
import pandas as pd

# Get the current directory (assumed to be "Emgs-formatted")
current_folder = os.getcwd()

# Iterate over all files in the current folder
for filename in os.listdir(current_folder):
    if filename.endswith(".csv"):  # Process only CSV files
        filepath = os.path.join(current_folder, filename)
        try:
            # Load the CSV file
            df = pd.read_csv(filepath)
            
            # Add the new column
            df['noisy'] = 1
            
            # Save the updated file back to the same location
            df.to_csv(filepath, index=False)
            print(f"Updated {filename} successfully.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("All files updated.")

import os
import pandas as pd

# Folder containing the datasets
input_folder = "../../Electro-Myography-EMG-Dataset/extracted_features_dataset"  # Replace with the actual folder path
output_folder = "../train_data/Processed"  # Folder to save modified files
os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

# Check if the input folder contains files
if not os.path.exists(input_folder):
    print(f"Error: Input folder '{input_folder}' does not exist.")
else:
    print(f"Processing files in folder: {input_folder}")

# Loop through all files in the input folder
import os
import pandas as pd

# Folder containing the datasets
input_folder = "../../Electro-Myography-EMG-Dataset/extracted_features_dataset"  # Replace with the actual folder path
output_folder = "../train_data/Processed"  # Folder to save modified files
os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

# Process all CSV files in the input folder and subfolders
for root, dirs, files in os.walk(input_folder):
    for filename in files:
        if filename.endswith(".csv"):  # Process only CSV files
            file_path = os.path.join(root, filename)
            print(f"Processing file: {file_path}")

            # Load the dataset
            df = pd.read_csv(file_path)
            
            # Add a new column to indicate spasticity
            df["Spasticity"] = 1  # Indicating spasticity with a value of 1
            
            # Define the output file path with a modified name in the output folder
            # This will keep the original subfolder structure in the output folder
            relative_path = os.path.relpath(file_path, input_folder)
            output_path = os.path.join(output_folder, f"{os.path.splitext(relative_path)[0]}_with_spasticity.csv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create any necessary subdirectories
            
            # Save the modified dataset
            df.to_csv(output_path, index=False)
            print(f"Dataset saved with spasticity indicator as '{output_path}'")

    else:
        print(f"Skipping non-CSV file: {filename}")

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import io  # Import io.StringIO instead of using pd.compat.StringIO

# Define the file path
file_path = "../train_data/PredictionData/MixedHalfandHalf.csv"

# Load the data and split by the separator
with open(file_path, "r") as f:
    lines = f.readlines()

# Find the separator line
separator_index = next(i for i, line in enumerate(lines) if line.strip() == "#SEPARATION")

# Split the lines into two parts
part1_lines = lines[:separator_index]
part2_lines = lines[separator_index + 1:]

# Load each part into a DataFrame using io.StringIO
part1_df = pd.read_csv(io.StringIO("".join(part1_lines)), header=None)
part2_df = pd.read_csv(io.StringIO("".join(part2_lines)), header=None)

# Initialize the scaler (MinMaxScaler for 0-1 normalization, or use StandardScaler for Z-score)
scaler1 = MinMaxScaler()  # For first part
scaler2 = MinMaxScaler()  # For second part (you can also try StandardScaler if desired)

# Normalize each part
part1_normalized = scaler1.fit_transform(part1_df.iloc[:, :-1])  # Exclude label column if present
part2_normalized = scaler2.fit_transform(part2_df.iloc[:, 1:])   # Exclude timestamp column for normalization

# Reassemble each part, adding back the last column if needed
part1_normalized = pd.DataFrame(part1_normalized).assign(label=part1_df.iloc[:, -1].values)
part2_normalized = pd.DataFrame(part2_normalized).assign(timestamp=part2_df.iloc[:, 0].values)

# Save the normalized data back to CSV
output_path = "normalized_data.csv"
with open(output_path, "w") as f:
    part1_normalized.to_csv(f, header=False, index=False)
    f.write("#SEPARATION\n")
    part2_normalized.to_csv(f, header=False, index=False)

print(f"Normalized data saved to {output_path}")

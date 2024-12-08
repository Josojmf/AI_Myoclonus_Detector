import pandas as pd
import numpy as np

# Define parameters
num_samples = 150  # Number of rows (samples)
num_features = 8  # Match the spastic dataset's 8 columns

# Function to generate smooth, periodic data with slight variations
def generate_healthy_emg_data(num_samples, num_features, amplitude=1, noise_level=0.05, label=6):
    np.random.seed(0)  # For reproducibility
    data = []

    for i in range(num_samples):
        # Generate smooth periodic signals (e.g., sine waves) with slight variations
        base_signal = amplitude * np.sin(2 * np.pi * (i / num_samples) * np.linspace(0.5, 1, num_features))
        noise = np.random.normal(0, noise_level, num_features)  # Small random noise
        feature_values = base_signal + noise
        data.append(feature_values)

    # Convert to DataFrame and add labels
    df = pd.DataFrame(data, columns=[f"Feature_{i+1}" for i in range(num_features)])
    df["Label"] = label  # Label '6' for resting state, representing healthy relaxed muscle activity
    df["Spasticity"] = 0  # Indicating non-spastic with a value of 0
    
    return df

# Generate and save three different healthy EMG datasets
for i, (amplitude, noise_level) in enumerate([(1, 0.05), (0.8, 0.04), (1.2, 0.06)], start=1):
    df = generate_healthy_emg_data(num_samples, num_features, amplitude=amplitude, noise_level=noise_level)
    file_name = f"healthy_EMG_synthetic_{i}.csv"
    df.to_csv(file_name, index=False)
    print(f"Synthetic healthy EMG dataset saved as '{file_name}'")

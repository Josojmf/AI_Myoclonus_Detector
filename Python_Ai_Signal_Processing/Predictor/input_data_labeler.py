import pandas as pd
import numpy as np
import os

# Define gesture labels
gesture_map = {
    'fistdwn': 'Closed Hand', 'fistout': 'Closed Hand',
    'opendwn': 'Open Hand', 'openout': 'Open Hand',
    'twodwn': 'Victory Sign', 'twout': 'Victory Sign',
    'tap': 'Tap Action',
    'right': 'Wrist Extension', 'left': 'Wrist Flexion',
    'neut': 'Neutral'
}

extract_dir = '../../../25-healthy-parts-emg-datastaset'
# Load and label each file
data = []
labels = []

for root, dirs, files in os.walk(extract_dir):
    for file in files:
        if file.endswith('.csv'):  # Assuming files are in CSV format
            filepath = os.path.join(root, file)
            df = pd.read_csv(filepath)
            
            # Extract the gesture label from the filename
            gesture = gesture_map.get(file.split('.')[0])
            if gesture:
                data.append(df)
                labels.append(gesture)

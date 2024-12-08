import os
import sys
import signal
from Predictor.predictor import EMGPredictor

# CTRL+C handler
def signal_handler(sig, frame):
    print("\nProgram interrupted! Exiting gracefully...")
    sys.exit(0)

# Register the handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

def main():
    # Parameters
    window_size = 127
    model_path = "./Model/final_trained_model_spastic_vs_healthy.keras"

    # Initialize and use EMGPredictor for new predictions with GUI display
    predictor = EMGPredictor(
        model_path=model_path,
        window_size=51,  # Adjust this if different
        num_channels=9
    )
    # Load the trained model
    predictor.load_model() 
    # Prompt user for a new data file for predictions
    input_new_file_name = input("Enter the name of the file to predict (e.g., sample.csv): ")
    base_folder = r"train_data/PredictionData"
    new_data_path = os.path.join(base_folder, input_new_file_name)
    new_data_path = os.path.normpath(new_data_path)
    predictor.set_data_path(new_data_path)
    predictor.preprocess_data()
    predictor.predict()
    predictor.show_results()


    # Print the final path for debugging
    print(f"Final path to check: {new_data_path}")

    # Verify if the file exists
    if not os.path.isfile(new_data_path):
        print(f"Error: The file '{new_data_path}' does not exist.")
        return

    # Set the new data path and preprocess the new data file
    predictor.set_data_path(new_data_path)
    predictor.preprocess_data()

    # Only proceed if there is data to predict
    if predictor.new_data is not None and predictor.new_data.shape[0] > 0:
        predictor.predict()  # Make predictions on the new data
        predictor.show_results()  # Display results in a Tkinter window
    else:
        print("Not enough data to perform prediction.")

if __name__ == "__main__":
    main()

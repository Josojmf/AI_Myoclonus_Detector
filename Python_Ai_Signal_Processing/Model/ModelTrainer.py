from .cnn_model import model_handler
from tensorflow.keras.models import load_model

class EMGModelTrainer:
    """Handles training, evaluation, loading, and saving of the EMG model."""
    
    def __init__(self, window_size=128, num_channels=8, num_classes=2, model_path="./Model/final_trained_model_spastic_vs_healthy.keras"):
        self.window_size = window_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.model_path = model_path
        
        # Initialize model using model_handler
        self.model = model_handler(window_size=window_size, num_channels=num_channels, num_classes=num_classes)
    
    def train_and_evaluate(self, X_train, Y_train, X_val, Y_val, epochs=50):
        # Train the model
        history = self.model.train_model(X_train, Y_train, X_val, Y_val, epochs=epochs)
        return history
    
    def save_model(self):
        """Save the trained model to the specified path."""
        self.model.save_model(self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load the trained model from the specified path."""
        self.model.model = load_model(self.model_path)
        print(f"Model loaded from {self.model_path}")
        return self.model.model

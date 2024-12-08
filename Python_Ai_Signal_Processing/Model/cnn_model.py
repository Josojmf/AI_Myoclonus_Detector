from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model


class model_handler:
    """
    This class builds and manages a CNN model for EMG signal classification.
    """

    def __init__(self, window_size, num_channels, num_classes):
        """
        Initialize the model handler with model parameters and build the CNN model.

        :param window_size: Size of the input time window for each sample.
        :param num_channels: Number of channels in the input data.
        :param num_classes: Number of output classes for classification.
        """
        self.window_size = window_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        """
        Build and compile a 1D CNN model.

        :return: Compiled Keras model.
        """
        model = Sequential([
            Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(self.window_size, self.num_channels)),
            MaxPooling1D(pool_size=2),
            Dropout(0.5),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, X_train, Y_train, X_val, Y_val, epochs=50, batch_size=32):
        """
        Train the CNN model with training and validation data.

        :param X_train: Training data features.
        :param Y_train: Training data labels.
        :param X_val: Validation data features.
        :param Y_val: Validation data labels.
        :param epochs: Number of training epochs.
        :param batch_size: Size of the batches for each training step.
        :return: Training history object for the model.
        """
        # Model checkpoint to save the best model during training
        checkpoint_path = 'emg_model_checkpoint.keras'  # Modify path as needed
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', verbose=1)

        history = self.model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stopping],
            verbose=1
        )
        return history

    def evaluate_model(self, X_test, Y_test):
        """
        Evaluate the model on test data.

        :param X_test: Test data features.
        :param Y_test: Test data labels.
        :return: Loss and accuracy of the model on the test set.
        """
        loss, accuracy = self.model.evaluate(X_test, Y_test, verbose=1)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
        return loss, accuracy

    def save_model(self, filepath):
        """
        Save the trained model to a specified file.

        :param filepath: Path where the model will be saved.
        """
        self.model.save(filepath)
        print(f"Model saved at {filepath}")

    def load_model(self, filepath):
        """
        Load a trained model from a specified file.

        :param filepath: Path to the saved model file.
        """
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")

import pandas as pd
import joblib
import numpy as np
from utils.logger import logger

class Predictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        try:
            # Load the model using joblib
            model = joblib.load(self.model_path)
            logger.info("Model loaded successfully.")
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        except Exception as e:
            logger.info("Error loading the model.")
            raise Exception(f"Error loading the model: {str(e)}")
        

    def validate_input(self, input_data):
        # Ensure input_data is a DataFrame
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        return input_data

    def predict(self, input_data):
        # Validate input data
        input_data = self.validate_input(input_data)
        try:
            # Make predictions
            predictions = self.model.predict(input_data)
            return predictions
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    # Path to the model
    model_path = "Artifact/model.pkl"
    
    # Example input data
    input_data = pd.DataFrame({
        'age': [30, 40],
        'job': [1, 2],
        'marital': [1, 0],
        'education': [2, 1],
        'default': [0, 1],
        'balance': [500, 1500],
        'housing': [1, 0],
        'loan': [0, 1],
        'contact': [1, 2],
        'day_of_week': [3, 5],
        'month': [6, 7],
        'duration': [120, 300],
        'campaign': [1, 2],
        'pdays': [999, 10],
        'previous': [0, 2]
    })

    # Instantiate the Predictor class
    predictor = Predictor(model_path)

    try:
        # Make predictions
        predictions = predictor.predict(input_data)
        print("Predictions:", predictions)
    except Exception as e:
        print("Error:", str(e))

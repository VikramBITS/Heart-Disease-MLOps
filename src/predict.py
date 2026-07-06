import os
import joblib
import pandas as pd

# Get the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build the model path
MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_disease_model.pkl")

# Load model
model = joblib.load(MODEL_PATH)


def predict_heart_disease(input_data):
    """
    Predict heart disease from input features.

    Parameters:
        input_data (list): List containing 13 input features.

    Returns:
        prediction (int): 0 = No Heart Disease, 1 = Heart Disease
        probability (float): Probability of Heart Disease
    """

    # Convert input into DataFrame
    columns = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal"
    ]

    input_df = pd.DataFrame([input_data], columns=columns)

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Prediction probability
    probability = model.predict_proba(input_df)[0][1]

    return prediction, probability
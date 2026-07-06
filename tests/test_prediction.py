import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict_heart_disease


def test_prediction():
    sample = [
        63, 1, 3, 145, 233,
        1, 0, 150, 0,
        2.3, 0, 0, 1
    ]

    prediction, probability = predict_heart_disease(sample)

    assert prediction in [0, 1]
    assert 0 <= probability <= 1
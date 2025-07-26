import joblib
import numpy as np

def test_prediction_shape():
    model = joblib.load("app/model.joblib")
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    pred = model.predict(sample)
    assert pred.shape == (1,)

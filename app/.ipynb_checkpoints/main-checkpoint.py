from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("app/model.joblib")

class IrisInput(BaseModel):
    features: list

@app.post("/predict")
def predict(input: IrisInput):
    data = np.array([input.features])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np

app = FastAPI()

STUDENT_NAME = "Urvashi Salonia"
ROLL_NO = "2022BCS0222"

# Load the latest model
model = mlflow.sklearn.load_model("models/model")

class PredictRequest(BaseModel):
    features: list

@app.get("/")
def root():
    return {
        "name": STUDENT_NAME,
        "roll_no": ROLL_NO,
        "message": "MLOps Assignment API is running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "name": STUDENT_NAME,
        "roll_no": ROLL_NO
    }

@app.post("/predict")
def predict(request: PredictRequest):
    features = np.array(request.features).reshape(1, -1)
    prediction = model.predict(features)
    return {
        "prediction": int(prediction[0]),
        "name": STUDENT_NAME,
        "roll_no": ROLL_NO
    }
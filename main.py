from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ECGRequest(BaseModel):
    ecg: list[float]

@app.post("/predict")
def predict(req: ECGRequest):

    ecg = req.ecg

    return {
        "length": len(ecg),
        "max": max(ecg),
        "min": min(ecg),
        "mean": sum(ecg) / len(ecg)
    }
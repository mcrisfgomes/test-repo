import numpy as np
import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from keras.models import load_model

app = FastAPI(title="Breast Cancer Classifier API")

model = load_model("model.keras")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")


NUM_FEATURES = 30


class Features(BaseModel):
    features: list[float] = Field(min_length=NUM_FEATURES, max_length=NUM_FEATURES)


class Prediction(BaseModel):
    diagnosis: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(data: Features):
    X = scaler.transform([data.features])
    probabilities = model.predict(X)[0]
    predicted_index = int(np.argmax(probabilities))
    diagnosis = encoder.inverse_transform([predicted_index])[0]

    return {
        "diagnosis": diagnosis,
        "confidence": round(float(probabilities[predicted_index]), 4),
        "probabilities": {
            label: round(float(prob), 4)
            for label, prob in zip(encoder.classes_, probabilities)
        },
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)#, reload=True)



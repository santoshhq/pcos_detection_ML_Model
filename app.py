from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field,computed_field
from typing import Annotated, Literal
import pandas as pd
import joblib
import asyncio

app = FastAPI(title="PCOS Prediction API (Random Forest)")

# -------------------------------
# Load model + scaler
# -------------------------------
model = None
scaler = None

@app.on_event("startup")
async def load_model():
    global model, scaler
    try:
        model = await asyncio.to_thread(joblib.load, "pcos_model.pkl")
        scaler = await asyncio.to_thread(joblib.load, "scaler.pkl")
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")


# -------------------------------
# Features (must match training)
# -------------------------------
FEATURES = [
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "BMI",
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Cycle length(days)",
    "Skin darkening (Y/N)",
    "Pimples(Y/N)",
    "Fast food (Y/N)",
    "Reg.Exercise(Y/N)"
]


# -------------------------------
# Input Schema
# -------------------------------
class PatientInput(BaseModel):
    cp: int
    trestbps: float
    chol: float
    fbs: Literal[0, 1]
    restecg: int
    thalach: float

    weight: float
    height: float

    weight_gain: Literal[0, 1]
    hair_growth: Literal[0, 1]
    cycle_length: int
    skin_darkening: Literal[0, 1]
    pimples: Literal[0, 1]
    fast_food: Literal[0, 1]
    reg_exercise: Literal[0, 1]

    # ✅ FIXED BMI (inside class)
    @computed_field
    @property
    def bmi(self) -> float:
        height_m = self.height / 100
        return round(self.weight / (height_m ** 2), 2)
# -------------------------------
# Health Check
# -------------------------------
@app.get("/")
async def home():
    return {"message": "PCOS Prediction API (RF) running 🚀"}


# -------------------------------
# Core Prediction Logic
# -------------------------------
def run_model(data: PatientInput):

    patient = pd.DataFrame([{
        "cp": data.cp,
        "trestbps": data.trestbps,
        "chol": data.chol,
        "fbs": data.fbs,
        "restecg": data.restecg,
        "thalach": data.thalach,
        "BMI": data.bmi,
        "Weight gain(Y/N)": data.weight_gain,
        "hair growth(Y/N)": data.hair_growth,
        "Cycle length(days)": data.cycle_length,
        "Skin darkening (Y/N)": data.skin_darkening,
        "Pimples(Y/N)": data.pimples,
        "Fast food (Y/N)": data.fast_food,
        "Reg.Exercise(Y/N)": data.reg_exercise
    }])

    # Ensure correct order
    patient = patient[FEATURES]

    # Apply scaling (IMPORTANT)
    patient_scaled = pd.DataFrame(
        scaler.transform(patient),
        columns=FEATURES
    )

    # Prediction
    prob = float(model.predict_proba(patient_scaled)[:, 1][0])
    score = round(prob * 10, 2)

    # Status + Advice
    if score >= 8:
        status = "High Risk (Red)"
        advice = "Consult a doctor soon and monitor symptoms."
    elif score >= 5:
        status = "Moderate Risk (Yellow)"
        advice = "Maintain healthy lifestyle and track symptoms."
    else:
        status = "Low Risk (Green)"
        advice = "Keep maintaining a healthy routine."

    return prob, score, status, advice


# -------------------------------
# Prediction Endpoint
# -------------------------------
@app.post("/predict")
async def predict(data: PatientInput):

    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        prob, score, status, advice = await asyncio.to_thread(run_model, data)

        return {
            "PCOS Probability": round(prob, 2),
            "Score": score,
            "Status": status,
            "Advice": advice
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# PCOS Prediction API

A FastAPI-based machine learning API that predicts PCOS risk using a trained Random Forest model.

The API:
- loads a saved model and scaler at startup
- computes BMI automatically from weight and height
- transforms input with the same feature order/scaling used in training
- returns probability, score, risk status, and simple advice

---

## 1) Project Overview

This project exposes one main endpoint:
- POST /predict: accepts patient features and returns PCOS risk prediction.

It also includes:
- GET /: simple health message.

The app is asynchronous:
- startup model loading is offloaded with asyncio.to_thread
- prediction work is offloaded with asyncio.to_thread

This keeps the FastAPI event loop responsive while running blocking ML operations.

---

## 2) Tech Stack

- Python
- FastAPI
- Uvicorn
- Pydantic v2
- pandas
- scikit-learn / joblib

Main files:
- app.py: API app, schema, prediction logic
- requirements.txt: dependency versions used for this project
- pcos_model.pkl: trained model file (must exist)
- scaler.pkl: scaler file (must exist)

---

## 3) Setup (Windows PowerShell)

From the project folder, create and activate virtual environment (if needed):

```powershell
python -m venv venv310
.\venv310\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Important:
- Ensure pcos_model.pkl and scaler.pkl are present in the project root.
- If activation is blocked by policy, run PowerShell as Admin and set execution policy for current user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 4) Run The API

Use the project interpreter explicitly:

```powershell
.\venv310\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8001 --reload
```

Open:
- API root: http://127.0.0.1:8001/
- Interactive docs (Swagger): http://127.0.0.1:8001/docs
- ReDoc docs: http://127.0.0.1:8001/redoc

Why use python -m uvicorn?
- It guarantees Uvicorn runs from this environment.
- Running plain uvicorn app:app can accidentally use a global install and cause DLL/version mismatch errors.

---

## 5) API Endpoints

### GET /

Returns service status message.

Example response:

```json
{
	"message": "PCOS Prediction API (RF) running 🚀"
}
```

### POST /predict

Accepts patient data and returns prediction details.

Request body fields:

| Field | Type | Notes |
|---|---|---|
| cp | integer | Chest pain type |
| trestbps | number | Resting blood pressure |
| chol | number | Cholesterol |
| fbs | 0 or 1 | Fasting blood sugar flag |
| restecg | integer | Resting ECG result |
| thalach | number | Max heart rate |
| weight | number | Weight in kg |
| height | number | Height in cm |
| weight_gain | 0 or 1 | Weight gain Y/N |
| hair_growth | 0 or 1 | Hair growth Y/N |
| cycle_length | integer | Cycle length in days |
| skin_darkening | 0 or 1 | Skin darkening Y/N |
| pimples | 0 or 1 | Pimples Y/N |
| fast_food | 0 or 1 | Fast food Y/N |
| reg_exercise | 0 or 1 | Regular exercise Y/N |

Computed automatically:
- BMI = weight / (height_in_meters^2)

Example request:

```json
{
	"cp": 1,
	"trestbps": 120,
	"chol": 180,
	"fbs": 0,
	"restecg": 1,
	"thalach": 150,
	"weight": 62,
	"height": 160,
	"weight_gain": 1,
	"hair_growth": 1,
	"cycle_length": 40,
	"skin_darkening": 1,
	"pimples": 1,
	"fast_food": 1,
	"reg_exercise": 0
}
```

Example response:

```json
{
	"PCOS Probability": 0.83,
	"Score": 8.3,
	"Status": "High Risk (Red)",
	"Advice": "Consult a doctor soon and monitor symptoms."
}
```

---

## 6) How Scoring Works

The model outputs probability from 0.0 to 1.0.

Score formula:
- score = probability x 10

Risk mapping:
- score >= 8: High Risk (Red)
- score >= 5 and < 8: Moderate Risk (Yellow)
- score < 5: Low Risk (Green)

---

## 7) Error Handling

Common API errors:

- 500 Model not loaded
	- cause: model/scaler files missing or unreadable
	- fix: place pcos_model.pkl and scaler.pkl in project root

- 422 Unprocessable Entity
	- cause: request body invalid type or missing field
	- fix: send all required fields with correct types

---

## 8) Troubleshooting

If server exits with code 1:
- run with project interpreter:

```powershell
.\venv310\Scripts\python.exe -m uvicorn app:app --port 8001
```

- verify model files:

```powershell
Get-ChildItem .\pcos_model.pkl, .\scaler.pkl
```

- verify package versions:

```powershell
pip show fastapi uvicorn pydantic pandas scikit-learn xgboost joblib
```

- ensure only one active environment and no conflicting global uvicorn in PATH.

---

## 9) Notes

- This API is for educational/support use and is not a medical diagnosis tool.
- Always consult a qualified doctor for clinical decisions.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib

app = FastAPI(
    title="Ford Fleet AI Copilot API",
    description="Predicts 30-day vehicle failure risk for fleet vehicles",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "failure_risk_model.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "model_training_table.csv"
EXPLANATIONS_PATH = BASE_DIR / "data" / "processed" / "vehicle_risk_explanations.csv"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

training_df = pd.read_csv(DATA_PATH)

if EXPLANATIONS_PATH.exists():
    explanations_df = pd.read_csv(EXPLANATIONS_PATH)
else:
    explanations_df = pd.DataFrame()


class VehicleRequest(BaseModel):
    vehicle_id: str


class FeatureRequest(BaseModel):
    features: dict


def get_risk_level(probability: float) -> str:
    if probability >= 0.66:
        return "High"
    elif probability >= 0.33:
        return "Medium"
    return "Low"


@app.get("/")
def home():
    return {
        "message": "Ford Fleet AI Copilot API is running",
        "available_endpoints": [
            "/health",
            "/predict/vehicle",
            "/predict/features"
        ]
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_count": len(feature_columns),
        "vehicle_count": len(training_df)
    }


@app.post("/predict/vehicle")
def predict_vehicle(request: VehicleRequest):
    vehicle_id = request.vehicle_id

    vehicle_row = training_df[training_df["vehicle_id"] == vehicle_id]

    if vehicle_row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Vehicle ID {vehicle_id} not found"
        )

    X = vehicle_row[feature_columns]

    probability = float(model.predict_proba(X)[0, 1])
    prediction = int(model.predict(X)[0])
    risk_level = get_risk_level(probability)

    response = {
        "vehicle_id": vehicle_id,
        "failure_probability": round(probability, 4),
        "failure_probability_percent": f"{probability:.2%}",
        "prediction": prediction,
        "risk_level": risk_level
    }

    if not explanations_df.empty:
        explanation_row = explanations_df[
            explanations_df["vehicle_id"] == vehicle_id
        ]

        if not explanation_row.empty:
            response["explanation"] = {
                "top_features": explanation_row.iloc[0]["top_features"],
                "recommendations": explanation_row.iloc[0]["recommendations"]
            }

    return response


@app.post("/predict/features")
def predict_from_features(request: FeatureRequest):
    input_data = pd.DataFrame([request.features])

    missing_cols = set(feature_columns) - set(input_data.columns)

    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {list(missing_cols)[:10]}"
        )

    input_data = input_data[feature_columns]

    probability = float(model.predict_proba(input_data)[0, 1])
    prediction = int(model.predict(input_data)[0])
    risk_level = get_risk_level(probability)

    return {
        "failure_probability": round(probability, 4),
        "failure_probability_percent": f"{probability:.2%}",
        "prediction": prediction,
        "risk_level": risk_level
    }
# =============== Model service for CO2 concentration prediction ===============

import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from src.model import CO2Transformer
from src.utils import load_scalers

# --- Config ---
model_dir = os.path.join(_PROJECT_ROOT, "saved_models")
INPUT_WINDOW = int(os.environ.get("INPUT_WINDOW", "19"))
FORECAST_WINDOW = int(os.environ.get("FORECAST_WINDOW", "18"))
MODEL_VERSION = f"transformer_iw{INPUT_WINDOW}_fw{FORECAST_WINDOW}"

# --- Load model + scalers on startup ---
app = FastAPI(title="CO2 Prediction API", version="1.0.0")

model = None
general_scaler = None
conc_scaler = None


@app.on_event("startup")
def load_model():
    global model, general_scaler, conc_scaler

    # Load scalers
    general_scaler, conc_scaler = load_scalers()

    # Load model
    model = CO2Transformer(forecast_window=FORECAST_WINDOW)
    save_path = os.path.join(model_dir, f"model_iw{INPUT_WINDOW}_fw{FORECAST_WINDOW}.pt")
    if os.path.exists(save_path):
        state = torch.load(save_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded model from {save_path}")
    else:
        print(f"WARNING: No saved model at {save_path}, using random weights")
    model.eval()


# --- Request/Response schemas ---
class SensorReading(BaseModel):
    """Per timestep: 90 sensor values + sampling point label."""
    features: list[float] = Field(..., min_length=90, max_length=90, description="90 sensor feature values (raw, unnormalized)")
    sampling_point: int = Field(..., ge=1, le=6, description="Current sampling point (1-6)")


class PredictRequest(BaseModel):
    readings: list[SensorReading] = Field(..., min_length=1, description=f"Sequence of sensor readings (ideally {INPUT_WINDOW} timesteps)")
    experiment_id: Optional[str] = None


class PointPrediction(BaseModel):
    point: int
    co2_percent: float


class PredictResponse(BaseModel):
    predictions: list[list[PointPrediction]]  # [forecast_step][point]
    model_version: str
    input_window: int
    forecast_window: int
    timestamp: str


class IngestRequest(BaseModel):
    experiment_id: str
    timestamp: str
    sampling_point: int = Field(..., ge=1, le=6)
    features: list[float] = Field(..., min_length=90, max_length=90)
    co2_measured: Optional[float] = None


# --- Endpoints ---
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION,
        "input_window": INPUT_WINDOW,
        "forecast_window": FORECAST_WINDOW,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(500, "Model not loaded")

    readings = req.readings
    if len(readings) < INPUT_WINDOW:
        raise HTTPException(
            400,
            f"Need at least {INPUT_WINDOW} readings, got {len(readings)}. "
            f"Pad with repeated first reading or provide more data."
        )

    # Use last INPUT_WINDOW readings
    readings = readings[-INPUT_WINDOW:]

    # Build feature matrix: normalize 90 sensors, add one-hot label
    raw_features = np.array([r.features for r in readings])  # (iw, 90)
    normalized = general_scaler.transform(raw_features)       # (iw, 90)

    # One-hot encode sampling point
    onehot = np.zeros((INPUT_WINDOW, 6))
    for i, r in enumerate(readings):
        onehot[i, r.sampling_point - 1] = 1

    # Combine: (iw, 96)
    full_features = np.hstack([normalized, onehot]).astype(np.float32)
    x = torch.from_numpy(full_features).unsqueeze(0)  # (1, iw, 96)

    # Predict
    with torch.no_grad():
        pred_norm = model(x).numpy()[0]  # (fw, 6)

    # Inverse transform to real CO2 %
    pred_real = conc_scaler.inverse_transform(
        pred_norm.reshape(-1, 1)
    ).reshape(FORECAST_WINDOW, 6)

    # Format response
    predictions = []
    for step in range(FORECAST_WINDOW):
        step_preds = [
            PointPrediction(point=p + 1, co2_percent=round(float(pred_real[step, p]), 4))
            for p in range(6)
        ]
        predictions.append(step_preds)

    # Store prediction in DB 
    try:
        from deployment.db import insert_prediction
        pred_dict = {
            f"point_{p+1}": [round(float(pred_real[s, p]), 4) for s in range(FORECAST_WINDOW)]
            for p in range(6)
        }
        insert_prediction(
            datetime.utcnow(), MODEL_VERSION, INPUT_WINDOW, FORECAST_WINDOW, pred_dict
        )
    except Exception:
        pass  # DB is optional

    return PredictResponse(
        predictions=predictions,
        model_version=MODEL_VERSION,
        input_window=INPUT_WINDOW,
        forecast_window=FORECAST_WINDOW,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/ingest")
def ingest(req: IngestRequest):
    """Store sensor reading in PostgreSQL."""
    try:
        from deployment.db import insert_reading
        reading_id = insert_reading(
            req.timestamp, req.experiment_id, req.sampling_point,
            req.features, req.co2_measured
        )
        return {"id": reading_id, "status": "stored"}
    except Exception as e:
        raise HTTPException(500, f"Database error: {e}")


@app.get("/predictions")
def get_predictions_endpoint(start: Optional[str] = None, end: Optional[str] = None, limit: int = 100):
    """Retrieve historical predictions."""
    try:
        from deployment.db import get_predictions
        rows = get_predictions(start, end, limit)
        return {"predictions": rows, "count": len(rows)}
    except Exception as e:
        raise HTTPException(500, f"Database error: {e}")

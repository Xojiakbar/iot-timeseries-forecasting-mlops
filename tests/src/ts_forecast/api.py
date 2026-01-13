from __future__ import annotations



import os

from typing import List



import numpy as np

import torch

from fastapi import FastAPI, HTTPException

from pydantic import BaseModel, Field



from ts_forecast.predict import load_model, predict_next

from ts_forecast.utils import resolve_device



app = FastAPI(title="IoT Time-Series Forecast API", version="0.1.0")



MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pt")

DEVICE_CFG = os.getenv("DEVICE", "auto")



_loaded = None





class PredictRequest(BaseModel):

    # series shape: [seq_len, D]

    series: List[List[float]] = Field(..., description="2D list shaped [seq_len, num_features]")





class PredictResponse(BaseModel):

    horizon: int

    prediction: List[float]





@app.on_event("startup")

def startup_event():

    global _loaded

    device = resolve_device(DEVICE_CFG)

    if not os.path.exists(MODEL_PATH):

        # Provide a helpful message rather than crashing

        _loaded = None

        print(f"[WARN] Model not found at {MODEL_PATH}. Train first.")

        return

    _loaded = load_model(MODEL_PATH, device=device)

    print(f"[INFO] Loaded model from {MODEL_PATH} on {device}")





@app.get("/health")

def health():

    return {"status": "ok", "model_loaded": _loaded is not None}





@app.post("/predict", response_model=PredictResponse)

def predict(req: PredictRequest):

    if _loaded is None:

        raise HTTPException(status_code=400, detail="Model not loaded. Train first or mount model file.")

    series = np.array(req.series, dtype=np.float32)

    if series.ndim != 2:

        raise HTTPException(status_code=400, detail="series must be 2D: [seq_len, num_features].")

    if series.shape[1] != _loaded.input_size:

        raise HTTPException(

            status_code=400,

            detail=f"Expected num_features={_loaded.input_size}, got {series.shape[1]}",

        )

    pred = predict_next(_loaded, series)

    return PredictResponse(horizon=int(_loaded.horizon), prediction=[float(x) for x in pred])



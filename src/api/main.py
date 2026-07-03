"""
FastAPI Inference Service

Phase 1 deliverable: Basic inference API for ML model serving.
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
model: Any = None
model_path = PROJECT_ROOT / "models/xgb_severity.joblib"


def load_model() -> Any:
    if not os.path.exists(model_path):
        return None
    start = time.time()
    loaded = joblib.load(model_path)
    # MODEL_LOAD_TIME.set(time.time() - start)
    return loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    yield


app = FastAPI(
    title="ML Inference API",
    description="Production inference service for ML models",
    version="1.0.0",
    lifespan=lifespan,
)


# Request/Response models
class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="Input features for prediction")


class PredictResponse(BaseModel):
    prediction: float | int
    probabilities: list[float] | None = None
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", model_loaded=model is not None)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Make a prediction using the loaded model.

    Phase 1 deliverable: Basic inference endpoint.
    """
    global model

    # INFERENCE_REQUESTS.inc()

    if model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Train a model first."
        )

    try:
        # Reshape for single prediction
        x = np.array(request.features).reshape(1, -1)

        # Get prediction
        pred = model.predict(x)

        # Get probabilities if available
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0].tolist()

        # PREDICTIONS_MADE.inc()

        return PredictResponse(
            prediction=float(pred[0])
            if hasattr(pred[0], "__float__")
            else int(pred[0]),
            probabilities=probs,
            model_version="1.0.0",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.get("/predict/batch")
# async def predict_batch(features: str):
#     """
#     Batch prediction endpoint.

#     Query param: features as comma-separated values (e.g., "1.0,2.0,3.0")
#     """
#     global model

#     if model is None:
#         raise HTTPException(status_code=503, detail="Model not loaded.")

#     try:
#         # Parse comma-separated features
#         feature_list = [float(x) for x in features.split(",")]
#         x = np.array(feature_list).reshape(1, -1)

#         pred = model.predict(x)

#         return {"prediction": float(pred[0])}

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))


# @app.get("/metrics")
# async def metrics():
#     """Prometheus metrics endpoint."""
#     return Response(content=generate_latest(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

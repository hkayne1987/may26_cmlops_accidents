"""
FastAPI Inference Service

Phase 1 deliverable: Basic inference API for ML model serving.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
model: Any = None
model_path = PROJECT_ROOT / "models/xgb_severity.joblib"

# Registry name and alias set by src/training/train.py.
REGISTERED_MODEL_NAME = "xgb_severity"
PRODUCTION_ALIAS = "production"
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}"

# Reported by /health and /predict so callers know which model answered.
model_version: str = "unknown"


def load_model_from_registry() -> tuple[Any, str]:
    """Pulls the production model from the MLflow registry (DagsHub).

    Returns (None, "unknown") when MLFLOW_TRACKING_URI is unset, so the
    caller can fall back to the local file.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        log.info("MLFLOW_TRACKING_URI unset: skipping registry pull")
        return None, "unknown"

    mlflow.set_tracking_uri(tracking_uri)
    start = time.time()
    loaded = mlflow.xgboost.load_model(MODEL_URI)

    # Resolve the concrete version behind the alias for traceability.
    try:
        version = str(mlflow.MlflowClient().get_model_version_by_alias(
            REGISTERED_MODEL_NAME, PRODUCTION_ALIAS
        ).version)
    except Exception:
        version = "unknown"

    log.info(f"Model pulled from registry: {MODEL_URI} (v{version}) "
             f"in {time.time() - start:.1f}s")
    return loaded, version


def load_model_from_disk() -> tuple[Any, str]:
    """Loads the model from the local joblib file (fallback)."""
    if not os.path.exists(model_path):
        return None, "unknown"
    loaded = joblib.load(model_path)
    log.info(f"Model loaded from local file: {model_path}")
    return loaded, "local"


def load_model() -> tuple[Any, str]:
    """Pulls from the MLflow registry, falling back to the local file."""
    try:
        loaded, version = load_model_from_registry()
        if loaded is not None:
            return loaded, version
    except Exception as e:
        log.warning(f"Registry pull failed ({type(e).__name__}: {e}); "
                    f"falling back to local file")
    return load_model_from_disk()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_version
    model, model_version = load_model()
    if model is None:
        log.warning("No model available from registry or local file")
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
    model_version: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_version=model_version,
    )


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

        DECISION_THRESHOLD = 0.30

        # Get probabilities and apply our custom decision threshold
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0].tolist()
            pred = int(probs[1] >= DECISION_THRESHOLD)
        else:
            pred = model.predict(x)[0]

        # PREDICTIONS_MADE.inc()
        return PredictResponse(
            prediction=pred,
            probabilities=probs,
            model_version=model_version,
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

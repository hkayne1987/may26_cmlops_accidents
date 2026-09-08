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
from fastapi import Depends, FastAPI, HTTPException, status

# Kept for the /metrics endpoint at the bottom of this file, still commented out.
from fastapi.responses import Response  # noqa: F401
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.auth import (
    Role,
    TokenResponse,
    User,
    UserInfo,
    authenticate_user,
    create_access_token,
    create_user,
    get_current_user,
    get_secret_key,
    get_session,
    require_role,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
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

# Number of features the loaded model expects, read from the model itself so
# it cannot drift from the served version. None until a model is loaded.
n_features_expected: int | None = None

DECISION_THRESHOLD = 0.30  # decision threshold for the severe class


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


def load_model(attempts: int = 3, backoff: float = 5.0) -> tuple[Any, str]:
    """Pulls from the MLflow registry, falling back to the local file.

    Retries a few times: pulling artifacts from DagsHub occasionally times
    out, and a transient failure should not leave the API without a model.
    """
    for attempt in range(1, attempts + 1):
        try:
            loaded, version = load_model_from_registry()
            if loaded is not None:
                return loaded, version
            break  # tracking URI unset: no point retrying
        except Exception as e:
            log.warning(f"Registry pull failed (attempt {attempt}/{attempts}, "
                        f"{type(e).__name__}: {e})")
            if attempt < attempts:
                time.sleep(backoff)

    log.info("Falling back to local model file")
    return load_model_from_disk()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_version, n_features_expected

    # Fail fast on a missing or weak signing key rather than starting an
    # API whose tokens anyone could forge.
    get_secret_key()

    model, model_version = load_model()
    if model is None:
        log.warning("No model available from registry or local file")
    else:
        n_features_expected = getattr(model, "n_features_in_", None)
        log.info(f"Model expects {n_features_expected} features")
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


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=12, max_length=72)
    role: Role = Role.OPERATOR


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.

    Left unauthenticated on purpose: Docker and the future reverse proxy
    probe it, and it exposes no data beyond which model version is served.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_version=model_version,
    )


@app.post("/token", response_model=TokenResponse, tags=["auth"])
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session),
):
    """Exchanges a username and password for a JWT access token."""
    user = authenticate_user(session, form_data.username, form_data.password)
    if user is None:
        # Same message whether the user is unknown, disabled or the password
        # is wrong: do not tell an attacker which usernames exist.
        log.warning(f"Failed login attempt for {form_data.username!r}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token, expires_in = create_access_token(user.username, user.role)
    log.info(f"Token issued to {user.username} (role={user.role})")
    return TokenResponse(access_token=token, expires_in=expires_in)


@app.get("/me", response_model=UserInfo, tags=["auth"])
async def read_current_user(user: User = Depends(get_current_user)):
    """Returns the account behind the current token."""
    return UserInfo(username=user.username, role=user.role, is_active=user.is_active)


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(
    request: PredictRequest,
    user: User = Depends(require_role(Role.OPERATOR, Role.ADMIN)),
):
    """Makes a prediction using the loaded model. Requires a valid token."""
    global model

    # INFERENCE_REQUESTS.inc()

    if model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Train a model first."
        )

    # Check the input size before inference: a wrong length otherwise fails
    # deep inside XGBoost and surfaces as an opaque 500.
    if n_features_expected is not None and len(request.features) != n_features_expected:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Expected {n_features_expected} features, "
                f"got {len(request.features)}"
            ),
        )

    try:
        # Reshape for single prediction
        x = np.array(request.features).reshape(1, -1)

        # Get probabilities and apply our custom decision threshold
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0].tolist()
            pred = int(probs[1] >= DECISION_THRESHOLD)
        else:
            pred = model.predict(x)[0]

        # PREDICTIONS_MADE.inc()
        log.info(f"Prediction by {user.username}: {pred} (model v{model_version})")
        return PredictResponse(
            prediction=pred,
            probabilities=probs,
            model_version=model_version,
        )

    except HTTPException:
        raise
    except Exception:
        # Log the details but do not return them: an internal traceback can
        # leak paths and library versions to the caller. log.exception
        # records the full traceback server-side.
        log.exception(f"Prediction failed for {user.username}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/admin/users", response_model=UserInfo, status_code=201, tags=["admin"])
async def add_user(
    request: CreateUserRequest,
    admin: User = Depends(require_role(Role.ADMIN)),
    session: Session = Depends(get_session),
):
    """Creates an account. Admin only."""
    try:
        user = create_user(session, request.username, request.password, request.role)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    log.info(f"{admin.username} created account {user.username}")
    return UserInfo(username=user.username, role=user.role, is_active=user.is_active)


@app.get("/admin/users", response_model=list[UserInfo], tags=["admin"])
async def list_users(
    admin: User = Depends(require_role(Role.ADMIN)),
    session: Session = Depends(get_session),
):
    """Lists all accounts. Admin only."""
    users = session.scalars(select(User).order_by(User.username)).all()
    return [
        UserInfo(username=u.username, role=u.role, is_active=u.is_active)
        for u in users
    ]


@app.delete("/admin/users/{username}", response_model=UserInfo, tags=["admin"])
async def deactivate_user(
    username: str,
    admin: User = Depends(require_role(Role.ADMIN)),
    session: Session = Depends(get_session),
):
    """Disables an account. Admin only.

    The row is kept rather than deleted so past predictions stay attributable.
    """
    if username == admin.username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot deactivate your own account",
        )

    user = session.get(User, username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    user.is_active = False
    session.commit()
    log.info(f"{admin.username} deactivated account {username}")
    return UserInfo(username=user.username, role=user.role, is_active=user.is_active)


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

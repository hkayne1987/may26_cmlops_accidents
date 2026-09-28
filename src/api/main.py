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
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import Response
from fastapi.security import OAuth2PasswordRequestForm
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api import metrics
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
from src.api.schema import (
    FeatureSchema,
    InvalidFeaturesError,
    describe,
    schema_from_model,
    to_model_input,
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

# Columns and allowed category values of the loaded model, read from the model
# itself so they always match the served version. None until a model loads.
feature_schema: FeatureSchema | None = None

DECISION_THRESHOLD = 0.35  # decision threshold for the severe class


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
        version = str(
            mlflow.MlflowClient()
            .get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
            .version
        )
    except Exception:
        version = "unknown"

    log.info(
        f"Model pulled from registry: {MODEL_URI} (v{version}) "
        f"in {time.time() - start:.1f}s"
    )
    return loaded, version


def load_model_from_disk() -> tuple[Any, str]:
    """Loads the model from the local joblib file (fallback).

    An interrupted training run can leave an empty or truncated file here, so
    report no model rather than crashing the API on a corrupt one.
    """
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        return None, "unknown"
    try:
        loaded = joblib.load(model_path)
    except Exception as e:
        log.error(f"Local model file is unusable ({type(e).__name__}): {model_path}")
        return None, "unknown"
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
            log.warning(
                f"Registry pull failed (attempt {attempt}/{attempts}, "
                f"{type(e).__name__}: {e})"
            )
            if attempt < attempts:
                time.sleep(backoff)

    log.info("Falling back to local model file")
    return load_model_from_disk()


def refresh_model() -> None:
    """Loads the production model and updates the module state.

    Shared by startup and the reload endpoint so both go through exactly the
    same path.
    """
    global model, model_version, feature_schema

    start = time.time()
    model, model_version = load_model()
    metrics.MODEL_LOAD_SECONDS.set(time.time() - start)

    feature_schema = None
    if model is None:
        log.warning("No model available from registry or local file")
    else:
        try:
            feature_schema = schema_from_model(model)
            log.info(
                f"Model expects {len(feature_schema.names)} features, "
                f"{len(feature_schema.categories)} of them categorical"
            )
        except Exception:
            # Without its schema the model cannot read real values, so serving
            # it would only produce wrong predictions: refuse instead.
            log.exception("Could not read the input schema from the model")
            model = None

    metrics.set_served_model(model_version, loaded=model is not None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast on a missing or weak signing key rather than starting an
    # API whose tokens anyone could forge.
    get_secret_key()

    refresh_model()
    yield


app = FastAPI(
    title="ML Inference API",
    description="Production inference service for ML models",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def record_http_metrics(request: Request, call_next):
    """Counts every request and times it, labelled by route template."""
    start = time.time()
    response = await call_next(request)

    # The matched route is only known once routing ran. Unmatched paths share
    # one label so random URLs cannot create new time series.
    route = request.scope.get("route")
    path = getattr(route, "path", "unmatched")
    if path != "/metrics":  # Prometheus scraping itself is not traffic
        metrics.HTTP_REQUESTS.labels(
            request.method, path, str(response.status_code)
        ).inc()
        metrics.HTTP_LATENCY.labels(request.method, path).observe(time.time() - start)
    return response


# A real severe accident from the test set: a driver in a car on an 80 km/h
# road in Haute-Corse, in daylight. Shown in the Swagger UI as the example.
EXAMPLE_FEATURES = {
    "place": "1", "catu": "1", "sexe": "2", "secu1": "1", "secu2": "-1",
    "secu3": "-1", "locp": "0", "actp": "0", "etatp": "-1", "senc": "2",
    "catv": "7", "obs": "2", "obsm": "0", "choc": "3", "manv": "13",
    "occutc": None, "jour": 19, "mois": 6, "an": 2023, "lum": "1",
    "dep": "2B", "com": "2B193", "agg": "1", "int": "1", "atm": "1",
    "col": "6", "lat": 42.37663, "long": 9.18949, "catr": "2", "circ": "2",
    "nbv": 2, "vosp": "0", "prof": "1", "plan": "3", "surf": "1",
    "infra": "0", "situ": "3", "vma": 80, "heure": 16, "minute": 57,
}  # fmt: skip


# Request/Response models
class PredictRequest(BaseModel):
    features: dict[str, str | int | float | None] = Field(
        ...,
        description=(
            'Real BAAC values keyed by column name, e.g. "catv": "7" for '
            "a car. Categorical codes may be strings or numbers; null means "
            "unknown. GET /model/schema lists the expected columns and values."
        ),
        examples=[EXAMPLE_FEATURES],
    )


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
        metrics.LOGIN_ATTEMPTS.labels(result="failure").inc()
        log.warning(f"Failed login attempt for {form_data.username!r}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token, expires_in = create_access_token(user.username, user.role)
    metrics.LOGIN_ATTEMPTS.labels(result="success").inc()
    log.info(f"Token issued to {user.username} (role={user.role})")
    return TokenResponse(access_token=token, expires_in=expires_in)


@app.get("/me", response_model=UserInfo, tags=["auth"])
async def read_current_user(user: User = Depends(get_current_user)):
    """Returns the account behind the current token."""
    return UserInfo.from_user(user)


@app.get("/model/schema", tags=["inference"])
async def model_schema(
    user: User = Depends(require_role(Role.OPERATOR, Role.SERVICE, Role.ADMIN)),
):
    """Lists the columns /predict expects and the values each one accepts.

    Read from the served model, so it always matches the current version.
    """
    if feature_schema is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"model_version": model_version, "features": describe(feature_schema)}


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(
    request: PredictRequest,
    user: User = Depends(require_role(Role.OPERATOR, Role.SERVICE, Role.ADMIN)),
):
    """Makes a prediction using the loaded model. Requires a valid token."""
    global model

    if model is None or feature_schema is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Train a model first."
        )

    # Validate against the served model before inference: an unknown column
    # or category value otherwise fails deep inside XGBoost as an opaque 500.
    try:
        x = to_model_input(request.features, feature_schema)
    except InvalidFeaturesError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=e.errors
        )

    try:
        # Get probabilities and apply our custom decision threshold
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0].tolist()
            pred = int(probs[1] >= DECISION_THRESHOLD)
            metrics.SEVERE_PROBABILITY.observe(probs[1])
        else:
            pred = model.predict(x)[0]

        metrics.PREDICTIONS.labels(
            outcome="severe" if pred == 1 else "non_severe",
            model_version=model_version,
        ).inc()
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
    return UserInfo.from_user(user)


@app.post("/admin/reload-model", response_model=HealthResponse, tags=["admin"])
async def reload_model(
    caller: User = Depends(require_role(Role.SERVICE, Role.ADMIN)),
):
    """Reloads the production model from the registry, without a restart.

    Called by the Airflow pipeline once a new version clears the release
    gate, so operators get the new model with no downtime.
    """
    previous = model_version
    refresh_model()
    metrics.MODEL_RELOADS.labels(
        result="success" if model is not None else "no_model"
    ).inc()
    log.info(f"{caller.username} reloaded the model: v{previous} -> v{model_version}")
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_version=model_version,
    )


@app.get("/admin/users", response_model=list[UserInfo], tags=["admin"])
async def list_users(
    admin: User = Depends(require_role(Role.ADMIN)),
    session: Session = Depends(get_session),
):
    """Lists all accounts. Admin only."""
    users = session.scalars(select(User).order_by(User.username)).all()
    return [UserInfo.from_user(u) for u in users]


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
    return UserInfo.from_user(user)


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


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Prometheus scrape endpoint.

    Unauthenticated because Prometheus scrapes it from inside the Docker
    network. It must not be exposed publicly: the reverse proxy should not
    forward /metrics.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

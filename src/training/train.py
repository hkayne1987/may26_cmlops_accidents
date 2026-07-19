"""
Training of the accident severity model (XGBoost).
Reads the preprocessed dataset (data/processed/) and performs a grouped split by accident.
Trains and saves the model to models/.

Run via 'python -m src.training.train'.
"""

import logging
import os
from pathlib import Path

import pandas as pd

from src.data.preprocess import build_feature_matrix

from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier
import joblib
import mlflow
import mlflow.xgboost

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TRAIN_PATH = Path("data/processed/train.parquet")
MODEL_DIR = Path("models")
TEST_SIZE = 0.2
RANDOM_STATE = 0
DECISION_THRESHOLD = 0.30  # decision threshold for the severe class

REGISTERED_MODEL_NAME = "xgb_severity"
PRODUCTION_ALIAS = "production"
MODEL_FILENAME = f"{REGISTERED_MODEL_NAME}.joblib"
RUN_ID_PATH = MODEL_DIR / "run_id.txt"  # read by evaluate.py to resume this run

# Hyperparameters retained after a GridSearchCV (scoring="f1", 3-fold StratifiedGroupKFold) run on year 2023 only (~125k users).
# Tested grid: max_depth=[3, 5, 7], learning_rate=[0.05, 0.1, 0.2], n_estimators=[50, 100, 200, 300].
# Best F1 score obtained: 0.586.
# See tune_hyperparameters() to reproduce or re-optimize on the full 2019-2024 dataset.
HYPERPARAMS = {
    "max_depth": 3,
    "learning_rate": 0.2,
    "n_estimators": 300,
}


def load_train() -> pd.DataFrame:
    """Loads the preprocessed training set."""
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training set not found: {TRAIN_PATH}. "
            f"Run `python -m src.data.preprocess` first."
        )
    df = pd.read_parquet(TRAIN_PATH)
    log.info(f"Training set loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def train_model(X_train, y_train) -> XGBClassifier:
    """Trains XGBoost with class imbalance handling (scale_pos_weight)."""
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    log.info(f"scale_pos_weight = {scale:.2f}")

    model = XGBClassifier(
        **HYPERPARAMS,
        scale_pos_weight=scale,
        enable_categorical=True,
        tree_method="hist",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    log.info("Training complete.")
    return model


def tune_hyperparameters(X_train, y_train, groups_train):
    """Hyperparameter search via grid search (optional, not run by default).
    Takes time.
    Usage: python -m src.training.train --tune
    """
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

    scale = (y_train == 0).sum() / (y_train == 1).sum()
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "n_estimators": [50, 100, 200, 300],
    }
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        XGBClassifier(scale_pos_weight=scale, enable_categorical=True, tree_method="hist",
                      eval_metric="logloss", random_state=RANDOM_STATE),
        param_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=2,
    )
    grid.fit(X_train, y_train, groups=groups_train)

    log.info(f"Best parameters: {grid.best_params_}")
    log.info(f"Best F1 score: {grid.best_score_:.4f}")
    return grid.best_params_


def save_model(model, X_train):
    """Saves the model + the ordered list of columns expected at inference."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / MODEL_FILENAME
    joblib.dump(model, model_path)
    log.info(f"Model saved: {model_path}")

    # Also save the column schema
    # Required for the API which must reconstruct a DataFrame with exactly these columns
    schema_path = MODEL_DIR / "feature_columns.json"
    pd.Series(X_train.columns).to_json(schema_path, orient="values")
    log.info(f"Feature schema saved: {schema_path}")


def get_data_version() -> str:
    """Returns the DVC data version for lineage tracking.

    Placeholder until the DVC pipeline is in place (tracked separately by the team).
    """
    return os.environ.get("DATA_VERSION", "pending-dvc")


def log_mlflow_run(model, X_train) -> str:
    """Logs the training run to MLflow: hyperparameters, data lineage, and the
    model registered under REGISTERED_MODEL_NAME with a 'production' alias + tag.

    Returns the MLflow run_id so evaluate.py can resume this same run to log metrics.
    """
    # MLflow >=3 puts the filesystem backend in maintenance mode by default;
    # opt back in since the local fallback (file:./mlruns) relies on it.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))

    with mlflow.start_run() as run:
        mlflow.log_params(HYPERPARAMS)
        mlflow.log_param("decision_threshold", DECISION_THRESHOLD)
        mlflow.log_param("data_version", get_data_version())

        model_info = mlflow.xgboost.log_model(
            model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        client = mlflow.MlflowClient()
        version = model_info.registered_model_version
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME, alias=PRODUCTION_ALIAS, version=version,
        )
        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME, version=version, key="stage", value="production",
        )
        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME, version=version,
            key="decision_threshold", value=str(DECISION_THRESHOLD),
        )

        log.info(f"MLflow run {run.info.run_id}: registered {REGISTERED_MODEL_NAME} "
                 f"v{version} (alias={PRODUCTION_ALIAS})")
        return run.info.run_id


if __name__ == "__main__":
    import sys

    df = load_train()

    if "--sample" in sys.argv:
        df = df.sample(n=10000, random_state=RANDOM_STATE)
        log.info(f"--sample mode: reduced to {len(df)} rows")

    X, y, groups = build_feature_matrix(df)
    log.info(f"X={X.shape}, y={y.shape}")

    if "--tune" in sys.argv:
        best_params = tune_hyperparameters(X, y, groups)
        log.info(f"Best parameters: {best_params}")
    else:
        model = train_model(X, y)
        save_model(model, X)

        run_id = log_mlflow_run(model, X)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        RUN_ID_PATH.write_text(run_id)
        log.info(f"MLflow run_id saved: {RUN_ID_PATH} (used by evaluate.py)")

        log.info("Training complete.")

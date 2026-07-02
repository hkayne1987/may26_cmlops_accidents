"""
Training of the accident severity model (XGBoost).
Reads the preprocessed dataset (data/processed/) and performs a grouped split by accident.
Trains and saves the model to models/.

Run via 'python -m src.training.train'.
"""

import logging
from pathlib import Path

import pandas as pd

from src.data.preprocess import build_feature_matrix

from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TRAIN_PATH = Path("data/processed/train.parquet")
MODEL_DIR = Path("models")
TEST_SIZE = 0.2
RANDOM_STATE = 0

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
    model_path = MODEL_DIR / "xgb_severity.joblib"
    joblib.dump(model, model_path)
    log.info(f"Model saved: {model_path}")

    # Also save the column schema
    # Required for the API which must reconstruct a DataFrame with exactly these columns
    schema_path = MODEL_DIR / "feature_columns.json"
    pd.Series(X_train.columns).to_json(schema_path, orient="values")
    log.info(f"Feature schema saved: {schema_path}")


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
        log.info("Training complete.")

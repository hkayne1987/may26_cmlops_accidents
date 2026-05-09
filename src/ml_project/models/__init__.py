"""Model training and inference."""

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model: BaseEstimator,
    **kwargs
) -> BaseEstimator:
    """Train a sklearn model."""
    model.fit(X, y)
    return model


def predict(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    """Make predictions with trained model."""
    return model.predict(X)


def save_model(model: BaseEstimator, path: str) -> None:
    """Save model to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> BaseEstimator:
    """Load model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
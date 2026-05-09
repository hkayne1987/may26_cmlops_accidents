"""Data loading and processing utilities."""

import pandas as pd
from pathlib import Path


def load_data(path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv(path)


def save_data(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
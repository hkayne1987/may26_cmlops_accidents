"""Log of the predictions served by the API, the input of drift detection.

Every call to /predict is recorded with the real values received. Labels
(whether the accident actually turned out severe) only arrive once a year
with the next BAAC release, so these inputs are the only signal available in
between that the model is being asked about data unlike its training set.

Stored in its own SQLite file, apart from user accounts, under data/ which is
mounted from the host so the log survives container recreation.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "predictions.db"


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    model_version: Mapped[str] = mapped_column(String(32))
    username: Mapped[str] = mapped_column(String(64))
    severe_probability: Mapped[float] = mapped_column(Float)
    prediction: Mapped[int] = mapped_column(Integer)
    # Kept as JSON rather than one column per feature: the values stay
    # readable, and a retrained model with different columns needs no
    # migration.
    features: Mapped[str] = mapped_column(Text)


_engine = None


def get_engine(db_path: str | None = None):
    """Creates the SQLite engine and the table if needed."""
    path = db_path or os.environ.get("PREDICTIONS_DB", str(DEFAULT_DB_PATH))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{path}",
        # FastAPI may write from different threads of its pool.
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return engine


def _default_engine():
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


def record_prediction(
    features: dict[str, Any],
    severe_probability: float,
    prediction: int,
    model_version: str,
    username: str,
    engine=None,
) -> None:
    """Appends one prediction to the log."""
    with Session(engine or _default_engine()) as session:
        session.add(
            PredictionRecord(
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                model_version=model_version,
                username=username,
                severe_probability=severe_probability,
                prediction=prediction,
                features=json.dumps(features),
            )
        )
        session.commit()


def load_predictions(since: datetime | None = None, engine=None) -> pd.DataFrame:
    """Returns logged predictions as a frame, one column per feature.

    `since` is a naive UTC datetime. Feature values come back as sent: the
    drift job types them against the training data.
    """
    query = select(PredictionRecord).order_by(PredictionRecord.created_at)
    if since is not None:
        query = query.where(PredictionRecord.created_at >= since)

    with Session(engine or _default_engine()) as session:
        records = session.scalars(query).all()

    if not records:
        return pd.DataFrame()

    features = pd.DataFrame([json.loads(r.features) for r in records])
    meta = pd.DataFrame(
        {
            "created_at": [r.created_at for r in records],
            "model_version": [r.model_version for r in records],
            "severe_probability": [r.severe_probability for r in records],
            "prediction": [r.prediction for r in records],
        }
    )
    return pd.concat([meta, features], axis=1)

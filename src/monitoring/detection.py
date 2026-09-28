"""Data drift detection on the predictions served by the API.

Compares the requests logged by the API (src/api/prediction_log.py) over a
recent window with a sample of the training data, using Evidently, then:

- saves the Evidently HTML report under reports/drift/,
- pushes the results to the Prometheus Pushgateway so Grafana can show them.

Why drift on requests: labels (whether an accident really was severe) only
arrive once a year with the next BAAC release. Until then, a change in what
operators send is the only early sign that the model may be degrading.

Drift raises an alert; it does not trigger retraining. Retraining needs new
labelled data, and retraining on the same data would give the same model.
See the Monitoring section of the README for the decision process.

Run: python -m src.monitoring.detection
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from src.api.prediction_log import load_predictions
from src.data.preprocess import build_feature_matrix

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

TRAIN_PATH = Path("data/processed/train.parquet")
REPORT_DIR = Path("reports/drift")

# Every setting can be overridden from the environment.
WINDOW_DAYS = int(os.environ.get("DRIFT_WINDOW_DAYS", "7"))
MIN_ROWS = int(os.environ.get("DRIFT_MIN_ROWS", "200"))
REFERENCE_ROWS = int(os.environ.get("DRIFT_REFERENCE_ROWS", "20000"))
# Share of drifted columns at which the dashboard raises the alert. Lower than
# Evidently's 0.5 "dataset drift" default: for an emergency service, a quarter
# of the inputs shifting is already worth a look.
ALERT_SHARE = float(os.environ.get("DRIFT_ALERT_SHARE", "0.25"))
PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL", "")

# Left out of the comparison:
# - com: ~24k communes, too sparse for any test to say something reliable.
# - an: the accident year always moves forward, so it would always "drift".
EXCLUDED_COLUMNS = {"com", "an"}

RANDOM_STATE = 0


@dataclass
class DriftResult:
    current_rows: int
    columns: list[str] = field(default_factory=list)
    # column -> (method, score, threshold, drifted)
    per_column: dict[str, tuple[str, float, float, bool]] = field(default_factory=dict)
    report_path: Path | None = None

    @property
    def drifted(self) -> list[str]:
        return [c for c, (_, _, _, d) in self.per_column.items() if d]

    @property
    def drift_share(self) -> float:
        return len(self.drifted) / len(self.columns) if self.columns else 0.0

    @property
    def enough_data(self) -> bool:
        return bool(self.columns)

    @property
    def alert(self) -> bool:
        return self.enough_data and self.drift_share >= ALERT_SHARE


def load_reference(n_rows: int = REFERENCE_ROWS) -> pd.DataFrame:
    """A sample of the training data, typed as the model saw it."""
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"{TRAIN_PATH} not found: pull the data first "
            "(docker-compose --profile dvc run --rm dvc pull)"
        )
    df = pd.read_parquet(TRAIN_PATH)
    df = df.sample(min(n_rows, len(df)), random_state=RANDOM_STATE)
    X, _, _ = build_feature_matrix(df)
    return X


def _align(
    reference: pd.DataFrame, current: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Types both frames the same way: categories as text, numerics as floats.

    Missing values stay missing on both sides; the training categories would
    otherwise turn them into the text "nan".
    """
    columns = [
        c
        for c in reference.columns
        if c in current.columns and c not in EXCLUDED_COLUMNS
    ]
    # A column empty on either side cannot be compared, and Evidently refuses
    # it. It happens in practice: occutc (public transport occupants) is null
    # for almost every accident, and entirely null in many windows.
    empty = [c for c in columns if reference[c].isna().all() or current[c].isna().all()]
    if empty:
        log.info(f"Skipping columns empty in the reference or window: {empty}")
    columns = [c for c in columns if c not in empty]
    categorical = [c for c in columns if str(reference[c].dtype) == "category"]
    numerical = [c for c in columns if c not in categorical]

    def typed(frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        for c in categorical:
            values = frame[c].astype(object)
            out[c] = values.where(values.notna(), None).map(
                lambda v: None if v is None else str(v)
            )
        for c in numerical:
            out[c] = pd.to_numeric(frame[c], errors="coerce").astype(float)
        return out

    return typed(reference), typed(current), categorical, numerical


def _is_drifted(method: str, score: float, threshold: float) -> bool:
    """Evidently's rule: p-values drift when low, distances when high."""
    if "p_value" in method:
        return score < threshold
    return score >= threshold


def detect(
    reference: pd.DataFrame, current: pd.DataFrame, report_dir: Path | None = None
) -> DriftResult:
    """Runs Evidently on the two frames and summarises the result."""
    result = DriftResult(current_rows=len(current))
    if len(current) < MIN_ROWS:
        log.warning(
            f"Only {len(current)} predictions in the window, need {MIN_ROWS}: "
            "not enough to conclude on drift"
        )
        return result

    ref, cur, categorical, numerical = _align(reference, current)
    definition = DataDefinition(
        numerical_columns=numerical, categorical_columns=categorical
    )
    # Same threshold as the Grafana alert, so the HTML report's "dataset
    # drift" verdict agrees with the dashboard (Evidently defaults to 0.5).
    snapshot = Report([DataDriftPreset(drift_share=ALERT_SHARE)]).run(
        current_data=Dataset.from_pandas(cur, data_definition=definition),
        reference_data=Dataset.from_pandas(ref, data_definition=definition),
    )

    for metric in snapshot.dict()["metrics"]:
        config = metric["config"]
        if "ValueDrift" not in config.get("type", ""):
            continue
        method, score, threshold = (
            config["method"],
            metric["value"],
            config["threshold"],
        )
        result.per_column[config["column"]] = (
            method,
            float(score),
            float(threshold),
            _is_drifted(method, float(score), float(threshold)),
        )
    result.columns = list(result.per_column)

    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        result.report_path = report_dir / f"drift_report_{stamp}.html"
        snapshot.save_html(str(result.report_path))
        # Stable name for "the latest report", handy to open by hand.
        snapshot.save_html(str(report_dir / "drift_report_latest.html"))

    return result


def push_metrics(result: DriftResult, gateway: str) -> None:
    """Publishes the result to the Pushgateway, replacing the previous run."""
    registry = CollectorRegistry()

    def gauge(name: str, doc: str, labels: tuple[str, ...] = ()) -> Gauge:
        return Gauge(name, doc, labels, registry=registry)

    gauge("drift_current_rows", "Predictions compared in the last run.").set(
        result.current_rows
    )
    gauge(
        "drift_enough_data", "1 when the last run had enough predictions to conclude."
    ).set(int(result.enough_data))
    gauge("drift_columns_checked", "Columns compared in the last run.").set(
        len(result.columns)
    )
    gauge("drift_drifted_columns", "Columns in drift in the last run.").set(
        len(result.drifted)
    )
    gauge("drift_share", "Share of compared columns in drift.").set(result.drift_share)
    gauge("drift_alert_threshold", "Drift share at which the alert fires.").set(
        ALERT_SHARE
    )
    gauge("drift_alert", "1 when the drift share reached the alert threshold.").set(
        int(result.alert)
    )
    gauge(
        "drift_last_run_timestamp_seconds", "When the last drift check ran."
    ).set_to_current_time()

    score = gauge(
        "drift_column_score",
        "Drift score per column (a distance, or a p-value for K-S).",
        ("column", "method"),
    )
    drifted = gauge("drift_column_drifted", "1 if the column drifted.", ("column",))
    for column, (method, value, _, is_drifted) in result.per_column.items():
        score.labels(column, method).set(value)
        drifted.labels(column).set(int(is_drifted))

    push_to_gateway(gateway, job="drift_detection", registry=registry)
    log.info(f"Drift metrics pushed to {gateway}")


def main() -> None:
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        days=WINDOW_DAYS
    )
    current = load_predictions(since=since)
    log.info(f"{len(current)} predictions logged over the last {WINDOW_DAYS} days")

    reference = load_reference()
    result = detect(reference, current, report_dir=REPORT_DIR)

    if result.enough_data:
        log.info(
            f"{len(result.drifted)}/{len(result.columns)} columns in drift "
            f"({result.drift_share:.0%}): {', '.join(result.drifted) or 'none'}"
        )
        if result.alert:
            log.warning(
                f"DRIFT ALERT: {result.drift_share:.0%} of columns drifted, "
                f"threshold {ALERT_SHARE:.0%}"
            )
        log.info(f"Report: {result.report_path}")

    if PUSHGATEWAY_URL:
        push_metrics(result, PUSHGATEWAY_URL)
    else:
        log.info("PUSHGATEWAY_URL unset: metrics not pushed")


if __name__ == "__main__":
    main()

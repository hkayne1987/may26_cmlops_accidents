"""Tests for the prediction log that feeds drift detection."""

from datetime import datetime, timedelta, timezone

from prometheus_client import REGISTRY

from src.api import prediction_log
from tests.api_helpers import FEATURES, OPERATOR_PW, auth_header


def test_each_prediction_is_logged_with_real_values(client, prediction_db):
    client.post(
        "/predict", json=FEATURES, headers=auth_header(client, "operator1", OPERATOR_PW)
    )

    logged = prediction_log.load_predictions(engine=prediction_db)
    assert len(logged) == 1
    row = logged.iloc[0]
    assert row["catv"] == "7"  # the value as sent, not an internal code
    assert row["vma"] == 80
    assert row["model_version"] == "test"
    assert row["severe_probability"] == 0.8
    assert row["prediction"] == 1


def test_rejected_requests_are_not_logged(client, prediction_db):
    client.post(
        "/predict",
        json={"features": {"catv": "999", "lum": "1", "vma": 80}},
        headers=auth_header(client, "operator1", OPERATOR_PW),
    )
    assert prediction_log.load_predictions(engine=prediction_db).empty


def test_a_logging_failure_does_not_block_the_prediction(client, monkeypatch):
    def broken(**kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(prediction_log, "record_prediction", broken)
    before = REGISTRY.get_sample_value("api_prediction_log_failures_total") or 0.0

    response = client.post(
        "/predict", json=FEATURES, headers=auth_header(client, "operator1", OPERATOR_PW)
    )

    assert response.status_code == 200
    assert REGISTRY.get_sample_value("api_prediction_log_failures_total") == before + 1


def test_load_predictions_filters_by_date(prediction_db):
    for _ in range(3):
        prediction_log.record_prediction(
            features={"catv": "7"},
            severe_probability=0.5,
            prediction=1,
            model_version="6",
            username="op",
            engine=prediction_db,
        )

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    assert len(prediction_log.load_predictions(engine=prediction_db)) == 3
    recent = prediction_log.load_predictions(
        since=now - timedelta(minutes=5), engine=prediction_db
    )
    assert len(recent) == 3
    future = prediction_log.load_predictions(
        since=now + timedelta(minutes=5), engine=prediction_db
    )
    assert future.empty

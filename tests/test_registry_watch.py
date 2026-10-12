"""Tests for the API's registry watcher, which reloads a newly promoted model."""

import asyncio

import pytest
from prometheus_client import REGISTRY

from src.api import main as main_module


def run_one_check(monkeypatch, latest: str | None) -> list[str]:
    """Runs a single iteration of watch_registry and returns the reloads."""
    reloads = []
    monkeypatch.setattr(main_module, "production_version", lambda: latest)
    monkeypatch.setattr(main_module, "refresh_model", lambda: reloads.append("x"))

    calls = {"n": 0}

    async def fake_sleep(_):
        calls["n"] += 1
        if calls["n"] > 1:  # stop after the first check
            raise asyncio.CancelledError

    monkeypatch.setattr(main_module.asyncio, "sleep", fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(main_module.watch_registry(0))
    return reloads


def test_a_newly_promoted_version_is_loaded(monkeypatch):
    monkeypatch.setattr(main_module, "model_version", "6")
    monkeypatch.setattr(main_module, "model", object())
    before = (
        REGISTRY.get_sample_value("api_model_reloads_total", {"result": "auto"}) or 0.0
    )

    assert run_one_check(monkeypatch, latest="7") == ["x"]
    after = REGISTRY.get_sample_value("api_model_reloads_total", {"result": "auto"})
    assert after == before + 1


def test_same_version_is_not_reloaded(monkeypatch):
    monkeypatch.setattr(main_module, "model_version", "6")
    assert run_one_check(monkeypatch, latest="6") == []


def test_unreachable_registry_keeps_the_current_model(monkeypatch):
    monkeypatch.setattr(main_module, "model_version", "6")
    assert run_one_check(monkeypatch, latest=None) == []


def test_without_tracking_uri_the_registry_is_not_queried(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert main_module.production_version() is None


# --- What a reload puts in place -------------------------------------------------


def test_health_is_503_without_a_model(client, monkeypatch):
    monkeypatch.setattr(main_module, "model", None)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["model_loaded"] is False


def test_predict_applies_the_threshold_of_the_served_version(client, monkeypatch):
    from tests.api_helpers import FEATURES, OPERATOR_PW, auth_header

    # The stub model gives the severe class 0.8: above 0.35, below 0.9.
    monkeypatch.setattr(main_module, "decision_threshold", 0.9)
    response = client.post(
        "/predict", json=FEATURES, headers=auth_header(client, "operator1", OPERATOR_PW)
    )
    assert response.json()["prediction"] == 0


def test_threshold_falls_back_to_the_training_default(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert main_module.threshold_for("7") == main_module.DEFAULT_DECISION_THRESHOLD
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://unused")
    # The local file has no registry version, hence no tag to read.
    assert main_module.threshold_for("local") == main_module.DEFAULT_DECISION_THRESHOLD


def test_a_failed_reload_never_leaves_a_half_updated_state(monkeypatch):
    # Registered with monkeypatch so the module state is restored afterwards.
    for name in ("model", "feature_schema", "model_version", "decision_threshold"):
        monkeypatch.setattr(main_module, name, getattr(main_module, name))
    monkeypatch.setattr(main_module, "load_model", lambda: (object(), "8"))

    def broken_schema(_):
        raise ValueError("not an XGBoost model")

    monkeypatch.setattr(main_module, "schema_from_model", broken_schema)
    monkeypatch.setattr(main_module, "threshold_for", lambda v: 0.35)
    main_module.refresh_model()
    # A model whose schema cannot be read is refused as a whole.
    assert main_module.model is None
    assert main_module.feature_schema is None

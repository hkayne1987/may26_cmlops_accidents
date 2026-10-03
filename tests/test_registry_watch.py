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

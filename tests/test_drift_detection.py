"""Tests for src/monitoring/detection.py, on synthetic data shaped like BAAC."""

import numpy as np
import pandas as pd
import pytest

from src.monitoring import detection


def make_frame(n: int, seed: int, motorway_share: float = 0.1) -> pd.DataFrame:
    """Frame typed like build_feature_matrix output: categories and floats."""
    rng = np.random.default_rng(seed)
    motorway = rng.random(n) < motorway_share
    return pd.DataFrame(
        {
            "catr": pd.Categorical(np.where(motorway, "1", rng.choice(["3", "4"], n))),
            "lum": pd.Categorical(rng.choice(["1", "2", "5"], n, p=[0.7, 0.2, 0.1])),
            "vma": np.where(motorway, 130.0, rng.choice([30.0, 50.0, 80.0], n)),
            "heure": rng.integers(0, 24, n).astype(float),
            # Never compared: too sparse, and the year always moves forward.
            "com": pd.Categorical(rng.choice(["75056", "13055"], n)),
            "an": np.full(n, 2023.0),
        }
    )


@pytest.fixture(autouse=True)
def small_minimum(monkeypatch):
    monkeypatch.setattr(detection, "MIN_ROWS", 100)


def test_same_distribution_does_not_drift():
    result = detection.detect(make_frame(4000, seed=0), make_frame(800, seed=1))
    assert result.enough_data
    assert result.drifted == []
    assert not result.alert


def test_a_shift_in_the_requests_raises_the_alert():
    """Mostly motorway accidents: road type and speed limit move together."""
    result = detection.detect(
        make_frame(4000, seed=0), make_frame(800, seed=1, motorway_share=0.8)
    )
    assert {"catr", "vma"} <= set(result.drifted)
    assert result.drift_share >= detection.ALERT_SHARE
    assert result.alert


def test_too_few_predictions_do_not_conclude():
    result = detection.detect(make_frame(4000, seed=0), make_frame(50, seed=1))
    assert not result.enough_data
    assert result.current_rows == 50
    assert not result.alert


def test_excluded_columns_are_never_compared():
    result = detection.detect(make_frame(4000, seed=0), make_frame(800, seed=1))
    assert "com" not in result.columns
    assert "an" not in result.columns


def test_a_column_empty_in_the_window_is_skipped():
    """occutc is null for almost every accident; Evidently rejects empty columns."""
    reference = make_frame(4000, seed=0)
    reference["occutc"] = np.where(np.arange(4000) % 50 == 0, 1.0, np.nan)
    current = make_frame(800, seed=1)
    current["occutc"] = np.nan

    result = detection.detect(reference, current)
    assert result.enough_data
    assert "occutc" not in result.columns


def test_requests_logged_as_json_are_typed_like_the_reference():
    """Logged values come back as sent: text codes, numbers, and None."""
    reference = make_frame(4000, seed=0)
    current = make_frame(800, seed=1).astype(object)
    current.loc[:10, "lum"] = None
    current["vma"] = current["vma"].astype(int)

    result = detection.detect(reference, current)
    assert result.enough_data
    assert "lum" in result.columns and "vma" in result.columns


@pytest.mark.parametrize(
    "method, score, threshold, expected",
    [
        ("K-S p_value", 0.01, 0.05, True),  # p-values drift when low
        ("K-S p_value", 0.30, 0.05, False),
        ("Jensen-Shannon distance", 0.15, 0.1, True),  # distances when high
        ("Jensen-Shannon distance", 0.05, 0.1, False),
        ("Wasserstein distance (normed)", 0.1, 0.1, True),  # threshold included
    ],
)
def test_drift_rule_follows_evidently(method, score, threshold, expected):
    assert detection._is_drifted(method, score, threshold) is expected


def test_report_is_written(tmp_path):
    result = detection.detect(
        make_frame(4000, seed=0), make_frame(800, seed=1), report_dir=tmp_path
    )
    assert result.report_path is not None and result.report_path.exists()
    assert (tmp_path / "drift_report_latest.html").exists()


def test_metrics_are_pushed_under_one_job(monkeypatch):
    pushed = {}

    def fake_push(gateway, job, registry):
        pushed["gateway"], pushed["job"] = gateway, job
        pushed["samples"] = {
            (s.name, tuple(sorted(s.labels.items()))): s.value
            for metric in registry.collect()
            for s in metric.samples
        }

    monkeypatch.setattr(detection, "push_to_gateway", fake_push)
    result = detection.detect(
        make_frame(4000, seed=0), make_frame(800, seed=1, motorway_share=0.8)
    )
    detection.push_metrics(result, "pushgateway:9091")

    samples = pushed["samples"]
    assert pushed["job"] == "drift_detection"
    assert samples[("drift_alert", ())] == 1
    assert samples[("drift_share", ())] == pytest.approx(result.drift_share)
    assert samples[("drift_current_rows", ())] == 800
    assert samples[("drift_column_drifted", (("column", "catr"),))] == 1

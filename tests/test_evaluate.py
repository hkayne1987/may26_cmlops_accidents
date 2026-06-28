"""Unit tests for src/training/evaluate.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import training.evaluate as evaluate_module


def test_load_test_missing_file(monkeypatch):
    fake_path = MagicMock()
    fake_path.exists.return_value = False
    monkeypatch.setattr(evaluate_module, "TEST_PATH", fake_path)

    with pytest.raises(FileNotFoundError):
        evaluate_module.load_test()


def test_load_model_missing_file(monkeypatch):
    fake_path = MagicMock()
    fake_path.exists.return_value = False
    monkeypatch.setattr(evaluate_module, "MODEL_PATH", fake_path)

    with pytest.raises(FileNotFoundError):
        evaluate_module.load_model()


def test_load_test_reads_parquet(monkeypatch):
    fake_path = MagicMock()
    fake_path.exists.return_value = True
    monkeypatch.setattr(evaluate_module, "TEST_PATH", fake_path)

    expected = pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [0, 1]})
    monkeypatch.setattr(evaluate_module.pd, "read_parquet", MagicMock(return_value=expected))

    result = evaluate_module.load_test()
    pd.testing.assert_frame_equal(result, expected)


def test_load_model_loads_joblib(monkeypatch):
    fake_path = MagicMock()
    fake_path.exists.return_value = True
    monkeypatch.setattr(evaluate_module, "MODEL_PATH", fake_path)

    expected_model = MagicMock()
    monkeypatch.setattr(evaluate_module.joblib, "load", MagicMock(return_value=expected_model))

    model = evaluate_module.load_model()
    assert model is expected_model


def test_evaluate_metrics_for_threshold():
    model = MagicMock()
    model.predict_proba.return_value = np.array([
        [0.1, 0.9],
        [0.8, 0.2],
        [0.4, 0.6],
        [0.3, 0.7],
    ])

    X_test = pd.DataFrame({"a": [1, 2, 3, 4]})
    y_test = pd.Series([1, 0, 1, 1])

    metrics = evaluate_module.evaluate(model, X_test, y_test, threshold=0.5)

    assert metrics["threshold"] == 0.5
    assert metrics["auc_roc"] >= 0.0
    assert metrics["precision_grave"] >= 0.0
    assert metrics["recall_grave"] >= 0.0
    assert metrics["f1_grave"] >= 0.0


def test_threshold_search_returns_reliable_rows():
    model = MagicMock()
    model.predict_proba.return_value = np.array([
        [0.1, 0.9],
        [0.8, 0.2],
        [0.4, 0.6],
        [0.3, 0.7],
    ])

    X_test = pd.DataFrame({"a": [1, 2, 3, 4]})
    y_test = pd.Series([1, 0, 1, 1])

    rows = evaluate_module.threshold_search(model, X_test, y_test)

    assert isinstance(rows, list)
    assert len(rows) > 0
    assert all("threshold" in row for row in rows)
    assert all("precision" in row for row in rows)
    assert all("recall" in row for row in rows)
    assert all("f1" in row for row in rows)

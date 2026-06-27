"""Unit tests for src/training/train.py."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Provide a fake xgboost module when it is not installed.
xgboost_module = types.ModuleType("xgboost")
xgboost_module.XGBClassifier = MagicMock()
sys.modules.setdefault("xgboost", xgboost_module)

import training.train as train_module


def test_load_train_missing_file(monkeypatch):
    fake_path = MagicMock()
    fake_path.exists.return_value = False
    monkeypatch.setattr(train_module, "TRAIN_PATH", fake_path)

    with pytest.raises(FileNotFoundError):
        train_module.load_train()


@patch.object(train_module, "XGBClassifier")
def test_train_model_trains(mock_xgb):
    X_train = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])

    mock_model = MagicMock()
    mock_xgb.return_value = mock_model

    model = train_module.train_model(X_train, y_train)

    mock_xgb.assert_called_once()
    mock_model.fit.assert_called_once_with(X_train, y_train)
    assert model is mock_model

    call_kwargs = mock_xgb.call_args.kwargs
    assert call_kwargs["enable_categorical"] is True
    assert call_kwargs["tree_method"] == "hist"
    assert call_kwargs["eval_metric"] == "logloss"
    assert call_kwargs["random_state"] == train_module.RANDOM_STATE


@patch.object(train_module, "XGBClassifier")
def test_train_model_scale_pos_weight(mock_xgb):
    X_train = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    y_train = pd.Series([0, 0, 1, 1])

    mock_model = MagicMock()
    mock_xgb.return_value = mock_model

    train_module.train_model(X_train, y_train)

    call_kwargs = mock_xgb.call_args.kwargs
    assert call_kwargs["scale_pos_weight"] == 1.0


def test_save_model_writes_files(monkeypatch, tmp_path):
    model = MagicMock()
    X_train = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path)

    dump_mock = MagicMock()
    monkeypatch.setattr(train_module.joblib, "dump", dump_mock)

    to_json_mock = MagicMock()
    monkeypatch.setattr(train_module.pd.Series, "to_json", to_json_mock)

    train_module.save_model(model, X_train)

    dump_mock.assert_called_once()
    assert to_json_mock.called


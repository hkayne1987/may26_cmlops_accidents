"""Tests for src/api/schema.py, against a real (tiny) XGBoost model.

The point of the module is that callers send real BAAC values and still get
exactly the prediction the model gives on its own training-time encoding, so
these tests use a real model rather than a stub.
"""

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.api.schema import (
    MAX_LISTED_VALUES,
    FeatureSchema,
    InvalidFeaturesError,
    describe,
    schema_from_model,
    to_model_input,
)


@pytest.fixture(scope="module")
def trained():
    """A small model on BAAC-like columns, categories sorted as text."""
    rng = np.random.default_rng(0)
    n = 400
    catv = rng.choice(["-1", "1", "7", "16", "33"], n)
    lum = rng.choice(["1", "2", "3", "5"], n)
    vma = rng.choice([30.0, 50.0, 80.0, 110.0], n)
    y = ((catv == "33") | (vma > 90)).astype(int)
    X = pd.DataFrame(
        {
            # Same typing as preprocess.py: string values as categories.
            "catv": pd.Categorical(catv),
            "lum": pd.Categorical(lum),
            "vma": vma,
        }
    )
    model = XGBClassifier(
        n_estimators=20, max_depth=3, enable_categorical=True, tree_method="hist"
    )
    model.fit(X, y)
    return model, X


def request_for(X: pd.DataFrame, i: int) -> dict:
    row = X.iloc[i]
    return {"catv": str(row["catv"]), "lum": str(row["lum"]), "vma": float(row["vma"])}


def test_schema_is_read_from_the_model(trained):
    model, _ = trained
    schema = schema_from_model(model)
    assert schema.names == ["catv", "lum", "vma"]
    assert set(schema.categories) == {"catv", "lum"}
    assert schema.numeric == ["vma"]
    assert "7" in schema.categories["catv"]


def test_real_values_give_the_training_time_prediction(trained):
    """The core guarantee: no manual encoding, same prediction."""
    model, X = trained
    schema = schema_from_model(model)
    reference = model.predict_proba(X)[:, 1]

    for i in range(len(X)):
        p = model.predict_proba(to_model_input(request_for(X, i), schema))[0, 1]
        assert p == pytest.approx(reference[i], abs=1e-6)


def test_category_codes_as_numbers_are_accepted(trained):
    model, X = trained
    schema = schema_from_model(model)
    as_text = to_model_input({"catv": "7", "lum": "1", "vma": 50}, schema)
    as_number = to_model_input({"catv": 7, "lum": 1.0, "vma": 50}, schema)
    assert model.predict_proba(as_text)[0, 1] == model.predict_proba(as_number)[0, 1]


def test_null_values_are_passed_as_missing(trained):
    model, _ = trained
    schema = schema_from_model(model)
    frame = to_model_input({"catv": None, "lum": "1", "vma": None}, schema)
    assert frame["catv"].isna().all() and frame["vma"].isna().all()
    model.predict_proba(frame)  # must not raise


def test_unknown_category_is_rejected_with_allowed_values(trained):
    model, _ = trained
    schema = schema_from_model(model)
    with pytest.raises(InvalidFeaturesError) as exc:
        to_model_input({"catv": "999", "lum": "1", "vma": 50}, schema)
    assert "catv" in str(exc.value) and "allowed" in str(exc.value)


def test_all_problems_are_collected(trained):
    model, _ = trained
    schema = schema_from_model(model)
    with pytest.raises(InvalidFeaturesError) as exc:
        to_model_input({"catv": "999", "vma": "fast", "speed": 1}, schema)
    assert len(exc.value.errors) == 4  # missing lum, unknown speed, catv, vma


def test_booleans_are_not_category_values(trained):
    model, _ = trained
    schema = schema_from_model(model)
    with pytest.raises(InvalidFeaturesError):
        to_model_input({"catv": True, "lum": "1", "vma": 50}, schema)


def test_long_value_lists_are_summarised():
    many = [str(i) for i in range(MAX_LISTED_VALUES + 1)]
    schema = FeatureSchema(names=["com", "lum"], categories={"com": many, "lum": ["1"]})

    described = {f["name"]: f for f in describe(schema)}
    assert "values" not in described["com"]
    assert described["com"]["n_values"] == len(many)
    assert described["lum"]["values"] == ["1"]

    with pytest.raises(InvalidFeaturesError) as exc:
        to_model_input({"com": "x", "lum": "1"}, schema)
    assert "GET /model/schema" in str(exc.value)

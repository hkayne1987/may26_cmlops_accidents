"""Tests for src/monitoring/replay.py."""

import json

import numpy as np
import pandas as pd

from src.monitoring import replay


def test_requests_carry_real_values_as_json_types():
    rows = pd.DataFrame(
        {
            "catv": pd.Categorical(["7", "33"]),
            "occutc": [np.nan, 2.0],
            "vma": [80.0, 50.0],
            "lat": [42.37663, 48.85],
        }
    )
    requests = replay.build_requests(rows)

    assert requests[0] == {"catv": "7", "occutc": None, "vma": 80, "lat": 42.37663}
    assert requests[1]["occutc"] == 2
    # What the API receives must be plain JSON.
    json.dumps(requests)


def test_scenarios_use_columns_the_model_knows():
    """A scenario on a misspelled column would silently replay nothing."""
    features = set(json.load(open("models/feature_columns.json")))
    for column, _ in replay.SCENARIOS.values():
        assert column == "" or column in features

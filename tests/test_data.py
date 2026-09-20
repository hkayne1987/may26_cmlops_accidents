"""Placeholder tests — expand as you implement src/common and src/training."""

import pandas as pd


def test_dataframe_roundtrip(tmp_path):
    """Basic sanity check: DataFrame survives a CSV write/read cycle."""
    csv_file = tmp_path / "data.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df.to_csv(csv_file, index=False)

    result = pd.read_csv(csv_file)
    pd.testing.assert_frame_equal(result, df)

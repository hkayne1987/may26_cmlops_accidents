"""Tests for data module."""

import pandas as pd
import pytest

from ml_project.data import load_data, save_data


def test_load_data(tmp_path):
    """Test loading data from CSV."""
    csv_file = tmp_path / "data.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df.to_csv(csv_file, index=False)

    result = load_data(str(csv_file))
    pd.testing.assert_frame_equal(result, df)


def test_save_data(tmp_path):
    """Test saving data to CSV."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    output_path = tmp_path / "output" / "data.csv"

    save_data(df, str(output_path))
    assert output_path.exists()

    result = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(result, df)
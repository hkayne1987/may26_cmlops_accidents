"""Unit tests for src/data/preprocess.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import data.preprocess as preprocess


def make_csv(tmp_path, name, year, content):
    p = tmp_path / f"{name}-{year}.csv"
    p.write_text(content)
    return p


def test_load_year_success(tmp_path, monkeypatch):
    # Create CSVs for each table with simple content
    year = 2020
    csv_text = "col1;col2\na;1\nb;2\n"
    for t in preprocess.TABLES:
        make_csv(tmp_path, t, year, csv_text)

    # Point RAW_DIR to our tmp dir
    monkeypatch.setattr(preprocess, "RAW_DIR", tmp_path)

    tables = preprocess.load_year(year)
    assert isinstance(tables, dict)
    assert set(tables.keys()) == set(preprocess.TABLES)
    # Check that stripping/reading works
    assert list(tables[preprocess.TABLES[0]].columns) == ["col1", "col2"]


def test_load_year_missing_file(tmp_path, monkeypatch):
    year = 2021
    # Ensure no files exist in tmp_path
    monkeypatch.setattr(preprocess, "RAW_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        preprocess.load_year(year)


def test_merge_tables_success():
    # Build minimal consistent tables
    carac = pd.DataFrame({"Num_Acc": [1, 2], "c": [10, 20]})
    lieux = pd.DataFrame({"Num_Acc": [1, 2], "l": [100, 200]})
    vehicules = pd.DataFrame({"Num_Acc": [1, 2], "id_vehicule": [11, 22], "num_veh": [1, 2]})
    usagers = pd.DataFrame({"Num_Acc": [1, 2], "id_vehicule": [11, 22], "u": [5, 6]})

    tables = {"caracteristiques": carac, "lieux": lieux, "vehicules": vehicules, "usagers": usagers}

    out = preprocess.merge_tables(tables)
    # Should keep same number of usagers
    assert len(out) == len(usagers)
    # Columns from all tables present
    assert "c" in out.columns and "l" in out.columns and "u" in out.columns


def test_merge_tables_inconsistent_raises():
    # Create tables that will change row count after merge
    carac = pd.DataFrame({"Num_Acc": [1], "c": [10]})
    lieux = pd.DataFrame({"Num_Acc": [1], "l": [100]})
    # vehicules has two rows for same id causing cartesian expansion
    vehicules = pd.DataFrame({"Num_Acc": [1, 1], "id_vehicule": [11, 11]})
    usagers = pd.DataFrame({"Num_Acc": [1], "id_vehicule": [11]})

    tables = {"caracteristiques": carac, "lieux": lieux, "vehicules": vehicules, "usagers": usagers}

    with pytest.raises(ValueError):
        preprocess.merge_tables(tables)


def test_build_target_filters_and_labels():
    df = pd.DataFrame({"grav": ["-1", "2", "3", "0"]})
    out = preprocess.build_target(df)
    # Should remove the -1 row
    assert (out["grav"] == "-1").sum() == 0
    # grave should be 1 for 2 and 3
    assert out.loc[out["grav"] == "2", "grave"].iloc[0] == 1
    assert out.loc[out["grav"] == "3", "grave"].iloc[0] == 1
    # and 0 otherwise
    assert out.loc[out["grav"] == "0", "grave"].iloc[0] == 0


def test_engineer_features_time():
    df = pd.DataFrame({"an_nais": [1980, "notnum"], "an": [2020, 2020], "hrmn": ["12:34", "00:05"]})
    out = preprocess.engineer_features(df)
    # heure and minute derived from hrmn
    assert out.loc[0, "heure"] == "12"
    assert out.loc[0, "minute"] == "34"
    assert out.loc[1, "heure"] == "00"
    assert out.loc[1, "minute"] == "05"


def test_build_feature_matrix_types_and_groups():
    df = pd.DataFrame({
        "Num_Acc": [1, 1],
        "grave": [0, 1],
        "age": [30, 40],
        "heure": ["12", "13"],
        "grav": ["0", "1"],
    })
    X, y, groups = preprocess.build_feature_matrix(df)
    # y is grave
    assert list(y.values) == [0, 1]
    # groups is Num_Acc
    assert list(groups.values) == [1, 1]
    # X should not contain columns in DROP_COLS
    for c in preprocess.DROP_COLS:
        assert c not in X.columns


def test_split_train_test_no_overlap(monkeypatch):
    # Create df with two accidents each with two usagers
    df = pd.DataFrame({"Num_Acc": [1, 1, 2, 2], "grave": [0, 1, 0, 1]})
    # Force a simple splitter that keeps accidents together
    class SimpleSplitter:
        def __init__(self, n_splits, test_size, random_state):
            pass

        def split(self, df, y, groups):
            # Put first accident in train, second in test
            yield ([0, 1], [2, 3])

    monkeypatch.setattr(preprocess, "GroupShuffleSplit", SimpleSplitter)

    train, test = preprocess.split_train_test(df)
    assert set(train["Num_Acc"]) & set(test["Num_Acc"]) == set()


def test_split_train_test_overlap_raises(monkeypatch):
    df = pd.DataFrame({"Num_Acc": [1, 2, 3], "grave": [0, 1, 0]})

    class BadSplitter:
        def __init__(self, n_splits, test_size, random_state):
            pass

        def split(self, df, y, groups):
            # produce indices that cause overlap (both train and test include index 0)
            yield ([0, 1], [0, 2])

    monkeypatch.setattr(preprocess, "GroupShuffleSplit", BadSplitter)

    with pytest.raises(ValueError):
        preprocess.split_train_test(df)


def test_preprocess_all_uses_pipeline(monkeypatch, tmp_path):
    # Monkeypatch the sub-steps to avoid heavy IO
    df_sample = pd.DataFrame({"Num_Acc": [1, 1], "grave": [0, 1]})

    monkeypatch.setattr(preprocess, "YEARS", [2022])
    monkeypatch.setattr(preprocess, "load_year", lambda year: {"caracteristiques": pd.DataFrame({"Num_Acc": [1]}),
                                                                   "lieux": pd.DataFrame({"Num_Acc": [1]}),
                                                                   "vehicules": pd.DataFrame({"Num_Acc": [1], "id_vehicule": [1]}),
                                                                   "usagers": pd.DataFrame({"Num_Acc": [1], "id_vehicule": [1]})})
    monkeypatch.setattr(preprocess, "merge_tables", lambda tables: df_sample)
    monkeypatch.setattr(preprocess, "build_target", lambda df: df.assign(grave=[0, 1]))
    monkeypatch.setattr(preprocess, "engineer_features", lambda df: df)

    # Prevent actual file writes by overriding the output paths and
    # DataFrame.to_parquet. TRAIN_PATH and TEST_PATH are resolved at import
    # time, so patching PROCESSED_DIR alone leaves them pointing at the real
    # data/processed/, which no longer exists in Git since it moved to DVC.
    monkeypatch.setattr(preprocess, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(preprocess, "TRAIN_PATH", tmp_path / "train.parquet")
    monkeypatch.setattr(preprocess, "TEST_PATH", tmp_path / "test.parquet")
    saved = {}

    def fake_to_parquet(self, path, index=False):
        # Write a real (empty) file: preprocess_all calls .stat() on it to
        # log the written size.
        Path(path).write_bytes(b"")
        saved[str(path)] = True

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    # Avoid running the real GroupShuffleSplit inside preprocess_all
    monkeypatch.setattr(preprocess, "split_train_test", lambda full: (df_sample, df_sample))

    train, test = preprocess.preprocess_all()
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    # Ensure parquet write was attempted
    assert any("train.parquet" in p for p in saved.keys())

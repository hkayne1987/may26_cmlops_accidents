"""Unit tests for the preprocess module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from omegaconf import DictConfig

# Add src directory to path to import training modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_config():
    """Create a sample Hydra configuration for preprocessing."""
    return DictConfig({
        "data": {
            "raw_path": "data/raw/data.csv",
            "processed_path": "data/processed/train.csv",
        },
        "preprocessing": {
            "handle_missing": "drop",
            "test_size": 0.2,
            "random_state": 42,
            "scale_features": False,
        },
    })


@pytest.fixture
def sample_raw_data():
    """Create sample raw data with target column."""
    return pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature2": [2.0, 4.0, 6.0, 8.0, 10.0],
        "target": [0, 1, 0, 1, 0],
    })


class TestPreprocessModule:
    """Test cases for the preprocess module."""

    def test_preprocess_module_imports(self):
        """Test that preprocess module can be imported."""
        try:
            from training.preprocess import preprocess
            assert callable(preprocess)
        except ImportError:
            pytest.skip("training module not available")

    @patch("training.preprocess.Path")
    def test_preprocess_missing_raw_data(self, mock_path, sample_config):
        """Test preprocessing returns early when raw data is missing."""
        from training.preprocess import preprocess

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = False
        mock_path.return_value = mock_path_obj

        result = preprocess(sample_config)
        assert result is None

    @patch("training.preprocess.Path")
    def test_preprocess_with_drop_missing(self, mock_path, sample_config, sample_raw_data):
        """Test preprocessing with drop strategy for missing values."""
        from training.preprocess import preprocess

        data_with_missing = sample_raw_data.copy()
        data_with_missing.loc[0, "feature1"] = None

        def path_side_effect(p):
            mock_obj = MagicMock()
            if str(p) == sample_config.data.raw_path:
                mock_obj.exists.return_value = True
            else:
                mock_obj.parent.mkdir = MagicMock()
                mock_obj.mkdir = MagicMock()
            return mock_obj

        mock_path.side_effect = path_side_effect

        with patch("training.preprocess.pd.read_csv", return_value=data_with_missing):
            with patch("training.preprocess.train_test_split") as mock_split:
                x_train = data_with_missing[["feature1", "feature2"]].iloc[:4]
                x_test = data_with_missing[["feature1", "feature2"]].iloc[4:]
                y_train = data_with_missing["target"].iloc[:4]
                y_test = data_with_missing["target"].iloc[4:]
                mock_split.return_value = (x_train, x_test, y_train, y_test)

                preprocess(sample_config)

                # Should have been called
                assert mock_split.called

    @patch("training.preprocess.Path")
    def test_preprocess_train_test_split(self, mock_path, sample_config, sample_raw_data):
        """Test that data is split into train and test sets."""
        from training.preprocess import preprocess

        def path_side_effect(p):
            mock_obj = MagicMock()
            if str(p) == sample_config.data.raw_path:
                mock_obj.exists.return_value = True
            else:
                mock_obj.parent.mkdir = MagicMock()
                mock_obj.mkdir = MagicMock()
            return mock_obj

        mock_path.side_effect = path_side_effect

        with patch("training.preprocess.pd.read_csv", return_value=sample_raw_data):
            with patch("training.preprocess.train_test_split") as mock_split:
                x_train = sample_raw_data[["feature1", "feature2"]].iloc[:4]
                x_test = sample_raw_data[["feature1", "feature2"]].iloc[4:]
                y_train = sample_raw_data["target"].iloc[:4]
                y_test = sample_raw_data["target"].iloc[4:]
                mock_split.return_value = (x_train, x_test, y_train, y_test)

                preprocess(sample_config)

                # Verify split was called with correct test_size
                mock_split.assert_called_once()
                call_kwargs = mock_split.call_args[1]
                assert call_kwargs["test_size"] == 0.2
                assert call_kwargs["random_state"] == 42


    @patch("training.preprocess.Path")
    def test_preprocess_without_target_column(self, mock_path, sample_config):
        """Test preprocessing when target column is missing."""
        from training.preprocess import preprocess

        data_no_target = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [2.0, 4.0, 6.0],
        })

        def path_side_effect(p):
            mock_obj = MagicMock()
            if str(p) == sample_config.data.raw_path:
                mock_obj.exists.return_value = True
            else:
                mock_obj.parent.mkdir = MagicMock()
                mock_obj.mkdir = MagicMock()
            return mock_obj

        mock_path.side_effect = path_side_effect

        with patch("training.preprocess.pd.read_csv", return_value=data_no_target):
            preprocess(sample_config)

            # Should still process without error

    @patch("training.preprocess.Path")
    def test_preprocess_saves_csv(self, mock_path, sample_config, sample_raw_data):
        """Test that preprocessing saves CSV files."""
        from training.preprocess import preprocess

        def path_side_effect(p):
            mock_obj = MagicMock()
            if str(p) == sample_config.data.raw_path:
                mock_obj.exists.return_value = True
            else:
                mock_obj.parent.mkdir = MagicMock()
                mock_obj.mkdir = MagicMock()
            return mock_obj

        mock_path.side_effect = path_side_effect

        with patch("training.preprocess.pd.read_csv", return_value=sample_raw_data):
            with patch("training.preprocess.train_test_split") as mock_split:
                with patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
                    x_train = sample_raw_data[["feature1", "feature2"]].iloc[:4]
                    x_test = sample_raw_data[["feature1", "feature2"]].iloc[4:]
                    y_train = sample_raw_data["target"].iloc[:4]
                    y_test = sample_raw_data["target"].iloc[4:]
                    mock_split.return_value = (x_train, x_test, y_train, y_test)

                    preprocess(sample_config)

                    # CSV should be saved
                    assert mock_to_csv.called

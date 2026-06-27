"""Unit tests for the training module."""

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
    """Create a sample Hydra configuration for training."""
    return DictConfig({
        "data": {"processed_path": "data/processed/train.csv"},
        "model": {
            "name": "RandomForestClassifier",
            "n_estimators": 10,
            "max_depth": 5,
            "random_state": 42,
        },
        "mlflow": {"experiment_name": "test-experiment"},
    })


@pytest.fixture
def sample_train_data(tmp_path):
    """Create sample training data."""
    data = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature2": [2.0, 4.0, 6.0, 8.0, 10.0],
        "target": [0, 1, 0, 1, 0],
    })
    csv_path = tmp_path / "train.csv"
    data.to_csv(csv_path, index=False)
    return str(csv_path), data


class TestTrainModule:
    """Test cases for the training train module."""

    def test_train_module_imports(self):
        """Test that train module can be imported."""
        try:
            from training.train import train
            assert callable(train)
        except ImportError:
            pytest.skip("training module not available")

    @patch("training.train.mlflow")
    @patch("training.train.Path")
    def test_train_missing_data_file(self, mock_path, mock_mlflow, sample_config):
        """Test training returns early when data file is missing."""
        from training.train import train

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = False
        mock_path.return_value = mock_path_obj

        result = train(sample_config)
        assert result is None

    @patch("training.train.mlflow")
    @patch("training.train.RandomForestClassifier")
    @patch("training.train.joblib")
    def test_train_creates_and_fits_model(self, mock_joblib, mock_rf, mock_mlflow, sample_config):
        """Test that train creates and fits a RandomForest model."""
        from training.train import train

        data = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [2.0, 4.0, 6.0],
            "target": [0, 1, 0],
        })

        mock_model = MagicMock()
        mock_rf.return_value = mock_model

        with patch("training.train.Path") as mock_path:
            mock_path_obj = MagicMock()
            mock_path_obj.exists.return_value = True
            mock_path_obj.mkdir = MagicMock()
            mock_path.return_value = mock_path_obj

            with patch("training.train.pd.read_csv", return_value=data):
                train(sample_config)

                # Verify model creation
                mock_rf.assert_called_once_with(
                    n_estimators=10,
                    max_depth=5,
                    random_state=42,
                )

                # Verify model fitting
                assert mock_model.fit.called

    @patch("training.train.mlflow")
    @patch("training.train.RandomForestClassifier")
    @patch("training.train.joblib")
    def test_train_saves_model(self, mock_joblib, mock_rf, mock_mlflow, sample_config):
        """Test that trained model is saved using joblib."""
        from training.train import train

        data = pd.DataFrame({
            "feature1": [1.0, 2.0],
            "feature2": [2.0, 4.0],
            "target": [0, 1],
        })

        mock_model = MagicMock()
        mock_rf.return_value = mock_model

        with patch("training.train.Path") as mock_path:
            mock_path_obj = MagicMock()
            mock_path_obj.exists.return_value = True
            mock_path_obj.mkdir = MagicMock()
            mock_path.return_value = mock_path_obj

            with patch("training.train.pd.read_csv", return_value=data):
                train(sample_config)

                # Verify model is saved
                assert mock_joblib.dump.called

    @patch("training.train.mlflow")
    @patch("training.train.RandomForestClassifier")
    @patch("training.train.joblib")
    def test_train_logs_to_mlflow(self, mock_joblib, mock_rf, mock_mlflow, sample_config):
        """Test that train logs to MLflow."""
        from training.train import train

        data = pd.DataFrame({
            "feature1": [1.0, 2.0],
            "feature2": [2.0, 4.0],
            "target": [0, 1],
        })

        mock_model = MagicMock()
        mock_rf.return_value = mock_model

        with patch("training.train.Path") as mock_path:
            mock_path_obj = MagicMock()
            mock_path_obj.exists.return_value = True
            mock_path_obj.mkdir = MagicMock()
            mock_path.return_value = mock_path_obj

            with patch("training.train.pd.read_csv", return_value=data):
                train(sample_config)

                # Verify MLflow was called
                mock_mlflow.set_experiment.assert_called_with("test-experiment")
                assert mock_mlflow.start_run.called

    @patch("training.train.mlflow")
    @patch("training.train.RandomForestClassifier")
    @patch("training.train.joblib")
    def test_train_handles_missing_target(self, mock_joblib, mock_rf, mock_mlflow, sample_config):
        """Test training when target column is missing."""
        from training.train import train

        data = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [2.0, 4.0, 6.0],
            "other": [0, 1, 0],
        })

        mock_model = MagicMock()
        mock_rf.return_value = mock_model

        with patch("training.train.Path") as mock_path:
            mock_path_obj = MagicMock()
            mock_path_obj.exists.return_value = True
            mock_path_obj.mkdir = MagicMock()
            mock_path.return_value = mock_path_obj

            with patch("training.train.pd.read_csv", return_value=data):
                train(sample_config)

                # Should still fit using last column as target
                assert mock_model.fit.called

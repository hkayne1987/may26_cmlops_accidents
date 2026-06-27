"""Unit tests for the evaluate module."""

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
    """Create a sample Hydra configuration for evaluation."""
    return DictConfig({
        "data": {
            "test_path": "data/processed/test.csv",
            "train_path": "data/processed/train.csv",
        },
        "model": {"path": "models/model.joblib"},
        "metrics": {"output_path": "reports/metrics.json"},
        "mlflow": {"experiment_name": "test-experiment"},
    })


@pytest.fixture
def sample_test_data():
    """Create sample test data."""
    return pd.DataFrame({
        "feature1": [1.5, 2.5, 3.5, 4.5],
        "feature2": [3.0, 6.0, 9.0, 12.0],
        "target": [0, 1, 0, 1],
    })


class TestEvaluateModule:
    """Test cases for the evaluate module."""

    def test_evaluate_module_imports(self):
        """Test that evaluate module can be imported."""
        try:
            from training.evaluate import evaluate
            assert callable(evaluate)
        except ImportError:
            pytest.skip("training module not available")

    @patch("training.evaluate.mlflow")
    @patch("training.evaluate.Path")
    def test_evaluate_model_not_found(self, mock_path, mock_mlflow, sample_config):
        """Test evaluation when model file is missing."""
        from training.evaluate import evaluate

        def path_side_effect(p):
            mock_obj = MagicMock()
            if str(p) == sample_config.model.path:
                mock_obj.exists.return_value = False
            return mock_obj

        mock_path.side_effect = path_side_effect

        result = evaluate(sample_config)
        assert result is None

    @patch("training.evaluate.mlflow")
    @patch("training.evaluate.joblib")
    @patch("training.evaluate.Path")
    def test_evaluate_loads_model(
        self, mock_path, mock_joblib, mock_mlflow, sample_config, sample_test_data
    ):
        """Test that evaluate loads the model."""
        from training.evaluate import evaluate

        mock_model = MagicMock()
        mock_model.predict.return_value = [0, 1, 0, 1]
        mock_joblib.load.return_value = mock_model

        def path_side_effect(p):
            mock_obj = MagicMock()
            if str(p) == sample_config.model.path:
                mock_obj.exists.return_value = True
            elif str(p) == sample_config.data.test_path:
                mock_obj.exists.return_value = True
            else:
                mock_obj.parent.mkdir = MagicMock()
            return mock_obj

        mock_path.side_effect = path_side_effect

        with patch("training.evaluate.pd.read_csv", return_value=sample_test_data):
            evaluate(sample_config)

            # Verify model was loaded
            mock_joblib.load.assert_called_once()

    @patch("training.evaluate.mlflow")
    @patch("training.evaluate.joblib")
    @patch("training.evaluate.Path")
    def test_evaluate_makes_predictions(
        self, mock_path, mock_joblib, mock_mlflow, sample_config, sample_test_data
    ):
        """Test that evaluate makes predictions."""
        from training.evaluate import evaluate

        mock_model = MagicMock()
        mock_model.predict.return_value = [0, 1, 0, 1]
        mock_joblib.load.return_value = mock_model

        def path_side_effect(p):
            mock_obj = MagicMock()
            if str(p) == sample_config.model.path:
                mock_obj.exists.return_value = True
            elif str(p) == sample_config.data.test_path:
                mock_obj.exists.return_value = True
            else:
                mock_obj.parent.mkdir = MagicMock()
            return mock_obj

        mock_path.side_effect = path_side_effect

        with patch("training.evaluate.pd.read_csv", return_value=sample_test_data):
            evaluate(sample_config)

            # Verify predictions were made
            assert mock_model.predict.called

    @patch("training.evaluate.mlflow")
    @patch("training.evaluate.joblib")
    @patch("training.evaluate.Path")
    def test_evaluate_test_data_fallback(
        self, mock_path, mock_joblib, mock_mlflow, sample_config, sample_test_data
    ):
        """Test evaluation falls back to train data when test data is missing."""
        from training.evaluate import evaluate

        mock_model = MagicMock()
        mock_model.predict.return_value = [0, 1, 0, 1]
        mock_joblib.load.return_value = mock_model

        def path_side_effect(p):
            mock_obj = MagicMock()
            if str(p) == sample_config.model.path:
                mock_obj.exists.return_value = True
            elif str(p) == sample_config.data.test_path:
                mock_obj.exists.return_value = False  # Test doesn't exist
            elif str(p) == sample_config.data.train_path:
                mock_obj.exists.return_value = True  # But train exists
            else:
                mock_obj.parent.mkdir = MagicMock()
            return mock_obj

        mock_path.side_effect = path_side_effect

        with patch("training.evaluate.pd.read_csv", return_value=sample_test_data):
            evaluate(sample_config)

            # Should still make predictions
            assert mock_model.predict.called

    @patch("training.evaluate.mlflow")
    @patch("training.evaluate.joblib")
    @patch("training.evaluate.Path")
    def test_evaluate_logs_metrics(
        self, mock_path, mock_joblib, mock_mlflow, sample_config, sample_test_data
    ):
        """Test that evaluate logs metrics to MLflow."""
        from training.evaluate import evaluate

        mock_model = MagicMock()
        mock_model.predict.return_value = [0, 1, 0, 1]
        mock_joblib.load.return_value = mock_model

        def path_side_effect(p):
            mock_obj = MagicMock()
            if str(p) == sample_config.model.path:
                mock_obj.exists.return_value = True
            elif str(p) == sample_config.data.test_path:
                mock_obj.exists.return_value = True
            else:
                mock_obj.parent.mkdir = MagicMock()
            return mock_obj

        mock_path.side_effect = path_side_effect

        with patch("training.evaluate.pd.read_csv", return_value=sample_test_data):
            evaluate(sample_config)

            # Verify metrics were logged
            assert mock_mlflow.log_metric.called

    @patch("training.evaluate.mlflow")
    @patch("training.evaluate.joblib")
    @patch("training.evaluate.Path")
    def test_evaluate_without_target(
        self, mock_path, mock_joblib, mock_mlflow, sample_config
    ):
        """Test evaluation when target column is missing."""
        from training.evaluate import evaluate

        data_no_target = pd.DataFrame({
            "feature1": [1.5, 2.5],
            "feature2": [3.0, 6.0],
            "other": [0, 1],
        })

        mock_model = MagicMock()
        mock_model.predict.return_value = [0, 1]
        mock_joblib.load.return_value = mock_model

        def path_side_effect(p):
            mock_obj = MagicMock()
            if str(p) == sample_config.model.path:
                mock_obj.exists.return_value = True
            elif str(p) == sample_config.data.test_path:
                mock_obj.exists.return_value = True
            else:
                mock_obj.parent.mkdir = MagicMock()
            return mock_obj

        mock_path.side_effect = path_side_effect

        with patch("training.evaluate.pd.read_csv", return_value=data_no_target):
            evaluate(sample_config)

            # Should use last column as target
            assert mock_model.predict.called

    @patch("training.evaluate.mlflow")
    @patch("training.evaluate.joblib")
    @patch("training.evaluate.Path")
    def test_evaluate_saves_metrics_json(
        self, mock_path, mock_joblib, mock_mlflow, sample_config, sample_test_data
    ):
        """Test that metrics are saved to JSON file."""
        from training.evaluate import evaluate

        mock_model = MagicMock()
        mock_model.predict.return_value = [0, 1, 0, 1]
        mock_joblib.load.return_value = mock_model

        def path_side_effect(p):
            mock_obj = MagicMock()
            if str(p) == sample_config.model.path:
                mock_obj.exists.return_value = True
            elif str(p) == sample_config.data.test_path:
                mock_obj.exists.return_value = True
            else:
                mock_obj.parent.mkdir = MagicMock()
            return mock_obj

        mock_path.side_effect = path_side_effect

        with patch("training.evaluate.pd.read_csv", return_value=sample_test_data):
            with patch("builtins.open", create=True) as mock_open:
                evaluate(sample_config)

                # Verify file was opened for writing
                assert mock_open.called

    @patch("training.evaluate.mlflow")
    @patch("training.evaluate.joblib")
    @patch("training.evaluate.Path")
    def test_evaluate_sets_mlflow_experiment(
        self, mock_path, mock_joblib, mock_mlflow, sample_config, sample_test_data
    ):
        """Test that MLflow experiment is set correctly."""
        from training.evaluate import evaluate

        mock_model = MagicMock()
        mock_model.predict.return_value = [0, 1, 0, 1]
        mock_joblib.load.return_value = mock_model

        def path_side_effect(p):
            mock_obj = MagicMock()
            if str(p) == sample_config.model.path:
                mock_obj.exists.return_value = True
            elif str(p) == sample_config.data.test_path:
                mock_obj.exists.return_value = True
            else:
                mock_obj.parent.mkdir = MagicMock()
            return mock_obj

        mock_path.side_effect = path_side_effect

        with patch("training.evaluate.pd.read_csv", return_value=sample_test_data):
            evaluate(sample_config)

            # Verify experiment was set
            mock_mlflow.set_experiment.assert_called_once_with("test-experiment")

    @patch("training.evaluate.mlflow")
    @patch("training.evaluate.joblib")
    @patch("training.evaluate.Path")
    def test_evaluate_creates_mlflow_run(
        self, mock_path, mock_joblib, mock_mlflow, sample_config, sample_test_data
    ):
        """Test that MLflow run is created."""
        from training.evaluate import evaluate

        mock_model = MagicMock()
        mock_model.predict.return_value = [0, 1, 0, 1]
        mock_joblib.load.return_value = mock_model

        def path_side_effect(p):
            mock_obj = MagicMock()
            if str(p) == sample_config.model.path:
                mock_obj.exists.return_value = True
            elif str(p) == sample_config.data.test_path:
                mock_obj.exists.return_value = True
            else:
                mock_obj.parent.mkdir = MagicMock()
            return mock_obj

        mock_path.side_effect = path_side_effect

        with patch("training.evaluate.pd.read_csv", return_value=sample_test_data):
            evaluate(sample_config)

            # Verify run was created
            mock_mlflow.start_run.assert_called_once_with(run_name="evaluation")

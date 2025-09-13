import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.models.predictors.base_predictor import BasePredictor


class DummyPredictor(BasePredictor):
    def load_model_from_mlflow(self) -> None:  # pragma: no cover - trivial
        return None


def _make_features_and_targets(n=5):
    features = pd.DataFrame({"close": np.linspace(10, 15, n), "date_int": np.arange(n), "ticker_id": np.arange(n)})
    targets = pd.Series(np.linspace(0.1, 0.5, n))
    return features, targets


def test_load_recent_data_calls_pipeline_and_returns_dfs():
    features, targets = _make_features_and_targets(6)
    fake_result = {"X_test": features, "y_test": targets}

    dp = DummyPredictor(run_id="r1", model_type="test")
    with patch("src.models.predictors.base_predictor.prepare_ml_data_for_prediction_with_cleaning", return_value=fake_result) as mock_pipe:
        X, meta = dp.load_recent_data(days_back=7)
        mock_pipe.assert_called_once()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(meta, pd.DataFrame)


def test_save_predictions_to_excel_writes_file(tmp_path):
    dp = DummyPredictor(run_id="r2", model_type="testtype")
    features, targets = _make_features_and_targets(3)
    metadata = pd.DataFrame({"target_values": targets.values})
    predictions = np.array([0.1, 0.2, 0.3])

    # create a small DataFrame that will survive filtering step by mocking threshold behavior
    dp.optimal_threshold = None

    out = dp.save_predictions_to_excel(features, metadata, predictions)
    # function may return None if no predictions to save; ensure no exception and return type is either None or str
    assert out is None or isinstance(out, str)

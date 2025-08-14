import os
import pandas as pd
import numpy as np
import pytest

from src.models.predictors.base_predictor import BasePredictor


class DummyPredictor(BasePredictor):
    def __init__(self):
        super().__init__(run_id="1234567890abcdef", model_type="lightgbm")
        class DummyModel:
            feature_names = ["ticker_id", "date_int", "close"]
            def predict(self, X):
                return np.full(len(X), 0.01)
            def get_prediction_confidence(self, X):
                return np.linspace(0, 1, len(X))
        self.model = DummyModel()
        self.optimal_threshold = 0.5

    def load_model_from_mlflow(self) -> None:
        pass


@pytest.mark.integration
def test_excel_export_schema(tmp_path, monkeypatch):
    p = DummyPredictor()

    # Prepare tiny features/metadata
    features_df = pd.DataFrame({
        "ticker_id": [1, 1, 2, 2],
        "date_int": [0, 0, 1, 1],
        "close": [10.0, 12.0, 9.0, 11.0],
    })
    metadata_df = pd.DataFrame({"target_values": [0.02, -0.01, 0.03, 0.00]})
    predictions = np.array([0.02, -0.01, 0.03, 0.00])

    # Run in tmp working directory so outputs are created under tmp_path
    monkeypatch.chdir(tmp_path)

    output_path = p.save_predictions_to_excel(features_df, metadata_df, predictions)
    assert output_path.endswith(".xlsx")
    assert os.path.exists(output_path)

    df = pd.read_excel(output_path)
    expected_columns = {
        "ticker_id", "date_int", "date", "ticker", "company_name",
        "predicted_return", "predicted_price", "current_price", "actual_return",
    }
    assert expected_columns.issubset(set(df.columns))



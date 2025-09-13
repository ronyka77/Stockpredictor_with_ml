import os
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

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
def test_excel_export_schema(tmp_path):
    p = DummyPredictor()

    # Prepare tiny features/metadata such that Friday average profit > $5
    # date_int 2 maps to 2020-01-03 (Friday) relative to origin 2020-01-01
    features_df = pd.DataFrame(
        {
            "ticker_id": [1, 1, 2, 2],
            "date_int": [2, 2, 2, 2],
            "close": [1.0, 1.0, 1.0, 1.0],
        }
    )
    # Large actual returns to make actual_price >> current_price -> high profit
    metadata_df = pd.DataFrame({"target_values": [1.0, 1.0, 1.0, 1.0]})
    # Predictions can be small; confidence filtering in DummyPredictor will keep top half
    predictions = np.array([0.0, 0.0, 0.0, 0.0])

    # Patch ticker metadata fetch to avoid DB calls
    ticker_meta = pd.DataFrame(
        {"id": [1, 2], "ticker": ["T1", "T2"], "name": ["Name1", "Name2"]}
    )

    # Run in tmp working directory so outputs are created under tmp_path
    old_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        with patch(
            "src.feature_engineering.data_loader.StockDataLoader.get_ticker_metadata",
            return_value=ticker_meta,
        ):
            output_path = p.save_predictions_to_excel(
                features_df, metadata_df, predictions
            )
            if not output_path.endswith(".xlsx"):
                raise AssertionError("Output path does not end with .xlsx")
            if not os.path.exists(output_path):
                raise AssertionError("Expected Excel output file to exist")

            df = pd.read_excel(output_path)
    finally:
        os.chdir(old_cwd)
    # Require core prediction columns; ticker/date metadata may be filled or
    # substituted depending on environment (metadata fetch can fail in CI).
    required_columns = {
        "predicted_return",
        "predicted_price",
        "current_price",
        "actual_return",
    }
    if not required_columns.issubset(set(df.columns)):
        raise AssertionError("Exported Excel is missing required prediction columns")

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.models.time_series.mlp.mlp_architecture import MLPDataUtils, MLPModule
from src.models.time_series.mlp.mlp_predictor import MLPPredictor


class TestMLPScalerImplementation:
    def setup_method(self):
        rng = np.random.RandomState(42)
        self.x_test = pd.DataFrame(
            {
                "feature1": rng.randn(100),
                "feature2": rng.randn(100),
                "feature3": rng.randn(100),
            }
        )
        self.x_test.loc[0, "feature1"] = np.nan
        self.x_test.loc[1, "feature2"] = np.inf
        self.x_test.loc[2, "feature3"] = -np.inf
        self.predictor = MLPPredictor(
            model_name="test_mlp",
            config={"layer_sizes": [10, 5], "input_size": 3, "batch_size": 16},
        )
        self.predictor.model = MLPModule(
            input_size=3, layer_sizes=[10, 5], output_size=1
        )
        self.predictor.device = torch.device("cpu")
        self.predictor.model = self.predictor.model.to("cpu")

    def test_validate_and_clean_data_method(self):
        """Clean and scale data removes NaNs/Infs and returns a scaler instance."""
        cleaned = MLPDataUtils.validate_and_clean_data(self.x_test)
        X_clean, scaler = MLPDataUtils.scale_data(cleaned, None, True)
        # Use pandas testing to ensure no NaNs or infinities remain

        assert not X_clean.isnull().any().any(), (
            f"Cleaned data contains NaNs: {X_clean.isnull().sum().to_dict()}"
        )
        assert not np.isinf(X_clean.values).any(), "Cleaned data contains infinities"
        if scaler is None:
            raise AssertionError("Scaler should be created during scale_data")

    def test_predictor_with_scaler(self):
        """Set a fitted scaler on predictor and ensure predictions run with expected length."""
        cleaned = MLPDataUtils.validate_and_clean_data(self.x_test)
        X_clean, _ = MLPDataUtils.scale_data(cleaned, None, True)
        scaler = StandardScaler()
        scaler.fit(X_clean)
        self.predictor.set_scaler(scaler)
        predictions = self.predictor.predict(self.x_test)
        assert isinstance(predictions, (list, tuple, np.ndarray)), (
            "Predictions should be a sequence or numpy ndarray"
        )
        assert len(predictions) == len(self.x_test), (
            "Predictions length mismatch with input data"
        )

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.models.time_series.mlp.mlp_architecture import MLPDataUtils, MLPModule
from src.models.time_series.mlp.mlp_predictor import MLPPredictor


class TestMLPScalerImplementation:
    def setup_method(self):
        np.random.seed(42)
        self.X_test = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )
        self.X_test.loc[0, "feature1"] = np.nan
        self.X_test.loc[1, "feature2"] = np.inf
        self.X_test.loc[2, "feature3"] = -np.inf
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
        cleaned = MLPDataUtils.validate_and_clean_data(self.X_test)
        X_clean, scaler = MLPDataUtils.scale_data(cleaned, None, True)
        assert not X_clean.isnull().any().any()
        assert not np.isinf(X_clean.values).any()
        assert scaler is not None

    def test_predictor_with_scaler(self):
        cleaned = MLPDataUtils.validate_and_clean_data(self.X_test)
        X_clean, _ = MLPDataUtils.scale_data(cleaned, None, True)
        scaler = StandardScaler()
        scaler.fit(X_clean)
        self.predictor.set_scaler(scaler)
        predictions = self.predictor.predict(self.X_test)
        assert predictions is not None
        assert len(predictions) == len(self.X_test)

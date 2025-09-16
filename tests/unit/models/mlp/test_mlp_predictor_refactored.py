from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from src.models.time_series.mlp.mlp_predictor import MLPPredictor

VALIDATE_PATH = (
    "src.models.time_series.mlp.mlp_architecture.MLPDataUtils.validate_and_clean_data"
)
SCALE_PATH = "src.models.time_series.mlp.mlp_architecture.MLPDataUtils.scale_data"


class TestMLPPredictorRefactored:
    def setup_method(self):
        # Use a deterministic RNG for construction
        rng = np.random.RandomState(0)
        arr = rng.randn(5, 3)
        # inject NaN and inf deterministically
        arr[3, 0] = np.nan
        arr[2, 1] = np.inf
        self.X_test = pd.DataFrame(arr, columns=["feature1", "feature2", "feature3"])
        self.predictor = MLPPredictor(model_name="test_mlp")
        self.predictor.model = stub_model = MagicMock()
        stub_model.return_value = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5]])
        stub_model.eval.return_value = None
        self.predictor.device = "cpu"

    def test_predict_uses_validate_and_clean_data(self):
        """Ensure predictor calls validate_and_clean_data before predicting."""
        with patch(VALIDATE_PATH) as mock_validate:
            mock_validate.return_value = pd.DataFrame(
                {
                    "feature1": [1.0, 2.0, 3.0, 0.0, 5.0],
                    "feature2": [0.1, 0.2, 0.0, 0.4, 0.5],
                    "feature3": [10, 20, 30, 40, 50],
                }
            )
            result = self.predictor.predict(self.X_test)
            mock_validate.assert_called_once_with(self.X_test)
            if not isinstance(result, np.ndarray):
                raise AssertionError("Predictor result is not numpy ndarray")
            if len(result) != 5:
                raise AssertionError("Predictor result length mismatch")

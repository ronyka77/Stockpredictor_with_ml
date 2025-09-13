from unittest.mock import patch, MagicMock
from src.models.predictors.mlp_predictor import MLPPredictorWrapper


def test_mlp_loads_model_from_mlflow():
    with patch("src.models.predictors.mlp_predictor.MLPPredictorWithMLflow") as Mock:
        mock_inst = Mock.return_value
        mock_inst.load_model = MagicMock(return_value=True)
        mock_inst.model = MagicMock()
        mock_inst.feature_names = ["f1"]
        p = MLPPredictorWrapper(run_id="r")
        p.load_model_from_mlflow()
        assert p.model is not None


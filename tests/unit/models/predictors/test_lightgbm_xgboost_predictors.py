from unittest.mock import patch, MagicMock
from src.models.predictors.lightgbm_predictor import LightGBMPredictor
from src.models.predictors.xgboost_predictor import XGBoostPredictor


def test_lightgbm_loads_model_from_mlflow():
    """Load LightGBM model from mlflow and ensure model object is set on predictor."""
    with patch("src.models.predictors.lightgbm_predictor.LightGBMModel") as MockModel:
        MockModel.load_from_mlflow = MagicMock(
            return_value=MagicMock(model_name="lgbm", feature_names=["a"])
        )
        p = LightGBMPredictor(run_id="r")
        p.load_model_from_mlflow()
        assert p.model is not None


def test_xgboost_loads_model_from_mlflow():
    """Load XGBoost model from mlflow and ensure model object is set on predictor."""
    with patch("src.models.predictors.xgboost_predictor.XGBoostModel") as MockModel:
        MockModel.load_from_mlflow = MagicMock(
            return_value=MagicMock(model_name="xgb", feature_names=["a"])
        )
        p = XGBoostPredictor(run_id="r")
        p.load_model_from_mlflow()
        assert p.model is not None

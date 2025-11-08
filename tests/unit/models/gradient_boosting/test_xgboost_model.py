import numpy as np
import pytest
from unittest.mock import Mock, patch

from src.models.gradient_boosting.xgboost_model import XGBoostModel


@pytest.fixture()
def xgb_model_instance() -> XGBoostModel:
    return XGBoostModel(model_name="xgb_test_model", prediction_horizon=10)


def test_get_prediction_confidence_leaf_and_margin(xgb_model_instance, sample_tabular_data):
    """Setup: attach a dummy XGBoost model that returns leaf indices and margins."""

    X = sample_tabular_data["X"].iloc[:15]

    class DummyXGB:
        def __init__(self):
            self.best_iteration = 20

        def predict(self, dmatrix, pred_leaf=False, output_margin=False, ntree_limit=None):
            n = len(dmatrix.get_label()) if hasattr(dmatrix, "get_label") else len(dmatrix)
            if pred_leaf:
                return np.tile(np.arange(1, 11), (n, 1))
            if output_margin:
                return np.ones(n) * 2.0
            return np.arange(n).astype(float) + 0.5

    xgb_model_instance.model = DummyXGB()

    # Patch xgboost.DMatrix to return object with get_label matching input size
    class FakeDMatrix:
        def __init__(self, data, label=None, feature_names=None):
            self._data = data

        def get_label(self):
            return np.zeros(len(self._data))

    with patch(
        "src.models.gradient_boosting.xgboost_model.xgb.DMatrix",
        side_effect=lambda data, label=None, feature_names=None: FakeDMatrix(data),
    ):
        # Execution: leaf_depth
        conf_leaf = xgb_model_instance.get_prediction_confidence(X, method="leaf_depth")

        # Verification
        assert isinstance(conf_leaf, np.ndarray)
        assert conf_leaf.shape[0] == X.shape[0]
        assert conf_leaf.min() >= 0.0 and conf_leaf.max() <= 1.0

        # Execution: margin
        conf_margin = xgb_model_instance.get_prediction_confidence(X, method="margin")

        # Verification
        assert conf_margin.min() >= 0.0 and conf_margin.max() <= 1.0

        # Execution: variance
        conf_variance = xgb_model_instance.get_prediction_confidence(X, method="variance")

        # Verification
        assert conf_variance.min() >= 0.0 and conf_variance.max() <= 1.0

        # Execution: simple
        conf_simple = xgb_model_instance.get_prediction_confidence(X, method="simple")

        # Verification
        assert conf_simple.min() >= 0.0 and conf_simple.max() <= 1.0


def test_predict_uses_dmatrix_and_returns_array(xgb_model_instance, sample_tabular_data):
    """Setup: patch xgboost.DMatrix to return object accepted by Dummy model and verify predict call."""

    X = sample_tabular_data["X"].iloc[:12]

    # Create dummy model
    class DummyXGBModel:
        def __init__(self):
            self.best_iteration = 5

        def predict(self, dmatrix, ntree_limit=None):
            # Accept either DMatrix or ndarray-like
            size = len(dmatrix.get_label()) if hasattr(dmatrix, "get_label") else len(dmatrix)
            return np.zeros(size)

    xgb_model_instance.model = DummyXGBModel()

    # Patch xgboost.DMatrix to return the input X as an object with get_label
    class FakeDMatrix:
        def __init__(self, data, label=None, feature_names=None):
            self._data = data
            self._label = label

        def get_label(self):
            return np.zeros(len(self._data))

    with patch(
        "src.models.gradient_boosting.xgboost_model.xgb.DMatrix",
        side_effect=lambda data, label=None, feature_names=None: FakeDMatrix(
            data, label, feature_names
        ),
    ):
        # Execution
        preds = xgb_model_instance.predict(X)

        # Verification
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == X.shape[0]


def test_log_model_to_mlflow_calls_helper(xgb_model_instance, sample_tabular_data):
    """Setup: patch module-level log_to_mlflow helper to avoid real MLflow calls."""

    x_eval = sample_tabular_data["X"].iloc[:4]
    metrics = {"mse": 0.2}
    params = {"param": 2}

    with patch(
        "src.models.gradient_boosting.xgboost_model.log_to_mlflow", return_value="xg-run-1"
    ) as mock_log:
        xgb_model_instance.model = Mock()

        # Execution
        run_id = xgb_model_instance.log_model_to_mlflow(
            metrics=metrics, params=params, x_eval=x_eval, experiment_name="exp"
        )

        # Verification
        mock_log.assert_called_once()
        assert run_id == "xg-run-1"


def test_load_model_handles_missing_artifacts_and_signature(xgb_model_instance):
    """Setup: patch mlflow client and xgboost load_model to simulate missing signature/artifacts."""

    fake_run_id = "fake-xg-run"
    fake_run_info = Mock()
    fake_run_info.data = Mock()
    fake_run_info.data.params = {"model_model_name": "xg", "model_prediction_horizon": "10"}
    fake_run_info.info = Mock()
    fake_run_info.info.run_id = fake_run_id

    fake_client = Mock()
    fake_client.get_run.return_value = fake_run_info
    fake_client.list_artifacts.return_value = []

    with (
        patch(
            "src.models.gradient_boosting.xgboost_model.mlflow.tracking.MlflowClient",
            return_value=fake_client,
        ),
        patch(
            "src.models.gradient_boosting.xgboost_model.mlflow.xgboost.load_model",
            return_value=Mock(),
        ),
    ):
        # Execution
        xgb_model_instance.load_model(fake_run_id)

        # Verification
        assert xgb_model_instance.model is not None

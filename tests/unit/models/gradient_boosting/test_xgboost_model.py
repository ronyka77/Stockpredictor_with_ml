import numpy as np
import pytest
from unittest.mock import Mock, patch

from src.models.gradient_boosting.xgboost_model import XGBoostModel


@pytest.fixture()
def xgb_model_instance() -> XGBoostModel:
    """
    PyTest fixture that returns a preconfigured XGBoostModel for tests.
    
    Returns:
        XGBoostModel: An XGBoostModel instance with model_name="xgb_test_model" and prediction_horizon=10.
    """
    return XGBoostModel(model_name="xgb_test_model", prediction_horizon=10)


def test_get_prediction_confidence_leaf_and_margin(
    xgb_model_instance, sample_tabular_data
):
    """Setup: attach a dummy XGBoost model that returns leaf indices and margins."""

    X = sample_tabular_data["X"].iloc[:15]

    class DummyXGB:
        def __init__(self):
            """
            Initialize the instance and set a default best_iteration.
            
            Sets self.best_iteration to 20 to provide a stable default value for components that inspect the model's best_iteration attribute.
            """
            self.best_iteration = 20

        def predict(
            self, dmatrix, pred_leaf=False, output_margin=False, ntree_limit=None
        ):
            """
            Predict using a simplified dummy XGBoost-like model used in tests.
            
            This test helper accepts either an array-like input or an object exposing get_label(). It returns deterministic mock predictions:
            - If pred_leaf is True: a 2D integer array of shape (n, 10) where each row is the sequence 1..10 (replicated for each instance).
            - If output_margin is True: a 1D float array of length n filled with 2.0.
            - Otherwise: a 1D float array of length n with values [0.5, 1.5, 2.5, ...].
            
            Parameters:
                dmatrix: array-like or object with get_label() -> array-like
                    Input data or a wrapper providing labels; only its length is used to determine the number of predictions.
                pred_leaf (bool): If True, return per-instance leaf index vectors (shape (n, 10)).
                output_margin (bool): If True, return margin-style scores (1D array of 2.0).
                ntree_limit: Ignored (present for API compatibility).
            
            Returns:
                numpy.ndarray: Mock predictions as described above.
            """
            n = (
                len(dmatrix.get_label())
                if hasattr(dmatrix, "get_label")
                else len(dmatrix)
            )
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
            """
            Return a zero-valued label array matching the number of samples in the stored data.
            
            The returned NumPy 1-D array has length equal to len(self._data) and contains zeros (dtype float).
            """
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
        conf_variance = xgb_model_instance.get_prediction_confidence(
            X, method="variance"
        )

        # Verification
        assert conf_variance.min() >= 0.0 and conf_variance.max() <= 1.0

        # Execution: simple
        conf_simple = xgb_model_instance.get_prediction_confidence(X, method="simple")

        # Verification
        assert conf_simple.min() >= 0.0 and conf_simple.max() <= 1.0


def test_predict_uses_dmatrix_and_returns_array(
    xgb_model_instance, sample_tabular_data
):
    """Setup: patch xgboost.DMatrix to return object accepted by Dummy model and verify predict call."""

    X = sample_tabular_data["X"].iloc[:12]

    # Create dummy model
    class DummyXGBModel:
        def __init__(self):
            self.best_iteration = 5

        def predict(self, dmatrix, ntree_limit=None):
            # Accept either DMatrix or ndarray-like
            """
            Return a 1D numpy array of zeros matching the number of rows in the input.
            
            This method accepts either an xgboost.DMatrix-like object (providing get_label())
            or any array-like object with a length (e.g., numpy array, list). The optional
            ntree_limit parameter is accepted for API compatibility but ignored by this
            implementation.
            
            Parameters:
                dmatrix: DMatrix or array-like
                    Input data used to determine the output length. If a DMatrix-like object
                    is provided, its get_label() method is used to determine the number of
                    rows; otherwise, len(dmatrix) is used.
                ntree_limit (optional): int
                    Ignored; present for compatibility with XGBoost predict signatures.
            
            Returns:
                numpy.ndarray
                    1D array of zeros with length equal to the number of input rows.
            """
            size = (
                len(dmatrix.get_label())
                if hasattr(dmatrix, "get_label")
                else len(dmatrix)
            )
            return np.zeros(size)

    xgb_model_instance.model = DummyXGBModel()

    # Patch xgboost.DMatrix to return the input X as an object with get_label
    class FakeDMatrix:
        def __init__(self, data, label=None, feature_names=None):
            self._data = data
            self._label = label

        def get_label(self):
            """
            Return a zero-valued label array matching the number of samples in the stored data.
            
            The returned NumPy 1-D array has length equal to len(self._data) and contains zeros (dtype float).
            """
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

    X_eval = sample_tabular_data["X"].iloc[:4]
    metrics = {"mse": 0.2}
    params = {"param": 2}

    with patch(
        "src.models.gradient_boosting.xgboost_model.log_to_mlflow",
        return_value="xg-run-1",
    ) as mock_log:
        xgb_model_instance.model = Mock()

        # Execution
        run_id = xgb_model_instance.log_model_to_mlflow(
            metrics=metrics, params=params, X_eval=X_eval, experiment_name="exp"
        )

        # Verification
        mock_log.assert_called_once()
        assert run_id == "xg-run-1"


def test_load_model_handles_missing_artifacts_and_signature(xgb_model_instance):
    """Setup: patch mlflow client and xgboost load_model to simulate missing signature/artifacts."""

    fake_run_id = "fake-xg-run"
    fake_run_info = Mock()
    fake_run_info.data = Mock()
    fake_run_info.data.params = {
        "model_model_name": "xg",
        "model_prediction_horizon": "10",
    }
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

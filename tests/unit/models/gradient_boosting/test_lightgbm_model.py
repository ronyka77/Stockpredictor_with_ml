import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, patch


def test_identify_categorical_features_includes_ticker_and_low_cardinality(
    sample_tabular_data, lgb_model_instance
):
    """Setup: create dataframe with integer low-cardinality and object columns."""

    X = sample_tabular_data["X"].copy()
    # Make a low-cardinality integer column
    X["low_card_int"] = [1] * 50 + [2] * 70
    X["str_cat"] = ["a"] * 60 + ["b"] * 60

    # Execution
    cats = lgb_model_instance._identify_categorical_features(X)

    # Verification
    assert "ticker_id" in cats
    assert "low_card_int" in cats
    assert "str_cat" in cats


def test_validate_training_data_raises_on_many_constant_features(lgb_model_instance):
    """Setup: create X with >50% constant columns to trigger ValueError."""

    n_rows = 20
    # Create 10 columns, 6 of them constant
    data = {f"c{i}": np.ones(n_rows) if i < 6 else np.arange(n_rows) for i in range(10)}
    X = pd.DataFrame(data)

    # Execution & Verification
    with pytest.raises(ValueError, match="CRITICAL: .* features are constant"):
        lgb_model_instance._validate_training_data(X)


def test_predict_warns_on_feature_mismatch_and_returns_array(
    lgb_model_instance, sample_tabular_data
):
    """Setup: train a minimal model object stub and ensure predict handles feature mismatch."""

    X = sample_tabular_data["X"].iloc[:30]
    sample_tabular_data["y"].iloc[:30]

    # Create a simple trained model stub with expected attributes
    class DummyModel:
        def __init__(self):
            self.best_iteration = 1

        def predict(self, X_input, num_iteration=None, pred_leaf=False):
            # return zeros for predictions or leaf indices depending on pred_leaf
            if pred_leaf:
                return np.zeros((len(X_input), 10), dtype=int)
            return np.zeros(len(X_input))

    lgb_model_instance.model = DummyModel()
    lgb_model_instance.feature_names = ["non_matching_feature"]

    # Execution
    preds = lgb_model_instance.predict(X)

    # Verification
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]


# New tests: fully mock LightGBM internals and MLflow interactions


def test_get_prediction_confidence_various_methods(
    lgb_model_instance, sample_tabular_data
):
    """Setup: attach a DummyModel that simulates leaf indices and predictions for different confidence methods."""

    X = sample_tabular_data["X"].iloc[:20]

    class DummyLGB:
        def __init__(self):
            self.best_iteration = 50

        def predict(self, X_input, pred_leaf=False, num_iteration=None):
            if pred_leaf:
                # Simulate leaf indices: 50 trees, each returns leaf id 1..50
                return np.tile(np.arange(1, 11), (len(X_input), 1))
            # Return simple increasing predictions that depend on num_iteration
            base = np.arange(len(X_input)).astype(float)
            if num_iteration is None:
                return base + 0.5
            return base + float(num_iteration) * 0.01

    lgb_model_instance.model = DummyLGB()

    # Execution: leaf_depth
    conf_leaf = lgb_model_instance.get_prediction_confidence(X, method="leaf_depth")

    # Verification
    assert conf_leaf.min() >= 0.0 and conf_leaf.max() <= 1.0
    assert conf_leaf.shape[0] == X.shape[0]

    # Execution: margin
    conf_margin = lgb_model_instance.get_prediction_confidence(X, method="margin")

    # Verification
    assert conf_margin.min() >= 0.0 and conf_margin.max() <= 1.0

    # Execution: variance
    conf_variance = lgb_model_instance.get_prediction_confidence(X, method="variance")

    # Verification
    assert conf_variance.min() >= 0.0 and conf_variance.max() <= 1.0

    # Execution: simple
    conf_simple = lgb_model_instance.get_prediction_confidence(X, method="simple")

    # Verification
    assert conf_simple.min() >= 0.0 and conf_simple.max() <= 1.0


def test_log_model_to_mlflow_calls_helper(lgb_model_instance, sample_tabular_data):
    """Setup: patch the module-level logging helper to avoid real MLflow calls and verify run id is returned."""

    x_eval = sample_tabular_data["X"].iloc[:5]
    metrics = {"mse": 0.1}
    params = {"param": 1}

    with patch(
        "src.models.gradient_boosting.lightgbm_model.log_to_mlflow_lightgbm",
        return_value="run-123",
    ) as mock_log:
        # Execution
        # Ensure a model object exists to satisfy precondition
        lgb_model_instance.model = Mock()
        run_id = lgb_model_instance.log_model_to_mlflow(
            metrics=metrics, params=params, x_eval=x_eval, experiment_name="exp"
        )

        # Verification
        mock_log.assert_called_once()
        assert run_id == "run-123"


def test_load_model_uses_mlflow_client_and_handles_missing_signature(
    lgb_model_instance,
):
    """Setup: patch mlflow client and get_model_info to simulate missing signature and ensure no exception."""

    fake_run_id = "fake-run-1"

    fake_run_info = Mock()
    fake_run_info.data = Mock()
    fake_run_info.data.params = {
        "model_model_name": "name",
        "model_prediction_horizon": "10",
    }
    fake_run_info.info = Mock()
    fake_run_info.info.run_id = fake_run_id

    fake_artifacts = []

    fake_mlflow_client = Mock()
    fake_mlflow_client.get_run.return_value = fake_run_info
    fake_mlflow_client.list_artifacts.return_value = fake_artifacts

    # Patch MlflowClient and lightgbm.load_model to avoid real calls
    with (
        patch(
            "src.models.gradient_boosting.lightgbm_model.mlflow.tracking.MlflowClient",
            return_value=fake_mlflow_client,
        ),
        patch(
            "src.models.gradient_boosting.lightgbm_model.mlflow.lightgbm.load_model",
            return_value=Mock(),
        ),
    ):
        # Execution: should not raise
        lgb_model_instance.load_model(fake_run_id)

        # Verification: model attribute set
        assert lgb_model_instance.model is not None

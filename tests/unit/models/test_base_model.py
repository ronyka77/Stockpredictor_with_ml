from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.models.base_model import BaseModel


class DummyModel(BaseModel):
    """Minimal concrete implementation for testing BaseModel behavior."""

    def _create_model(self, **kwargs):
        return Mock()

    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        # simple behavior: store feature names and mark trained
        self.feature_names = list(X.columns) if hasattr(X, "columns") else None
        self.model = Mock()

        # create predictable predict / predict_proba outputs based on X shape
        def predict_fn(inp):
            return np.zeros(len(inp), dtype=int)

        def predict_proba_fn(inp):
            return np.tile(np.array([[0.3, 0.7]]), (len(inp), 1))

        self.model.predict = Mock(side_effect=predict_fn)
        self.model.predict_proba = Mock(side_effect=predict_proba_fn)
        self.is_trained = True
        return self


@pytest.fixture
def sample_dataframe():
    # three rows, two feature columns
    return pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.1, 0.2, 0.3]})


@pytest.fixture
def sample_series():
    return pd.Series([0, 1, 0])


def test_predict_raises_if_untrained(sample_dataframe):
    # Setup
    model = DummyModel("dummy")

    # Execution / Verification
    with pytest.raises(ValueError, match="Model must be trained before making predictions"):
        model.predict(sample_dataframe)


def test_predict_uses_feature_order_and_returns_array(sample_dataframe, sample_series):
    # Setup
    model = DummyModel("dummy")
    model.fit(sample_dataframe, sample_series)

    # Execution
    preds = model.predict(sample_dataframe)

    # Verification
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == sample_dataframe.shape[0]


def test_predict_proba_when_supported(sample_dataframe, sample_series):
    # Setup
    model = DummyModel("dummy")
    model.fit(sample_dataframe, sample_series)

    # Execution
    proba = model.predict_proba(sample_dataframe)

    # Verification
    assert isinstance(proba, np.ndarray)
    assert proba.shape == (sample_dataframe.shape[0], 2)


@pytest.mark.parametrize("feature_order", [("f1", "f2"), ("f2", "f1")])
def test_feature_name_order_enforced(sample_dataframe, sample_series, feature_order):
    # Setup
    df = sample_dataframe.copy()
    df = df[list(feature_order)]

    model = DummyModel("dummy")
    model.fit(sample_dataframe, sample_series)

    # Execution - call predict with columns reordered
    preds = model.predict(df)

    # Verification
    assert preds.shape[0] == df.shape[0]


def test_save_model_requires_trained():
    # Setup
    model = DummyModel("dummy")

    # Execution / Verification: saving untrained should raise
    with pytest.raises(ValueError, match="Cannot save untrained model"):
        model.save_model("exp")


def test_save_and_load_model_interacts_with_mlflow(sample_dataframe, sample_series):
    # Setup
    model = DummyModel("dummy")
    model.fit(sample_dataframe, sample_series)

    fake_mlflow = Mock()
    fake_mlflow.setup_experiment = Mock()
    fake_run = Mock()
    fake_run.info.run_id = "run-123"
    fake_mlflow.start_run = Mock(return_value=fake_run)
    fake_mlflow.log_params = Mock()
    fake_mlflow.log_model = Mock()
    fake_mlflow.load_sklearn_model = Mock(return_value=Mock())
    fake_mlflow.get_run = Mock(
        return_value=Mock(
            data=Mock(
                params={
                    "model_name": "dummy",
                    "config": "{}",
                    "feature_names": str(list(model.feature_names)),
                    "is_trained": "True",
                }
            )
        )
    )

    model.mlflow_integration = fake_mlflow

    # Execution
    run_id = model.save_model("exp")

    # Verification
    assert run_id == "run-123"
    fake_mlflow.setup_experiment.assert_called_once_with("exp")
    fake_mlflow.start_run.assert_called_once()
    fake_mlflow.log_params.assert_called()
    fake_mlflow.log_model.assert_called()

    # Execution: load the model back
    loaded = model.load_model("run-123")

    # Verification: load returns self with run_id set and feature_names parsed
    assert loaded.run_id == "run-123"
    assert isinstance(loaded.feature_names, list)


def test_predict_with_threshold_defaults_and_delegation(sample_dataframe, sample_series):
    # Setup
    model = DummyModel("dummy")
    model.fit(sample_dataframe, sample_series)

    # Replace threshold_evaluator with a mock to capture calls
    fake_evaluator = Mock()
    fake_evaluator.predict_with_threshold = Mock(return_value={"preds": np.array([0, 0, 0])})
    model.threshold_evaluator = fake_evaluator

    # Execution
    result = model.predict_with_threshold(sample_dataframe, return_confidence=False)

    # Verification: ensure evaluator was called with the model and defaults
    fake_evaluator.predict_with_threshold.assert_called_once()
    assert "preds" in result

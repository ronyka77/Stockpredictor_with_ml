import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock

from src.models.gradient_boosting.random_forest_model import RandomForestModel


def test_rf_fit_and_predict_basic(rf_model_instance, sample_tabular_data):
    """Setup: train RandomForestModel on sample data and verify predictions shape and type."""

    X = sample_tabular_data["X"].iloc[:80]
    y = sample_tabular_data["y"].iloc[:80]


    # Execution
    rf_model_instance.fit(X, y)


    # Verification
    preds = rf_model_instance.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]


@pytest.mark.parametrize("n_estimators", [50, 100])
def test_rf_confidence_variance_scales_with_estimators(rf_model_instance, sample_tabular_data, n_estimators):
    """Setup: Train with different n_estimators and ensure confidence arrays are valid and normalized."""

    X = sample_tabular_data["X"].iloc[:120]
    y = sample_tabular_data["y"].iloc[:120]


    # Execution
    rf_model_instance.fit(X, y, n_estimators=n_estimators)


    # Verification
    conf = rf_model_instance.get_prediction_confidence(X)
    assert isinstance(conf, np.ndarray)
    assert conf.min() >= 0.0 and conf.max() <= 1.0
    assert conf.shape[0] == X.shape[0]


def test_rf_evaluate_uses_threshold_evaluator(rf_model_instance, sample_tabular_data, mock_threshold_evaluator):
    """Setup: Inject a mock threshold evaluator and ensure evaluate delegates and returns expected keys."""

    X = sample_tabular_data["X"].iloc[:90]
    y = sample_tabular_data["y"].iloc[:90]

    rf_model_instance.threshold_evaluator = mock_threshold_evaluator


    # Execution
    # Train model to satisfy evaluate precondition and provide current prices
    rf_model_instance.fit(X, y)
    current_prices = X["close"].values
    metrics = rf_model_instance.evaluate(X, y, current_prices=current_prices)


    # Verification
    assert metrics.get("threshold_optimized") is True
    assert "optimal_threshold" in metrics
    assert metrics["threshold_investment_success_rate"] == mock_threshold_evaluator.optimize_prediction_threshold.return_value["best_result"]["investment_success_rate"]


def test_predict_raises_when_untrained(rf_model_instance, sample_tabular_data):
    """Execution & Verification: calling predict on untrained model raises clear error."""

    X = sample_tabular_data["X"].iloc[:10]

    with pytest.raises(ValueError, match="Model must be trained"):
        rf_model_instance.predict(X)

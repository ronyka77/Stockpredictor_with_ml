import numpy as np
import pandas as pd
import pytest


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
def test_rf_confidence_variance_scales_with_estimators(
    rf_model_instance, sample_tabular_data, n_estimators
):
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


def test_get_feature_importance(rf_model_instance, sample_tabular_data):
    """Test feature importance extraction"""
    X = sample_tabular_data["X"].iloc[:50]
    y = sample_tabular_data["y"].iloc[:50]

    rf_model_instance.fit(X, y)
    importance_df = rf_model_instance.get_feature_importance()

    assert isinstance(importance_df, pd.DataFrame)
    assert len(importance_df) == X.shape[1]
    assert "feature" in importance_df.columns
    assert "importance" in importance_df.columns

    # Check that importance values are valid
    assert (importance_df["importance"] >= 0).all()
    assert importance_df["importance"].sum() > 0  # At least some importance


def test_predict_proba_raises_not_implemented(rf_model_instance, sample_tabular_data):
    """Test predict_proba raises NotImplementedError for regression models"""
    X = sample_tabular_data["X"].iloc[:30]
    y = sample_tabular_data["y"].iloc[:30]

    rf_model_instance.fit(X, y)

    # predict_proba should raise NotImplementedError for regression models
    with pytest.raises(NotImplementedError, match="does not support predict_proba"):
        rf_model_instance.predict_proba(X)


def test_save_model_creates_run_id(rf_model_instance, sample_tabular_data):
    """Test model save functionality creates a run ID"""
    X = sample_tabular_data["X"].iloc[:30]
    y = sample_tabular_data["y"].iloc[:30]

    rf_model_instance.fit(X, y)

    # Save model
    run_id = rf_model_instance.save_model()

    # Verify we got a run ID
    assert isinstance(run_id, str)
    assert len(run_id) > 0


def test_evaluate_with_threshold_evaluator(
    rf_model_instance, sample_tabular_data, mock_threshold_evaluator
):
    """Test evaluation using threshold evaluator"""
    X = sample_tabular_data["X"].iloc[:40]
    y = sample_tabular_data["y"].iloc[:40]
    current_prices = np.array([10.0] * 40)

    rf_model_instance.fit(X, y)
    rf_model_instance.threshold_evaluator = mock_threshold_evaluator

    # Test evaluate method with current prices (triggers threshold evaluation)
    results = rf_model_instance.evaluate(X, y, current_prices=current_prices)

    # Verify threshold evaluator was called
    mock_threshold_evaluator.optimize_prediction_threshold.assert_called_once()

    assert isinstance(results, dict)


def test_get_best_trial_info(rf_model_instance, sample_tabular_data):
    """Test getting best trial information after hyperparameter tuning"""
    X = sample_tabular_data["X"].iloc[:30]
    y = sample_tabular_data["y"].iloc[:30]

    rf_model_instance.fit(X, y)

    # This tests the method exists and returns a dict
    trial_info = rf_model_instance.get_best_trial_info()

    assert isinstance(trial_info, dict)


def test_rf_evaluate_uses_threshold_evaluator(
    rf_model_instance, sample_tabular_data, mock_threshold_evaluator
):
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
    assert (
        metrics["threshold_investment_success_rate"]
        == mock_threshold_evaluator.optimize_prediction_threshold.return_value["best_result"][
            "investment_success_rate"
        ]
    )


def test_predict_raises_when_untrained(rf_model_instance, sample_tabular_data):
    """Execution & Verification: calling predict on untrained model raises clear error."""

    X = sample_tabular_data["X"].iloc[:10]

    with pytest.raises(ValueError, match="Model must be trained"):
        rf_model_instance.predict(X)

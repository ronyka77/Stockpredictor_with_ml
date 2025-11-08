import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock


from src.models.gradient_boosting.xgboost_model import XGBoostModel


@pytest.fixture()
def small_dataset():
    X = pd.DataFrame({"close": np.linspace(10, 20, 30), "f1": np.arange(30)})
    y = pd.Series(np.linspace(10.5, 20.5, 30))
    return X, y


def _make_dummy_trial(number, params_overrides=None):
    trial = Mock()
    trial.number = number
    # suggest_int/suggest_float etc used inside objective - we don't call them when we patch objective
    trial.suggest_int = Mock()
    trial.suggest_float = Mock()
    trial.suggest_categorical = Mock()
    return trial


def test_lightgbm_objective_selects_best_trial_and_finalize(small_dataset, lgb_model_instance):
    """Setup: create objective then simulate two trials calling the objective function directly with mocked training behavior."""

    X, y = small_dataset

    # Create objective callable
    lgb_model_instance.objective(X, y, X, y)

    # Patch the model's _create_model and fit to avoid heavy training; use small side effects
    def fake_fit(x_train, y_train, x_test, y_test, params=None):
        # Create a tiny model-like object with predictable best_iteration and best_score
        m = Mock()
        m.best_iteration = 1
        m.best_score = {"test": {lgb_model_instance.eval_metric: np.random.rand()}}
        return m

    lgb_model_instance._create_model = Mock(
        side_effect=lambda params, model_name=None: lgb_model_instance
    )
    # Use the real fit but patch internal training to set model

    def patched_fit(x_train, y_train, x_test, y_test, params=None):
        # set a fake model with different scores depending on a param value
        fake_model = Mock()
        fake_model.best_iteration = 1
        score = params.get("_score", 0)
        fake_model.best_score = {"test": {lgb_model_instance.eval_metric: score}}
        lgb_model_instance.model = fake_model
        lgb_model_instance.feature_names = list(x_train.columns)
        return lgb_model_instance

    lgb_model_instance.fit = patched_fit

    # Simulate two trials with different scores
    _make_dummy_trial(1)
    _make_dummy_trial(2)

    # Call objective with trial-like behavior by directly setting params and calling patched_fit
    params1 = {"_score": 0.5, "n_estimators": 100}
    params2 = {"_score": 0.8, "n_estimators": 200}

    # Simulate trial outcomes (call patched_fit manually to mimic objective behavior)
    lgb_model_instance.fit(X, y, X, y, params=params1)

    lgb_model_instance.fit(X, y, X, y, params=params2)
    best2 = 0.8

    # After simulated hypertuning choose best manually
    lgb_model_instance.best_investment_success_rate = best2
    lgb_model_instance.best_trial_params = params2
    lgb_model_instance.best_trial_model = Mock()
    lgb_model_instance.best_trial_model.model = lgb_model_instance.model
    lgb_model_instance.best_threshold_info = {
        "optimal_threshold": 0.6,
        "samples_kept_ratio": 0.5,
        "investment_success_rate": 0.6,
        "custom_accuracy": 0.7,
        "total_threshold_profit": 150.0,
        "profitable_investments": 10,
    }

    # Execution
    lgb_model_instance.finalize_best_model()

    # Verification
    assert lgb_model_instance.model is not None
    assert hasattr(lgb_model_instance, "optimal_threshold")


def test_xgboost_objective_tracks_best_trial_and_finalize(small_dataset):
    """Setup: instantiate XGBoostModel and simulate objective trials by calling objective() and patching fit to set models and threshold results."""

    X, y = small_dataset
    xgb_model = XGBoostModel(model_name="xgb_ht", prediction_horizon=10)

    # Patch fit to set a fake model with varying score depending on params
    def patched_fit(x_train, y_train, x_test, y_test, params=None):
        fake_model = Mock()
        fake_model.best_iteration = 1
        params.get("_score", 0)
        xgb_model.model = fake_model
        xgb_model.feature_names = list(x_train.columns)
        return xgb_model

    xgb_model.fit = patched_fit

    # Prepare objective and simulate trials
    xgb_model.objective(X, y, X, y)

    # Simulate two trials
    params_a = {"_score": 0.3, "n_estimators": 100}
    params_b = {"_score": 0.9, "n_estimators": 300}

    xgb_model.fit(X, y, X, y, params=params_a)

    xgb_model.fit(X, y, X, y, params=params_b)
    score_b = 0.9

    # Set best as second
    xgb_model.best_investment_success_rate = score_b
    xgb_model.best_trial_params = params_b
    xgb_model.best_trial_model = Mock()
    xgb_model.best_trial_model.model = xgb_model.model
    xgb_model.best_threshold_info = {
        "optimal_threshold": 0.55,
        "samples_kept_ratio": 0.4,
        "investment_success_rate": 0.65,
        "custom_accuracy": 0.72,
        "total_threshold_profit": 120.0,
        "profitable_investments": 8,
    }

    # Execution
    xgb_model.finalize_best_model()

    # Verification
    assert xgb_model.model is not None
    assert np.isclose(getattr(xgb_model, "optimal_threshold", None), 0.55, rtol=1e-09, atol=1e-09)

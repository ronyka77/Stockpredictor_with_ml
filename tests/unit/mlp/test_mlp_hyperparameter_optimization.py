import numpy as np
import optuna
import pandas as pd
import pytest

from src.models.evaluation import ThresholdEvaluator
from src.models.time_series.mlp.mlp_predictor import MLPPredictor


def create_test_data(n_samples=1000, n_features=20):
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=[f"feature_{i}" for i in range(n_features)])
    X["close"] = np.random.uniform(100, 200, n_samples)
    y = pd.Series(np.random.binomial(1, 0.3, n_samples), name="target")
    train_size = int(0.8 * n_samples)
    return X.iloc[:train_size], y.iloc[:train_size], X.iloc[train_size:], y.iloc[train_size:]


def test_mlp_hyperparameter_objective_callable():
    X_train, y_train, X_test, y_test = create_test_data(n_samples=200, n_features=10)
    threshold_evaluator = ThresholdEvaluator()

    from src.models.time_series.mlp.mlp_architecture import MLPDataUtils
    cleaned_train = MLPDataUtils.validate_and_clean_data(X_train)
    X_train_scaled, scaler = MLPDataUtils.scale_data(cleaned_train, None, True)
    cleaned_test = MLPDataUtils.validate_and_clean_data(X_test)
    X_test_scaled, _ = MLPDataUtils.scale_data(cleaned_test, scaler, False)

    from src.models.time_series.mlp.mlp_optimization import MLPPredictorWithOptimization
    opt_mixin = MLPPredictorWithOptimization(
        model_name="test_mlp",
        config={
            "input_size": X_train.shape[1],
            "output_size": 1,
            "task": "classification",
        },
        threshold_evaluator=threshold_evaluator,
    )

    objective = opt_mixin.objective(X_train, y_train, X_test, X_test_scaled, y_test, fitted_scaler=scaler)
    assert callable(objective)

@pytest.mark.slow
def test_mlp_hyperparameter_optimization_integration():
    X_train, y_train, X_test, y_test = create_test_data(n_samples=400, n_features=12)
    threshold_evaluator = ThresholdEvaluator()

    from src.models.time_series.mlp.mlp_architecture import MLPDataUtils
    cleaned_train = MLPDataUtils.validate_and_clean_data(X_train)
    X_train_scaled, scaler = MLPDataUtils.scale_data(cleaned_train, None, True)
    cleaned_test = MLPDataUtils.validate_and_clean_data(X_test)
    X_test_scaled, _ = MLPDataUtils.scale_data(cleaned_test, scaler, False)

    from src.models.time_series.mlp.mlp_optimization import MLPPredictorWithOptimization
    opt_mixin = MLPPredictorWithOptimization(
        model_name="test_mlp",
        config={"input_size": X_train.shape[1]},
        threshold_evaluator=threshold_evaluator,
    )

    objective = opt_mixin.objective(X_train, y_train, X_test, X_test_scaled, y_test, fitted_scaler=scaler)
    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=1)

    info = opt_mixin.get_best_trial_info()
    assert isinstance(info, dict)
    assert "best_investment_success_rate" in info
    assert "model_updated" in info



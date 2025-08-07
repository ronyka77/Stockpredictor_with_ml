"""
Test script for MLP Hyperparameter Optimization

This script tests the hyperparameter optimization functionality
implemented in the MLPPredictor class.
"""

import numpy as np
import optuna
import pandas as pd
import pytest

from src.models.evaluation import ThresholdEvaluator
from src.models.time_series.mlp.mlp_predictor import MLPPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_test_data(n_samples=1000, n_features=20):
    """Create synthetic test data for MLP training."""
    np.random.seed(42)

    # Create synthetic features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Add a 'close' column for current prices
    X["close"] = np.random.uniform(100, 200, n_samples)

    # Create synthetic targets (binary classification)
    y = pd.Series(np.random.binomial(1, 0.3, n_samples), name="target")

    # Split into train/test
    train_size = int(0.8 * n_samples)
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    return X_train, y_train, X_test, y_test


def test_mlp_hyperparameter_objective_callable():
    """Unit test: objective builder should return a callable without running Optuna."""

    X_train, y_train, X_test, y_test = create_test_data(n_samples=200, n_features=10)
    threshold_evaluator = ThresholdEvaluator()

    mlp_predictor = MLPPredictor(
        model_name="test_mlp",
        config={
            "input_size": X_train.shape[1],
            "output_size": 1,
            "task": "classification",
        },
        threshold_evaluator=threshold_evaluator,
    )

    # Fit scaler on cleaned training data and provide scaled test data to match strict API
    # Use MLPDataUtils for cleaning and scaling to produce required X_test_scaled
    from src.models.time_series.mlp.mlp_architecture import MLPDataUtils
    cleaned_train = MLPDataUtils.validate_and_clean_data(X_train)
    X_train_scaled, scaler = MLPDataUtils.scale_data(cleaned_train, None, True)
    cleaned_test = MLPDataUtils.validate_and_clean_data(X_test)
    X_test_scaled, _ = MLPDataUtils.scale_data(cleaned_test, scaler, False)

    # Tests expect objective on predictor to be available via the optimization mixin.
    # Create an optimization mixin and use it directly instead of relying on MLPPredictor wrappers.
    from src.models.time_series.mlp.mlp_optimization import MLPOptimizationMixin
    opt_mixin = MLPOptimizationMixin()
    opt_mixin.threshold_evaluator = mlp_predictor.threshold_evaluator
    opt_mixin.config = mlp_predictor.config
    opt_mixin.model_name = mlp_predictor.model_name
    opt_mixin.model = None

    objective = opt_mixin.objective(X_train, y_train, X_test, X_test_scaled, y_test, fitted_scaler=scaler)
    assert callable(objective)


@pytest.mark.slow
def test_mlp_hyperparameter_optimization_integration():
    """Small integration test for Optuna (tagged slow)."""
    X_train, y_train, X_test, y_test = create_test_data(n_samples=400, n_features=12)
    threshold_evaluator = ThresholdEvaluator()

    mlp_predictor = MLPPredictor(
        model_name="test_mlp",
        config={"input_size": X_train.shape[1]},
        threshold_evaluator=threshold_evaluator,
    )

    # Prepare cleaned and scaled test data and use the optimization mixin directly
    from src.models.time_series.mlp.mlp_architecture import MLPDataUtils
    cleaned_train = MLPDataUtils.validate_and_clean_data(X_train)
    X_train_scaled, scaler = MLPDataUtils.scale_data(cleaned_train, None, True)
    cleaned_test = MLPDataUtils.validate_and_clean_data(X_test)
    X_test_scaled, _ = MLPDataUtils.scale_data(cleaned_test, scaler, False)

    from src.models.time_series.mlp.mlp_optimization import MLPOptimizationMixin
    opt_mixin = MLPOptimizationMixin()
    opt_mixin.threshold_evaluator = mlp_predictor.threshold_evaluator
    opt_mixin.config = mlp_predictor.config
    opt_mixin.model_name = mlp_predictor.model_name

    objective = opt_mixin.objective(X_train, y_train, X_test, X_test_scaled, y_test, fitted_scaler=scaler)
    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    # Run just 1 trial to keep test short; heavier tests should be run separately
    study.optimize(objective, n_trials=1)

    # the optimization mixin should have best_trial info after optimization
    info = opt_mixin.get_best_trial_info()
    assert isinstance(info, dict)
    return info


def test_mlp_model_creation_for_tuning():
    """Test the _create_model_for_tuning method"""

    logger.info("Testing MLP model creation for tuning")

    # Create base MLP predictor
    base_config = {"input_size": 20, "output_size": 1, "task": "classification"}

    threshold_evaluator = ThresholdEvaluator()
    base_predictor = MLPPredictor(
        model_name="base_mlp",
        config=base_config,
        threshold_evaluator=threshold_evaluator,
    )

    # Test parameters for tuning
    tuning_params = {
        "layer_sizes": [128, 64],
        "learning_rate": 0.001,
        "dropout": 0.2,
        "batch_size": 32,
        "optimizer": "adam",
        "weight_decay": 1e-4,
        "activation": "relu",
        "epochs": 30,
        "batch_norm": True,
        "residual": False,
    }

    # Create tuned model
    # _create_model_for_tuning was removed in favor of direct model creation via MLPPredictor._create_model.
    # Construct a tuned predictor directly and verify its config merging behavior.
    tuned_predictor = MLPPredictor(model_name="tuned_mlp", config={**base_config, **tuning_params})

    # Verify the tuned predictor has the correct merged configuration
    for k, v in tuning_params.items():
        assert tuned_predictor.config.get(k) == v

    logger.info("✅ MLP model creation for tuning test completed successfully (refactored)")

    return tuned_predictor


def test_data_preparation():
    """Test the _prepare_data_for_training method"""

    logger.info("Testing data preparation for training")

    # Create test data
    X_train, y_train, X_test, y_test = create_test_data()

    # Create MLP predictor
    mlp_predictor = MLPPredictor(
        model_name="test_mlp",
        config={"input_size": X_train.shape[1]},
        threshold_evaluator=ThresholdEvaluator(),
    )

    # Test data preparation
    batch_size = 32
    # _prepare_data_for_training was removed; use MLPDataUtils to prepare loaders directly
    from src.models.time_series.mlp.mlp_architecture import MLPDataUtils
    X_train_clean = MLPDataUtils.validate_and_clean_data(X_train)
    X_train_scaled, scaler = MLPDataUtils.scale_data(X_train_clean, None, True)
    X_test_clean = MLPDataUtils.validate_and_clean_data(X_test)
    X_test_scaled, _ = MLPDataUtils.scale_data(X_test_clean, scaler, False)

    train_loader, val_loader = MLPDataUtils.create_train_val_dataloaders(
        X_train_scaled, y_train, X_test_scaled, y_test, batch_size
    )

    # Verify data loaders
    assert train_loader is not None
    assert val_loader is not None

    # Test a batch
    for batch_X, batch_y in train_loader:
        assert batch_X.shape[0] <= batch_size
        assert batch_X.shape[1] == X_train.shape[1]
        assert batch_y.shape[0] == batch_X.shape[0]
        break

    logger.info("✅ Data preparation test completed successfully")

    return train_loader, val_loader


# Module is test-only; run via pytest

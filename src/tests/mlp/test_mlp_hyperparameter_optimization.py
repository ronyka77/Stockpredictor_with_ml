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
    """Create synthetic test data for MLP training"""
    np.random.seed(42)
    
    # Create synthetic features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add a 'close' column for current prices
    X['close'] = np.random.uniform(100, 200, n_samples)
    
    # Create synthetic targets (binary classification)
    y = pd.Series(np.random.binomial(1, 0.3, n_samples), name='target')
    
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
            'input_size': X_train.shape[1],
            'output_size': 1,
            'task': 'classification'
        },
        threshold_evaluator=threshold_evaluator
    )

    objective = mlp_predictor.objective(X_train, y_train, X_test, y_test)
    assert callable(objective)


@pytest.mark.slow
def test_mlp_hyperparameter_optimization_integration():
    """Small integration test for Optuna (tagged slow)."""
    X_train, y_train, X_test, y_test = create_test_data(n_samples=400, n_features=12)
    threshold_evaluator = ThresholdEvaluator()

    mlp_predictor = MLPPredictor(
        model_name="test_mlp",
        config={'input_size': X_train.shape[1]},
        threshold_evaluator=threshold_evaluator,
    )

    objective = mlp_predictor.objective(X_train, y_train, X_test, y_test)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=42))
    # Run just 1 trial to keep test short; heavier tests should be run separately
    study.optimize(objective, n_trials=1)

    # the predictor should have best_trial info after optimization
    info = mlp_predictor.get_best_trial_info()
    assert isinstance(info, dict)
    return info


def test_mlp_model_creation_for_tuning():
    """Test the _create_model_for_tuning method"""
    
    logger.info("Testing MLP model creation for tuning")
    
    # Create base MLP predictor
    base_config = {
        'input_size': 20,
        'output_size': 1,
        'task': 'classification'
    }
    
    threshold_evaluator = ThresholdEvaluator()
    base_predictor = MLPPredictor(
        model_name="base_mlp",
        config=base_config,
        threshold_evaluator=threshold_evaluator
    )
    
    # Test parameters for tuning
    tuning_params = {
        'layer_sizes': [128, 64],
        'learning_rate': 0.001,
        'dropout': 0.2,
        'batch_size': 32,
        'optimizer': 'adam',
        'weight_decay': 1e-4,
        'activation': 'relu',
        'epochs': 30,
        'batch_norm': True,
        'residual': False
    }
    
    # Create tuned model
    tuned_model = base_predictor._create_model_for_tuning(
        params=tuning_params,
        model_name="tuned_mlp"
    )
    
    # Verify the tuned model has the correct parameters
    assert tuned_model.config['layer_sizes'] == [128, 64]
    assert tuned_model.config['learning_rate'] == 0.001
    assert tuned_model.config['dropout'] == 0.2
    assert tuned_model.config['batch_size'] == 32
    assert tuned_model.config['optimizer'] == 'adam'
    assert tuned_model.config['weight_decay'] == 1e-4
    assert tuned_model.config['activation'] == 'relu'
    assert tuned_model.config['epochs'] == 30
    assert tuned_model.config['batch_norm'] is True
    assert tuned_model.config['residual'] is False
    
    logger.info("✅ MLP model creation for tuning test completed successfully")
    
    return tuned_model


def test_data_preparation():
    """Test the _prepare_data_for_training method"""
    
    logger.info("Testing data preparation for training")
    
    # Create test data
    X_train, y_train, X_test, y_test = create_test_data()
    
    # Create MLP predictor
    mlp_predictor = MLPPredictor(
        model_name="test_mlp",
        config={'input_size': X_train.shape[1]},
        threshold_evaluator=ThresholdEvaluator()
    )
    
    # Test data preparation
    batch_size = 32
    train_loader, val_loader = mlp_predictor._prepare_data_for_training(
        X_train, y_train, X_test, y_test, batch_size
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
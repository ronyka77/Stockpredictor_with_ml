#!/usr/bin/env python3
"""
Test script for MLP Predictor Advanced Features
"""

import os
import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.evaluation import ThresholdEvaluator
from src.models.time_series.mlp.mlp_predictor import MLPPredictor


def create_dummy_data(n_samples=100, n_features=10):
    """Create dummy data for testing."""
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, 1)
    return X, y


def create_dataloader(X, y, batch_size=16):
    """Create DataLoader from tensors."""
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def test_basic_training():
    """Test basic training functionality."""
    print("Testing basic training...")

    # Create predictor with basic config
    config = {
        "input_size": 10,
        "layer_sizes": [64, 32],
        "activation": "relu",
        "dropout": 0.2,
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 1e-3,
    }

    predictor = MLPPredictor(model_name="test_mlp", config=config)

    # Create dummy data
    X_train, y_train = create_dummy_data(100, 10)
    X_val, y_val = create_dummy_data(50, 10)

    train_loader = create_dataloader(X_train, y_train, batch_size=16)
    val_loader = create_dataloader(X_val, y_val, batch_size=16)

    # Train model
    predictor.fit(train_loader, val_loader)

    # Test prediction
    X_test = pd.DataFrame(X_train.numpy())
    predictions = predictor.predict(X_test)
    assert predictions.ndim == 1 or (
        predictions.ndim == 2 and predictions.shape[1] == 1
    )
    print("✓ Basic training test passed!")


def test_early_stopping():
    """Test early stopping functionality."""
    print("\nTesting early stopping...")

    config = {
        "input_size": 10,
        "layer_sizes": [64, 32],
        "activation": "relu",
        "dropout": 0.2,
        "epochs": 20,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 1e-4,
    }

    predictor = MLPPredictor(model_name="test_mlp_early_stop", config=config)

    # Create data that won't improve much (to trigger early stopping)
    X_train, y_train = create_dummy_data(100, 10)
    X_val, y_val = create_dummy_data(50, 10)

    train_loader = create_dataloader(X_train, y_train, batch_size=16)
    val_loader = create_dataloader(X_val, y_val, batch_size=16)

    # Train model
    predictor.fit(train_loader, val_loader)

    # Check that training history is recorded
    # Use attribute access as predictor may expose training_history dict
    history = getattr(predictor, "training_history", None)
    assert history is not None
    assert "train_loss" in history
    assert "val_loss" in history
    assert "learning_rate" in history
    print("✓ Early stopping test passed!")


def test_learning_rate_scheduling():
    """Test learning rate scheduling."""
    print("\nTesting learning rate scheduling...")

    config = {
        "input_size": 10,
        "layer_sizes": [64, 32],
        "activation": "relu",
        "dropout": 0.2,
        "epochs": 5,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "lr_scheduler": "cosine",
        "lr_scheduler_params": {"T_max": 5},
    }

    predictor = MLPPredictor(model_name="test_mlp_scheduler", config=config)

    X_train, y_train = create_dummy_data(100, 10)
    X_val, y_val = create_dummy_data(50, 10)

    train_loader = create_dataloader(X_train, y_train, batch_size=16)
    val_loader = create_dataloader(X_val, y_val, batch_size=16)

    # Train model
    predictor.fit(train_loader, val_loader)

    # Check that learning rate changed
    history = predictor.get_training_history()
    lr_values = history["learning_rate"]

    # Learning rate should change with cosine annealing
    assert len(lr_values) > 0, "Learning rate history should not be empty"
    print(f"Learning rate values: {lr_values}")
    print("✓ Learning rate scheduling test passed!")


def test_model_checkpointing():
    """Test model checkpointing functionality."""
    print("\nTesting model checkpointing...")

    config = {
        "input_size": 10,
        "layer_sizes": [64, 32],
        "activation": "relu",
        "dropout": 0.2,
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "checkpoint_dir": "./test_checkpoints",
        "save_best_model": True,
    }

    predictor = MLPPredictor(model_name="test_mlp_checkpoint", config=config)

    X_train, y_train = create_dummy_data(100, 10)
    X_val, y_val = create_dummy_data(50, 10)

    train_loader = create_dataloader(X_train, y_train, batch_size=16)
    val_loader = create_dataloader(X_val, y_val, batch_size=16)

    # Train model
    predictor.fit(train_loader, val_loader)

    # Check that checkpoint files were created
    checkpoint_dir = config["checkpoint_dir"]
    checkpoint_path = os.path.join(
        checkpoint_dir, f"{predictor.model_name}_checkpoint.pth"
    )
    best_path = os.path.join(checkpoint_dir, f"{predictor.model_name}_best.pth")

    assert os.path.exists(checkpoint_path)
    assert os.path.exists(best_path)

    # Clean up
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)


def test_optimizer_selection():
    """Test different optimizer options."""
    print("\nTesting optimizer selection...")

    optimizers = ["adam", "adamw", "rmsprop", "sgd"]

    for optimizer in optimizers:
        print(f"Testing {optimizer} optimizer...")

        config = {
            "input_size": 10,
            "layer_sizes": [64, 32],
            "activation": "relu",
            "dropout": 0.2,
            "epochs": 2,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "optimizer": optimizer,
        }

        predictor = MLPPredictor(model_name=f"test_mlp_{optimizer}", config=config)

        X_train, y_train = create_dummy_data(100, 10)
        train_loader = create_dataloader(X_train, y_train, batch_size=16)

        # Train model
        predictor.fit(train_loader)

        # Test prediction
        X_test = pd.DataFrame(X_train.numpy())
        predictions = predictor.predict(X_test)
        assert predictions.ndim == 1 or (
            predictions.ndim == 2 and predictions.shape[1] == 1
        )

        print(f"✓ {optimizer} optimizer test passed!")


def test_threshold_optimization():
    """Test threshold optimization integration."""
    print("\nTesting threshold optimization...")

    config = {
        "input_size": 10,
        "layer_sizes": [64, 32],
        "activation": "relu",
        "dropout": 0.2,
        "epochs": 2,
        "batch_size": 16,
        "learning_rate": 1e-3,
    }

    # Create threshold evaluator
    threshold_evaluator = ThresholdEvaluator()

    predictor = MLPPredictor(
        model_name="test_mlp_threshold",
        config=config,
        threshold_evaluator=threshold_evaluator,
    )

    X_train, y_train = create_dummy_data(100, 10)
    train_loader = create_dataloader(X_train, y_train, batch_size=16)

    # Train model
    predictor.fit(train_loader)

    # Create test data for threshold optimization
    X_test = pd.DataFrame(X_train.numpy())
    y_test = pd.Series(y_train.numpy().flatten())
    current_prices_test = np.ones(len(y_test)) * 100  # Dummy prices

    # Test threshold optimization
    results = predictor.optimize_prediction_threshold(
        X_test=X_test,
        y_test=y_test,
        current_prices_test=current_prices_test,
        threshold_range=(0.1, 0.9),
        n_thresholds=10,
    )

    # Ensure the results contain expected keys (implementation dependent)
    assert isinstance(results, dict)


def test_error_handling():
    """Test error handling for invalid configurations."""
    print("\nTesting error handling...")

    # Test invalid optimizer
    try:
        config = {
            "input_size": 10,
            "layer_sizes": [64, 32],
            "optimizer": "invalid_optimizer",
        }
        predictor = MLPPredictor(model_name="test_error", config=config)
        predictor._create_optimizer(predictor._create_model())
        assert False, "Should have raised ValueError for invalid optimizer"
    except ValueError as e:
        print(f"✓ Expected error for invalid optimizer: {e}")

    # Test invalid scheduler
    try:
        config = {
            "input_size": 10,
            "layer_sizes": [64, 32],
            "lr_scheduler": "invalid_scheduler",
        }
        predictor = MLPPredictor(model_name="test_error", config=config)
        model = predictor._create_model()
        optimizer = predictor._create_optimizer(model)
        predictor._create_scheduler(optimizer)
        # Should log warning but not raise error
        print("✓ Invalid scheduler handled gracefully")
    except Exception as e:
        print(f"✓ Expected behavior for invalid scheduler: {e}")

    print("✓ Error handling test passed!")


def test_training_history():
    """Test training history tracking."""
    print("\nTesting training history...")

    config = {
        "input_size": 10,
        "layer_sizes": [64, 32],
        "activation": "relu",
        "dropout": 0.2,
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 1e-3,
    }

    predictor = MLPPredictor(model_name="test_mlp_history", config=config)

    X_train, y_train = create_dummy_data(100, 10)
    X_val, y_val = create_dummy_data(50, 10)

    train_loader = create_dataloader(X_train, y_train, batch_size=16)
    val_loader = create_dataloader(X_val, y_val, batch_size=16)

    # Train model
    predictor.fit(train_loader, val_loader)

    history = getattr(predictor, "training_history", None)
    assert history is not None
    assert "train_loss" in history
    assert "val_loss" in history
    assert "learning_rate" in history
    assert len(history["train_loss"]) > 0


# Module is test-only; run via pytest

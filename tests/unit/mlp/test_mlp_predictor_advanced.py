import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.evaluation import ThresholdEvaluator
from src.models.time_series.mlp.mlp_predictor import MLPPredictor


def create_dummy_data(n_samples=100, n_features=10):
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, 1)
    return X, y


def create_dataloader(X, y, batch_size=16):
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def test_basic_training():
    config = {"input_size": 10, "layer_sizes": [64, 32], "activation": "relu", "dropout": 0.2, "epochs": 3, "batch_size": 16, "learning_rate": 1e-3}
    predictor = MLPPredictor(model_name="test_mlp", config=config)
    X_train, y_train = create_dummy_data(100, 10)
    X_val, y_val = create_dummy_data(50, 10)
    train_loader = create_dataloader(X_train, y_train, batch_size=16)
    val_loader = create_dataloader(X_val, y_val, batch_size=16)
    predictor.fit(train_loader, val_loader)
    X_test = pd.DataFrame(X_train.numpy())
    predictions = predictor.predict(X_test)
    assert predictions.ndim == 1 or (predictions.ndim == 2 and predictions.shape[1] == 1)



import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from src.models.time_series.base_pytorch_model import PyTorchBasePredictor


class MinimalPytorchPredictor(PyTorchBasePredictor):
    def _create_model(self):
        # simple 1-layer linear model
        return nn.Sequential(nn.Linear(1, 1))


def _make_loader(n=20, batch_size=8):
    X = np.linspace(0, 1, n).reshape(-1, 1).astype(np.float32)
    y = (2 * X).reshape(-1).astype(np.float32)
    tensor_X = torch.tensor(X)
    tensor_y = torch.tensor(y)
    dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_fit_and_predict():
    """Train a minimal PyTorch predictor and verify predict returns correct shape."""
    cfg = {"epochs": 2, "learning_rate": 1e-2, "batch_size": 8}
    p = MinimalPytorchPredictor(model_name="m", config=cfg)
    train_loader = _make_loader(24, batch_size=8)
    # train
    p.fit(train_loader=train_loader, val_loader=None, feature_names=["x"], epochs=None)
    # mark trained
    assert p.is_trained

    # build features DataFrame for predict
    features = pd.DataFrame({"x": np.linspace(0, 1, 5)})
    preds = p.predict(features)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == 5


def test_get_prediction_confidence_simple_method():
    """Verify simple confidence method returns array matching input length."""
    cfg = {"epochs": 1, "learning_rate": 1e-2, "batch_size": 8}
    p = MinimalPytorchPredictor(model_name="m2", config=cfg)
    train_loader = _make_loader(16, batch_size=8)
    p.fit(train_loader=train_loader, val_loader=None, feature_names=["x"])
    features = pd.DataFrame({"x": np.linspace(0, 1, 4)})
    conf = p.get_prediction_confidence(features, method="simple")
    assert isinstance(conf, np.ndarray)
    assert conf.shape[0] == 4


def test_get_prediction_confidence_invalid_method_raises():
    """Invalid confidence method should raise ValueError."""
    cfg = {"epochs": 1, "learning_rate": 1e-2, "batch_size": 4}
    p = MinimalPytorchPredictor(model_name="m3", config=cfg)
    train_loader = _make_loader(8, batch_size=4)
    p.fit(train_loader=train_loader, val_loader=None, feature_names=["x"])
    features = pd.DataFrame({"x": np.linspace(0, 1, 2)})
    try:
        p.get_prediction_confidence(features, method="unknown")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_fit_with_huber_and_val_loader():
    """Training with Huber loss and validation loader runs without error and marks trained."""
    cfg = {
        "epochs": 1,
        "learning_rate": 1e-2,
        "batch_size": 8,
        "loss": "huber",
        "huber_delta": 0.5,
    }
    p = MinimalPytorchPredictor(model_name="huber", config=cfg)
    train_loader = _make_loader(16, batch_size=8)
    val_loader = _make_loader(8, batch_size=4)
    # Should run without raising and set is_trained
    p.fit(train_loader=train_loader, val_loader=val_loader, feature_names=["x"])
    assert p.is_trained


def test_predict_raises_when_not_trained():
    """Predict must raise when model has not been trained."""
    cfg = {"epochs": 1, "learning_rate": 1e-2}
    p = MinimalPytorchPredictor(model_name="notr", config=cfg)
    features = pd.DataFrame({"x": [0.1, 0.2]})
    raised = False
    try:
        p.predict(features)
    except ValueError:
        raised = True
    assert raised


def test_confidence_methods_variance_margin_leaf():
    """Validate variance, margin, and leaf_depth confidence methods produce bounded arrays."""
    cfg = {"epochs": 1, "learning_rate": 1e-2, "batch_size": 8}
    p = MinimalPytorchPredictor(model_name="conf", config=cfg)
    train_loader = _make_loader(20, batch_size=10)
    p.fit(train_loader=train_loader, val_loader=None, feature_names=["x"])
    features = pd.DataFrame({"x": np.linspace(0, 1, 6)})

    for method in ("variance", "margin", "leaf_depth"):
        conf = p.get_prediction_confidence(features, method=method)
        assert isinstance(conf, np.ndarray)
        assert conf.shape[0] == 6
        assert (conf >= 0).all() and (conf <= 1).all()

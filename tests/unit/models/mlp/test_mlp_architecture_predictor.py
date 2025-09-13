import os
import numpy as np
import pandas as pd
import torch

from src.models.time_series.mlp.mlp_architecture import MLPModule, MLPDataUtils
from src.models.time_series.mlp.mlp_predictor import MLPPredictor


def test_mlp_forward_and_architecture_info():
    # Small MLP for testing
    input_size = 5
    layer_sizes = [4, 3]
    output_size = 2

    model = MLPModule(
        input_size=input_size,
        layer_sizes=layer_sizes,
        output_size=output_size,
        dropout=0.0,
    )

    # Batch input
    x = torch.randn(7, input_size)
    out = model(x)
    if out.shape != (7, output_size):
        raise AssertionError("MLP forward output shape mismatch")

    # Single-dimension input should be expanded to batch dim
    x1 = torch.randn(input_size)
    out1 = model(x1)
    if out1.shape[0] != 1:
        raise AssertionError("Single-dim input not expanded to batch dimension")

    info = model.get_architecture_info()
    if info.get("input_size") != input_size:
        raise AssertionError("Architecture info input_size mismatch")
    if info.get("output_size") != output_size:
        raise AssertionError("Architecture info output_size mismatch")
    if info.get("num_layers") != len(model.layers):
        raise AssertionError("Architecture info num_layers mismatch")


def test_mlppredictor_fit_predict_and_confidence(tmp_path):
    # Create tiny synthetic dataset
    n_samples = 10
    n_features = 3
    # use local RNG to avoid global state
    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        rng.randn(n_samples, n_features), columns=[f"f{i}" for i in range(n_features)]
    )
    y = pd.Series(rng.randn(n_samples))

    # Scale training data and create dataloader
    X_scaled, scaler = MLPDataUtils.scale_data(X, scaler=None, fit_scaler=True)
    train_loader = MLPDataUtils.create_dataloader_from_dataframe(
        X_scaled, y, batch_size=4, shuffle=False, num_workers=0
    )

    config = {
        "input_size": n_features,
        "epochs": 1,
        "batch_size": 4,
        "checkpoint_dir": str(tmp_path),
        "save_best_model": False,
        "save_checkpoint_frequency": 1,
    }

    predictor = MLPPredictor(config=config)

    # Fit should complete quickly for 1 epoch
    predictor.fit(
        train_loader, val_loader=None, scaler=scaler, feature_names=list(X.columns)
    )

    if predictor.is_trained is not True:
        raise AssertionError("Predictor should be marked trained after fit")

    # Predictions should return correct length
    preds = predictor.predict(X)
    if not isinstance(preds, np.ndarray):
        raise AssertionError("Predictions should be a numpy ndarray")
    if preds.shape[0] != n_samples:
        raise AssertionError("Predictions length mismatch")

    # Confidence methods
    conf_simple = predictor.get_prediction_confidence(X, method="simple")
    if conf_simple.shape[0] != n_samples:
        raise AssertionError("Simple confidence length mismatch")

    conf_margin = predictor.get_prediction_confidence(X, method="margin")
    if conf_margin.shape[0] != n_samples:
        raise AssertionError("Margin confidence length mismatch")

    # Check that checkpoint file was created
    checkpoint_path = os.path.join(
        config["checkpoint_dir"], f"{predictor.model_name}_checkpoint.pth"
    )
    if not os.path.exists(checkpoint_path):
        raise AssertionError("Checkpoint file was not created")

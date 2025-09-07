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
    assert out.shape == (7, output_size)

    # Single-dimension input should be expanded to batch dim
    x1 = torch.randn(input_size)
    out1 = model(x1)
    assert out1.shape[0] == 1

    info = model.get_architecture_info()
    assert info["input_size"] == input_size
    assert info["output_size"] == output_size
    assert info["num_layers"] == len(model.layers)


def test_mlppredictor_fit_predict_and_confidence(tmp_path):
    # Create tiny synthetic dataset
    n_samples = 10
    n_features = 3
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

    assert predictor.is_trained is True

    # Predictions should return correct length
    preds = predictor.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == n_samples

    # Confidence methods
    conf_simple = predictor.get_prediction_confidence(X, method="simple")
    assert conf_simple.shape[0] == n_samples

    conf_margin = predictor.get_prediction_confidence(X, method="margin")
    assert conf_margin.shape[0] == n_samples

    # Check that checkpoint file was created
    checkpoint_path = os.path.join(
        config["checkpoint_dir"], f"{predictor.model_name}_checkpoint.pth"
    )
    assert os.path.exists(checkpoint_path)

import torch
import pytest
import numpy as np
import pandas as pd

from src.models.time_series.mlp.mlp_architecture import MLPModule
from src.models.time_series.mlp.mlp_predictor import MLPPredictor


@pytest.mark.parametrize(
    "activation",
    ["relu", "leaky_relu", "elu", "gelu"],
)
def test_activation_output_shape(activation):
    input_size = 5
    layer_sizes = [10, 5]
    batch_size = 8

    model = MLPModule(
        input_size=input_size,
        layer_sizes=layer_sizes,
        activation=activation,
        dropout=0.1,
    )
    x = torch.randn(batch_size, input_size)
    out = model(x)
    assert out.shape == (batch_size, 1)


def test_residual_and_architecture_info():
    input_size = 6
    layer_sizes = [6, 4]
    model = MLPModule(
        input_size=input_size,
        layer_sizes=layer_sizes,
        residual=True,
        batch_norm=True,
    )

    x = torch.randn(4, input_size)
    out = model(x)
    assert out.shape == (4, 1)

    info = model.get_architecture_info()
    assert info["input_size"] == input_size
    assert info["layer_sizes"] == layer_sizes
    assert "total_parameters" in info


def test_invalid_configurations_raise():
    with pytest.raises(ValueError):
        MLPModule(input_size=10, layer_sizes=[], output_size=1)

    with pytest.raises(ValueError):
        MLPModule(input_size=10, layer_sizes=[64], task="invalid")


def test_predictor_create_model_and_predict_raises_for_untrained():
    cfg = {"input_size": 10, "layer_sizes": [32, 16], "epochs": 1}
    predictor = MLPPredictor(model_name="t", config=cfg)
    model = predictor._create_model()
    assert isinstance(model, MLPModule)

    X = pd.DataFrame(np.random.randn(5, 10))
    with pytest.raises(ValueError):
        predictor.predict(X)

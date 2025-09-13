import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def rng_seed():
    # Maintain backwards compatibility: set global torch RNG but provide local numpy RNG use
    torch.manual_seed(42)
    # Return a deterministic numpy RandomState for tests to use (avoid global seed mutation)
    return np.random.RandomState(42)


@pytest.fixture
def small_df(rng_seed):
    rng = rng_seed
    data = rng.randn(16, 5)
    return pd.DataFrame(data, columns=[f"f{i}" for i in range(5)])


@pytest.fixture
def small_Xy(small_df, rng_seed):
    rng = rng_seed
    X = small_df.copy()
    y = pd.Series(rng.randn(len(X)))
    return X, y


@pytest.fixture
def cpu_device():
    return torch.device("cpu")


class StubModel:
    def eval(self):
        return None

    def __call__(self, x):
        if hasattr(x, "shape"):
            n = x.shape[0]
        else:
            n = len(x)
        return torch.tensor([[0.1]] * n, dtype=torch.float32)


@pytest.fixture
def stub_model():
    return StubModel()

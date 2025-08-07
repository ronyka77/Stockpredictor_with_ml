import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def rng_seed():
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture
def small_df(rng_seed):
    # Small deterministic DataFrame for unit tests
    data = np.random.randn(16, 5)
    return pd.DataFrame(data, columns=[f"f{i}" for i in range(5)])


@pytest.fixture
def small_Xy(small_df):
    X = small_df.copy()
    y = pd.Series(np.random.randn(len(X)))
    return X, y


@pytest.fixture
def cpu_device():
    return torch.device("cpu")


class StubModel:
    """Simple deterministic model stub that returns a constant prediction."""

    def eval(self):
        return None

    def __call__(self, x):
        # Accept pandas.DataFrame or torch.Tensor
        if hasattr(x, "shape"):
            n = x.shape[0]
        else:
            n = len(x)
        return torch.tensor([[0.1]] * n, dtype=torch.float32)


@pytest.fixture
def stub_model():
    return StubModel()

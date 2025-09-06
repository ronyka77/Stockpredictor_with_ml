import pandas as pd
import torch
import pytest
from src.data_utils.sequential_data_loader import TimeSeriesDataset


def test_timeseriesdataset_constructor_validates_types_and_sequence_length():
    with pytest.raises(TypeError):
        TimeSeriesDataset(features="not a df", targets="not a series")
    with pytest.raises(ValueError):
        TimeSeriesDataset(pd.DataFrame({"a":[1]}), pd.Series([1]), sequence_length=0)


def test_timeseriesdataset_len_and_getitem_shapes_and_indexerror():
    n = 10
    seq = 3
    df = pd.DataFrame({"f0": range(n), "f1": range(n)})
    y = pd.Series(range(n))
    ds = TimeSeriesDataset(df, y, sequence_length=seq)
    assert len(ds) == n - seq, f"Expected len={n - seq}, got {len(ds)}"
    X, yt = ds[0]
    assert isinstance(X, torch.Tensor) and X.shape == (seq, 2), f"Unexpected X shape {X.shape}"
    assert isinstance(yt, torch.Tensor), "Target not a tensor"
    with pytest.raises(IndexError):
        _ = ds[len(ds)]  # out of range



from hypothesis import given, settings, strategies as st
import pandas as pd
import numpy as np
import pytest

from src.models.time_series.mlp.mlp_architecture import MLPDataUtils


@pytest.mark.unit
@settings(max_examples=20, deadline=None)
@given(
    n_rows=st.integers(min_value=1, max_value=20),
    n_cols=st.integers(min_value=1, max_value=10),
    nan_probability=st.floats(min_value=0.0, max_value=0.5),
)
def test_validate_and_clean_data_property(n_rows, n_cols, nan_probability):
    """Property: validate_and_clean_data returns DataFrame free of NaN/Inf values."""

    rng = np.random.RandomState(42)
    data = rng.normal(size=(n_rows, n_cols))

    # Introduce NaN/Inf values probabilistically
    mask = rng.random(size=data.shape) < nan_probability
    data[mask] = rng.choice([np.nan, np.inf, -np.inf], size=mask.sum())

    df = pd.DataFrame(data, columns=[f"f{i}" for i in range(n_cols)])

    cleaned = MLPDataUtils.validate_and_clean_data(df)

    # Cleaned should contain no NaN or infinite values
    if cleaned.isnull().any().any():
        raise AssertionError("Cleaned data contains NaNs")
    if np.isinf(cleaned.values).any():
        raise AssertionError("Cleaned data contains infinite values")

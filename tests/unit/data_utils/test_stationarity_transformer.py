import pandas as pd
import numpy as np
import pytest

from src.data_utils import stationarity_transformer as st


def test_transform_to_stationary_skips_short_series(mocker):
    s = pd.Series([1.0] * 5, name="short")
    # should return original series and None for short series
    out, transform = st.transform_to_stationary(s, verbose=False)
    assert transform is None and out.equals(s), "Short series should be skipped (no transformation)"


def test_transform_to_stationary_selects_best_transformation_by_mocking_kpss(mocker):
    # create a longer series
    s = pd.Series(np.linspace(1, 100, 50), name="trend")
    # Patch kpss to return non-stationary for original, but stationary for pct-change
    # Sequence of kpss calls inside transform_to_stationary: original, pct, diff -> supply side_effect accordingly
    mocker.patch("src.data_utils.stationarity_transformer.kpss", side_effect=[
        (0.5, 0.01),   # original -> p=0.01 non-stationary
        (0.1, 0.1),    # pct -> p=0.1 stationary (statistic 0.1)
        (0.2, 0.02)    # diff -> p=0.02 non-stationary
    ])
    out_series, transform_name = st.transform_to_stationary(s, verbose=False)
    assert transform_name in ("percentage_change", "first_difference"), f"Unexpected transform: {transform_name}"
    assert isinstance(out_series, pd.Series), "Returned series not a pandas Series"


def test_inverse_transform_series_restores_percentage_change_and_difference():
    # Create an original series with deterministic values
    orig = pd.Series([100.0, 110.0, 121.0, 133.1], name="p")
    # percentage change transformation (NaN at first position)
    pct = orig.pct_change()
    pct.iloc[0] = np.nan
    inv_pct = st.inverse_transform_series(pct, orig, "percentage_change")
    # last element should match original last value approximately
    assert inv_pct.iloc[-1] == pytest.approx(orig.iloc[-1], rel=1e-6), f"Inverse pct change did not restore value: {inv_pct.iloc[-1]} vs {orig.iloc[-1]}"

    # first difference transformation
    diff = orig.diff()
    diff.iloc[0] = np.nan
    inv_diff = st.inverse_transform_series(diff, orig, "first_difference")
    assert inv_diff.iloc[-1] == pytest.approx(orig.iloc[-1], rel=1e-6), "Inverse diff did not restore original series"



import pandas as pd
import numpy as np
import pytest
import types

from src.data_utils.stationarity_transformer import (
    transform_dataframe_to_stationary,
    transform_to_stationary,
    inverse_transform_series,
)


def test_transform_to_stationary_detects_stationary(mocker):
    ser = pd.Series(np.ones(20), name='s')
    # Patch kpss to return a p-value >= 0.05 so the series is considered stationary
    mocker.patch('src.data_utils.stationarity_transformer.kpss', return_value=(0.1, 0.1))
    # Replace module logger with a simple object that supports setLevel used in code
    simple_logger = types.SimpleNamespace(setLevel=lambda lvl: None, info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)
    mocker.patch('src.data_utils.stationarity_transformer.logger', simple_logger)
    out, name = transform_to_stationary(ser, verbose=False)
    assert name == 'none' and out.equals(ser), f"Expected no transformation for stationary series, got {name}"


def test_inverse_transform_series_percentage_change():
    idx = pd.date_range('2020-01-01', periods=3)
    original = pd.Series([100.0, 110.0, 121.0], index=idx, name='price')
    transformed = pd.Series([float('nan'), 0.1, 0.1], index=idx, name='price')
    inversed = inverse_transform_series(transformed, original, 'percentage_change')
    expected = pd.Series([110.0, 121.0], index=idx[1:])
    pd.testing.assert_series_equal(inversed, expected, check_names=False)


def test_inverse_transform_series_first_difference():
    idx = pd.date_range('2020-01-01', periods=3)
    original = pd.Series([100.0, 110.0, 121.0], index=idx, name='price')
    transformed = pd.Series([float('nan'), 10.0, 11.0], index=idx, name='price')
    inversed = inverse_transform_series(transformed, original, 'first_difference')
    expected = pd.Series([110.0, 121.0], index=idx[1:])
    pd.testing.assert_series_equal(inversed, expected, check_names=False)


def test_transform_dataframe_to_stationary_nan_handling_drop(mocker):
    df = pd.DataFrame({'a': [100, 110, 121], 'b': [1, 2, 3]}, index=pd.date_range('2020-01-01', periods=3))
    # The transformer will call _kpss_test multiple times (initial, pct_change, diff).
    # Provide a deterministic sequence so 'a' and 'b' are treated as non-stationary then
    # pct_change becomes stationary. The exact counts depend on implementation; provide
    # a sequence long enough for the calls.
    side = [(0.0, 1), (0.0, 1), (0.06, 1), (0.06, 1), (0.06, 1), (0.06, 1)]
    mocker.patch('src.data_utils.stationarity_transformer._kpss_test', side_effect=side)
    # Replace module logger with a simple object that supports setLevel used in code
    simple_logger = types.SimpleNamespace(setLevel=lambda lvl: None, info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)
    mocker.patch('src.data_utils.stationarity_transformer.logger', simple_logger)
    transformed, manifest, failed = transform_dataframe_to_stationary(df.copy(), n_jobs=1, verbose=False, nan_handling='drop')
    assert isinstance(manifest, dict), 'Expected transformation manifest to be a dict'
    assert 'a' in manifest, "Expected column 'a' to be processed"
    assert transformed.isna().sum().sum() == 0, "NaN handling 'drop' should remove NaNs"


def test_transform_dataframe_to_stationary_returns_expected_types_and_manifest():
    df = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': [10.0, 10.0, 10.0]})

    X_stat, manifest, transforms = transform_dataframe_to_stationary(df.copy())

    assert isinstance(X_stat, pd.DataFrame), "Expected DataFrame from transform_dataframe_to_stationary"
    assert isinstance(manifest, dict), "Expected manifest dict from transform_dataframe_to_stationary"
    assert isinstance(transforms, list), "Expected list of transforms returned"

    # All numeric columns should be float64
    for col in X_stat.columns:
        assert X_stat[col].dtype == np.float64, f"Column {col} dtype expected float64, got {X_stat[col].dtype}"


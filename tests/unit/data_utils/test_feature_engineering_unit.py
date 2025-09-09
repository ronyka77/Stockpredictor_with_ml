import pandas as pd
import numpy as np
import pytest

from src.data_utils.feature_engineering import (
    add_price_normalized_features,
    add_prediction_bounds_features,
    clean_data_for_training,
    add_date_features,
)


def test_add_price_normalized_features_creates_expected_columns():
    df = pd.DataFrame({
        "close": [100.0, 200.0],
        "SMA_20": [50.0, 100.0],
        "BB_Lower": [90.0, 180.0],
        "BB_Width": [20.0, 40.0],
        "ATR": [2.0, 4.0],
        "volume": [1000, 2000],
        "Return_1D": [1.0, -2.0],
        "open": [90.0, 190.0],
    })

    out = add_price_normalized_features(df)

    expected_cols_subset = {"SMA_20_Ratio", "BB_Upper", "BB_Position", "Price_ATR_Ratio", "Return_Volume_Efficiency", "Close_Open_Ratio"}
    missing = expected_cols_subset - set(out.columns)
    assert not missing, f"Missing normalized columns: {missing}. Got columns: {list(out.columns)}"

    assert out.loc[0, "SMA_20_Ratio"] == pytest.approx(100.0 / 50.0), "SMA ratio incorrect"
    assert out.loc[1, "Price_ATR_Ratio"] == pytest.approx(200.0 / 4.0), "Price ATR ratio incorrect"


def test_add_prediction_bounds_features_computes_expected_values():
    df = pd.DataFrame({
        "ATR_Percent": [0.01, 0.02],
        "Return_5D": [0.05, 0.1],
        "Return_1D": [0.01, 0.02],
        "Vol_Regime_High": [2.0, 3.0],
        "Vol_Regime_Low": [1.0, 0.5],
        "BB_Percent": [0.02, 0.03],
        "RSI_14": [60, 40],
    })

    out = add_prediction_bounds_features(df)
    assert "Expected_Daily_Move" in out.columns, "Expected_Daily_Move missing"
    assert out.loc[1, "Expected_10D_Move"] == pytest.approx(0.02 * np.sqrt(10)), "Expected 10-day move incorrect"
    assert out.loc[0, "RSI_Mean_Reversion_Pressure"] == pytest.approx(abs(60 - 50) / 50), "RSI reversion pressure incorrect"


def test_clean_data_for_training_handles_infinite_and_nan_and_types():
    df = pd.DataFrame({
        "a": [1.0, np.inf, -np.inf, 1e40],
        "b": [np.nan, 2.0, 3.0, 4.0],
    })

    out = clean_data_for_training(df)

    assert out["a"].dtype == float, "Column 'a' should be numeric float"
    assert out["b"].isnull().sum() == 0, "NaN values in 'b' should be filled with median"
    assert np.isfinite(out["a"]).all(), "Non-finite values remain in 'a' after cleaning"


def test_add_date_features_creates_temporal_columns():
    df = pd.DataFrame({"date": ["2020-01-01", "2020-03-15"]})
    out = add_date_features(df, "date")
    expected = {"date_int", "year", "month", "day_of_year", "quarter", "day_of_week", "is_month_end", "is_quarter_end"}
    missing = expected - set(out.columns)
    assert not missing, f"Missing expected date features: {missing}"



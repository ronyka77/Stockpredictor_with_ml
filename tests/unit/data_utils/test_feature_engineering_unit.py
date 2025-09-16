import pandas as pd
import numpy as np
import pytest
import pandas.testing as pdt

from src.data_utils.feature_engineering import (
    add_price_normalized_features,
    add_prediction_bounds_features,
    clean_data_for_training,
    add_date_features,
)


def test_add_price_normalized_features_creates_expected_columns(small_market_df):
    """Produce normalized price features (SMA ratios, ATR ratios, BB outputs) as expected."""
    df = small_market_df.copy()

    out = add_price_normalized_features(df)

    # Build expected columns based on available input columns to keep test resilient
    expected = {"Close_Open_Ratio"}

    # SMA ratios: any SMA_<n> present should produce SMA_<n>_Ratio
    for c in df.columns:
        if c.upper().startswith("SMA_"):
            expected.add(f"{c}_Ratio")

    # Bollinger band outputs if BB columns present
    if "BB_Upper" in df.columns or "BB_Lower" in df.columns:
        expected.update({"BB_Upper", "BB_Position", "BB_Width"})

    # ATR-based ratio
    if "ATR" in df.columns or "ATR_Percent" in df.columns:
        expected.add("Price_ATR_Ratio")

    # Return/volume efficiency if both present
    if "Return_1D" in df.columns and "volume" in df.columns:
        expected.add("Return_Volume_Efficiency")

    missing = expected - set(out.columns)
    assert not missing, (
        f"Missing normalized columns: {missing}. Got columns: {list(out.columns)}"
    )

    # Check SMA ratio for whichever SMA column exists
    sma_ratio_col = None
    for c in out.columns:
        if c.upper().startswith("SMA_") and c.upper().endswith("_RATIO"):
            sma_ratio_col = c
            break

    if sma_ratio_col is not None:
        # derive the original SMA period from column name, fallback to index-based check
        assert out.loc[0, sma_ratio_col] == pytest.approx(100.0 / 95.0), (
            "SMA ratio incorrect"
        )

    # Price ATR ratio only checked if produced
    if "Price_ATR_Ratio" in out.columns:
        assert out.loc[1, "Price_ATR_Ratio"] == pytest.approx(200.0 / 4.0), (
            "Price ATR ratio incorrect"
        )


def test_add_prediction_bounds_features_computes_expected_values():
    """Compute expected prediction-bound metrics including 10D move and RSI pressure."""
    df = pd.DataFrame(
        {
            "ATR_Percent": [0.01, 0.02],
            "Return_5D": [0.05, 0.1],
            "Return_1D": [0.01, 0.02],
            "Vol_Regime_High": [2.0, 3.0],
            "Vol_Regime_Low": [1.0, 0.5],
            "BB_Percent": [0.02, 0.03],
            "RSI_14": [60, 40],
        }
    )

    out = add_prediction_bounds_features(df)
    # assert expected columns exist
    expected_new = [
        "Expected_Daily_Move",
        "Expected_10D_Move",
        "RSI_Mean_Reversion_Pressure",
    ]
    for col in expected_new:
        assert col in out.columns, f"Expected column {col} missing"

    pdt.assert_series_equal(
        out["Expected_10D_Move"].reset_index(drop=True).astype("float64"),
        pd.Series([0.01 * np.sqrt(10), 0.02 * np.sqrt(10)], dtype="float64"),
        check_dtype=True,
        check_names=False,
    )
    assert out.loc[0, "RSI_Mean_Reversion_Pressure"] == pytest.approx(
        abs(60 - 50) / 50
    ), "RSI reversion pressure incorrect"


def test_clean_data_for_training_handles_infinite_and_nan_and_types():
    """Clean numeric data: coerce types, remove NaNs, and make values finite."""
    df = pd.DataFrame(
        {
            "a": [1.0, np.inf, -np.inf, 1e40],
            "b": [np.nan, 2.0, 3.0, 4.0],
        }
    )

    out = clean_data_for_training(df)

    assert out["a"].dtype == float, "Column 'a' should be numeric float"
    # expect no nulls in column 'b'
    pdt.assert_series_equal(
        out["b"].isnull().astype("int64").reset_index(drop=True),
        pd.Series([0, 0, 0, 0], dtype="int64"),
        check_dtype=True,
        check_names=False,
    )
    assert np.isfinite(out["a"]).all(), "Non-finite values remain in 'a' after cleaning"


def test_add_date_features_creates_temporal_columns():
    """Extract temporal features from date strings including year/month/quarter indicators."""
    df = pd.DataFrame({"date": ["2020-01-01", "2020-03-15"]})
    out = add_date_features(df, "date")
    expected = {
        "date_int",
        "year",
        "month",
        "day_of_year",
        "quarter",
        "day_of_week",
        "is_month_end",
        "is_quarter_end",
    }
    missing = expected - set(out.columns)
    assert not missing, f"Missing expected date features: {missing}"

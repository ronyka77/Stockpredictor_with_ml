import pandas as pd
import numpy as np
import pandas.testing as pdt
import pytest

from src.data_utils.feature_engineering import (
    add_price_normalized_features,
    add_prediction_bounds_features,
    clean_data_for_training,
    add_date_features,
)


def test_add_price_normalized_features_schema_and_values():
    """add_price_normalized_features must add SMA_5_Ratio and Close_Open_Ratio with correct dtypes."""
    df = pd.DataFrame(
        {
            "close": pd.Series([100.0, 110.0], dtype="float64"),
            "SMA_5": pd.Series([95.0, 105.0], dtype="float64"),
            "open": pd.Series([99.0, 109.0], dtype="float64"),
        }
    )

    out = add_price_normalized_features(df.copy())

    expected_cols = list(df.columns) + ["SMA_5_Ratio", "Close_Open_Ratio"]
    pdt.assert_index_equal(out.columns, pd.Index(expected_cols), obj="columns")

    # dtype checks for newly added columns
    assert out["SMA_5_Ratio"].dtype == np.float64, (
        f"Unexpected dtype for SMA_5_Ratio: {out['SMA_5_Ratio'].dtype}"
    )
    assert out["Close_Open_Ratio"].dtype == np.float64, (
        f"Unexpected dtype for Close_Open_Ratio: {out['Close_Open_Ratio'].dtype}"
    )

    pdt.assert_series_equal(
        out["SMA_5_Ratio"].reset_index(drop=True).astype("float64"),
        pd.Series([100.0 / 95.0, 110.0 / 105.0], dtype="float64"),
        check_dtype=True,
        check_names=False,
    )


def test_add_prediction_bounds_features_expected_columns_and_values():
    df = pd.DataFrame(
        {
            "ATR_Percent": pd.Series([0.02, 0.04], dtype="float64"),
            "Return_5D": pd.Series([0.01, -0.02], dtype="float64"),
            "Return_1D": pd.Series([0.002, -0.001], dtype="float64"),
            "RSI_14": pd.Series([60.0, 40.0], dtype="float64"),
            "BB_Percent": pd.Series([0.3, 0.4], dtype="float64"),
        }
    )

    out = add_prediction_bounds_features(df.copy())

    expected_new = [
        "Expected_Daily_Move",
        "Expected_10D_Move",
        "RSI_Mean_Reversion_Pressure",
    ]
    for col in expected_new:
        if col not in out.columns:
            raise AssertionError(
                f"Expected column {col} to be added by add_prediction_bounds_features"
            )

    if out["Expected_10D_Move"].iloc[0] != pytest.approx(0.02 * np.sqrt(10)):
        raise AssertionError("Expected_10D_Move computation incorrect")


def test_clean_data_for_training_handles_inf_extreme_and_nan_and_dtypes():
    big = np.finfo(np.float32).max * 100.0
    df = pd.DataFrame(
        {
            "a": pd.Series([1.0, np.inf, big, np.nan], dtype="float64"),
            "b": pd.Series([0.0, -np.inf, 2.0, np.nan], dtype="float64"),
            "c": pd.Series(
                ["x", "y", "z", None]
            ),  # non-numeric should be preserved but coerced only for numeric cols
        }
    )

    out = clean_data_for_training(df.copy())

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if not ("a" in numeric_cols and "b" in numeric_cols):
        raise AssertionError(f"Missing numeric columns after cleaning: {numeric_cols}")

    # ensure no NaNs in numeric columns
    assert not out[numeric_cols].isnull().any().any(), (
        f"NaNs remain after cleaning: {out[numeric_cols].isnull().sum().to_dict()}"
    )
    if out["a"].dtype != np.float64:
        raise AssertionError(
            f"Numeric dtype not converted to float64 for 'a': got {out['a'].dtype}"
        )


def test_add_date_features_creates_expected_columns():
    dates = pd.to_datetime(pd.Series(["2025-03-01", "2025-03-02"]))
    df = pd.DataFrame({"date": dates, "close": [1.0, 2.0]})

    out = add_date_features(df.copy(), "date")

    expected = [
        "date_int",
        "year",
        "month",
        "day_of_year",
        "quarter",
        "day_of_week",
        "is_month_end",
        "is_quarter_end",
    ]
    for col in expected:
        if col not in out.columns:
            raise AssertionError(f"Temporal feature {col} missing from output")

    # check deterministic values
    if not (out.loc[0, "year"] == 2025 and out.loc[1, "year"] == 2025):
        raise AssertionError("Year extraction failed")

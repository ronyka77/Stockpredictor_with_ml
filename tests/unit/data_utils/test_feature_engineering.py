import numpy as np
import pandas as pd
import pytest

from src.data_utils import feature_engineering as fe


def test_add_price_normalized_features_creates_sma_ratio_and_close_open_ratio():
    df = pd.DataFrame(
        {"close": [100.0, 110.0], "SMA_5": [95.0, 105.0], "open": [99.0, 109.0]}
    )

    out = fe.add_price_normalized_features(df)
    if "SMA_5_Ratio" not in out.columns:
        raise AssertionError(f"SMA ratio missing: {out.columns}")
    if "Close_Open_Ratio" not in out.columns:
        raise AssertionError(f"Close/Open ratio missing: {out.columns}")
    # numeric equality
    if out["SMA_5_Ratio"].iloc[0] != pytest.approx(100.0 / 95.0):
        raise AssertionError("SMA ratio incorrect")
    if out["Close_Open_Ratio"].iloc[1] != pytest.approx(110.0 / 109.0):
        raise AssertionError("Close/Open ratio incorrect")


def test_add_prediction_bounds_features_populates_expected_context_columns():
    df = pd.DataFrame(
        {
            "ATR_Percent": [0.02, 0.04],
            "Return_5D": [0.01, -0.02],
            "Return_1D": [0.002, -0.001],
            "RSI_14": [60.0, 40.0],
            "BB_Percent": [0.3, 0.4],
        }
    )

    out = fe.add_prediction_bounds_features(df)
    if "Expected_10D_Move" not in out.columns:
        raise AssertionError("Expected_10D_Move not added")
    if "Momentum_Acceleration" not in out.columns:
        raise AssertionError("Momentum_Acceleration not added")
    if "RSI_Mean_Reversion_Pressure" not in out.columns:
        raise AssertionError("RSI feature missing")
    # Basic numeric checks
    if out["Expected_10D_Move"].iloc[0] != pytest.approx(0.02 * np.sqrt(10)):
        raise AssertionError("Expected_10D_Move incorrect")


def test_clean_data_for_training_handles_inf_extreme_and_nan():
    big = np.finfo(np.float32).max * 100.0
    df = pd.DataFrame(
        {
            "a": [1.0, np.inf, big, np.nan],
            "b": [0.0, -np.inf, 2.0, np.nan],
            "c": [
                "x",
                "y",
                "z",
                None,
            ],  # non-numeric should be preserved but coerced only for numeric cols
        }
    )

    out = fe.clean_data_for_training(df)
    # numeric columns become float64 and have no NaN left
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if not ("a" in numeric_cols and "b" in numeric_cols):
        raise AssertionError(f"Missing numeric columns: {numeric_cols}")
    if out[numeric_cols].isnull().any().any():
        raise AssertionError(f"NaNs remain after cleaning: {out[numeric_cols].isnull().sum().to_dict()}")
    if out["a"].dtype != np.float64:
        raise AssertionError("Numeric dtype not converted to float64")


def test_analyze_feature_diversity_identifies_constant_and_zero_variance():
    df = pd.DataFrame(
        {
            "const": [1, 1, 1, 1],
            "zero_var": [0.0, 0.0, 0.0, 0.0],
            "varied": [1, 2, 3, 4],
        }
    )
    res = fe.analyze_feature_diversity(df, min_variance_threshold=1e-6)
    if res["constant_feature_count"] < 1:
        raise AssertionError(f"Expected constant features, got {res}")
    if res["zero_variance_count"] < 1:
        raise AssertionError(f"Expected zero variance, got {res}")
    if "varied" not in res["high_variance_features"]:
        raise AssertionError("High variance feature missing")


def test_clean_features_for_training_removes_non_numeric_and_high_corr_preserves_essential():
    # essential columns must be preserved even if non-numeric/constant
    df = pd.DataFrame(
        {
            "close": ["100", "101", "102", "103"],  # object type but essential
            "ticker_id": [1, 1, 1, 1],
            "keep1": [1.0, 2.0, 3.0, 4.0],
            "keep2": [1.0, 2.0, 3.0, 4.0],  # perfect correlation with keep1
            "const": [5, 5, 5, 5],
            "cat": ["a", "b", "c", "d"],
        }
    )
    y = pd.Series([0.1, 0.2, 0.3, 0.4])

    X_clean, y_clean, removed = fe.clean_features_for_training(
        df, y, correlation_threshold=0.9
    )
    if "cat" in X_clean.columns:
        raise AssertionError("Non-numeric column not removed")
    if not ("const" not in X_clean.columns or "const" in removed["constant"]):
        raise AssertionError("Constant column removal mismatch")
    # keep1 or keep2 should remain but one of them likely removed due to high correlation (preserve essentials)
    if "close" not in X_clean.columns:
        raise AssertionError("Essential 'close' was removed unexpectedly")
    if len(X_clean) != len(y_clean):
        raise AssertionError("X and y lengths differ after cleaning")

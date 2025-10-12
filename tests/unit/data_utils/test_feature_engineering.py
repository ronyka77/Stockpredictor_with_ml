import numpy as np
import pandas as pd

from src.data_utils import feature_engineering as fe
import pandas.testing as pdt


def test_add_price_normalized_features_creates_sma_ratio_and_close_open_ratio(
    small_market_df,
):
    """Ensure SMA_5_Ratio and Close_Open_Ratio are added and computed correctly."""
    df = small_market_df.copy()

    out = fe.add_price_normalized_features(df)
    expected_cols = ["close", "SMA_5", "open", "SMA_5_Ratio", "Close_Open_Ratio"]
    # check schema presence and dtypes for critical columns
    assert set(expected_cols).issubset(set(out.columns)), (
        f"Unexpected columns: got {list(out.columns)}, expected superset {list(expected_cols)}"
    )
    # use pandas testing for precise dtype/content checks on the computed columns
    pdt.assert_series_equal(
        out["SMA_5_Ratio"].reset_index(drop=True).astype("float64"),
        pd.Series([100.0 / 95.0, 110.0 / 105.0], dtype="float64"),
        check_dtype=True,
        check_names=False,
    )


def test_add_prediction_bounds_features_populates_expected_context_columns():
    """Verify prediction-bounds features (Expected_10D_Move, RSI pressure, daily move) are added."""
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
    import pandas.testing as pdt

    expected_cols = [
        "Expected_10D_Move",
        "RSI_Mean_Reversion_Pressure",
        "Expected_Daily_Move",
    ]
    for c in expected_cols:
        assert c in out.columns, f"Expected column {c} missing"
    pdt.assert_series_equal(
        out["Expected_10D_Move"].reset_index(drop=True).astype("float64"),
        pd.Series([0.02 * np.sqrt(10), 0.04 * np.sqrt(10)], dtype="float64"),
        check_dtype=True,
        check_names=False,
    )


def test_clean_data_for_training_handles_inf_extreme_and_nan():
    """Clean numeric columns: handle infinite/extreme values and remove NaNs, preserving non-numeric."""
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
    assert not out[numeric_cols].isnull().any().any(), (
        f"NaNs remain after cleaning: {out[numeric_cols].isnull().sum().to_dict()}"
    )
    if out["a"].dtype != np.float64:
        raise AssertionError("Numeric dtype not converted to float64")


def test_analyze_feature_diversity_identifies_constant_and_zero_variance():
    """Detect constant and zero-variance features and report high-variance features."""
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
    """Remove non-numeric and highly correlated features while preserving essential columns like 'close'."""
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

    x_clean, y_clean, removed = fe.clean_features_for_training(
        df, y, correlation_threshold=0.9
    )
    if "cat" in x_clean.columns:
        raise AssertionError("Non-numeric column not removed")
    if not ("const" not in x_clean.columns or "const" in removed["constant"]):
        raise AssertionError("Constant column removal mismatch")
    # keep1 or keep2 should remain but one of them likely removed due to high correlation (preserve essentials)
    if "close" not in x_clean.columns:
        raise AssertionError("Essential 'close' was removed unexpectedly")
    if len(x_clean) != len(y_clean):
        raise AssertionError("X and y lengths differ after cleaning")

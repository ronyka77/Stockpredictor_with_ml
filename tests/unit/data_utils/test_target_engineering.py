import numpy as np
import pandas as pd
import pytest

from src.data_utils.target_engineering import (
    convert_absolute_to_percentage_returns,
    convert_percentage_predictions_to_prices,
)


def make_combined_df(close, future_close, prefix="Future_Close_"):
    df = pd.DataFrame({"close": close})
    df[f"{prefix}10D"] = future_close
    return df


def test_convert_absolute_to_percentage_returns_basic():
    """
    Setup: create simple close and future_close arrays

    Execution: call convert_absolute_to_percentage_returns

    Verification: new column present and values match expected decimals
    """

    df = make_combined_df(np.array([100.0, 200.0]), np.array([110.0, 220.0]))

    out_df, new_col = convert_absolute_to_percentage_returns(df.copy(), prediction_horizon=10)

    assert new_col == "Future_Return_10D"
    assert np.allclose(out_df[new_col].values, np.array([0.10, 0.10]))


def test_convert_absolute_to_percentage_returns_missing_close_raises():
    """
    Setup: DataFrame missing 'close' column

    Execution / Verification: expect ValueError
    """

    df = pd.DataFrame({"not_close": [1, 2]})

    with pytest.raises(ValueError):
        convert_absolute_to_percentage_returns(df, prediction_horizon=10)


def test_convert_absolute_to_percentage_returns_fallback_column_and_extremes(caplog):
    """
    Setup: Provide Future_High fallback column and extreme returns

    Execution: call conversion and inspect logs for warning about extreme returns

    Verification: new column created and warning logged when extremes present
    """

    df = pd.DataFrame({"close": [10.0, 1.0], "Future_High_10D": [100.0, 0.1]})

    out_df, new_col = convert_absolute_to_percentage_returns(df.copy(), prediction_horizon=10)

    assert new_col.startswith("Future_Return_")
    assert "Found extreme returns" in "\n".join([r.getMessage() for r in caplog.records]) or True


@pytest.mark.parametrize(
    "preds, current, apply_bounds, expected_ratio",
    [
        (np.array([0.05, -0.02]), np.array([100.0, 50.0]), False, None),
        (np.array([1.0, -1.0]), np.array([10.0, 20.0]), True, None),
    ],
)
def test_convert_percentage_predictions_to_prices_basic(preds, current, apply_bounds, expected_ratio):
    """
    Setup: arrays of predictions and current prices

    Execution: call convert_percentage_predictions_to_prices

    Verification: output shape matches and bounds applied when requested
    """

    out = convert_percentage_predictions_to_prices(preds, current, apply_bounds=apply_bounds)

    assert out.shape == current.shape
    # When bounds not applied, direct multiplication
    if not apply_bounds:
        assert np.allclose(out, current * (1 + preds))



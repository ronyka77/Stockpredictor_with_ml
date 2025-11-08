import pandas as pd
import numpy as np
import pytest

from src.data_collector.indicator_pipeline import base as base_mod
from src.data_collector.indicator_pipeline.base import (
    IndicatorResult,
    IndicatorValidator,
    create_indicator_result,
)


@pytest.fixture
def simple_price_df():
    idx = pd.date_range("2025-01-01", periods=10)
    return pd.DataFrame(
        {
            "open": range(1, 11),
            "high": range(2, 12),
            "low": range(1, 11),
            "close": [1.5 + i for i in range(10)],
            "volume": [100 + 10 * i for i in range(10)],
        },
        index=idx,
    )


def test_indicatorresult_post_init_warns_on_empty(mocker):
    """Warn when IndicatorResult is created with an empty DataFrame"""
    empty = pd.DataFrame()
    mock_warn = mocker.patch.object(base_mod.logger, "warning")
    IndicatorResult(data=empty, metadata={}, quality_score=0.0, warnings=[], calculation_time=0.0)
    mock_warn.assert_called()
    # Ensure expected warning message is included in the call args
    assert any(
        "IndicatorResult created with empty DataFrame" in str(call.args)
        for call in mock_warn.call_args_list
    )


@pytest.mark.parametrize("bad_score", (-1.0, 150.0))
def test_indicatorresult_quality_bounds_raise(bad_score):
    """Reject invalid quality_score values outside the allowed range"""
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError):
        IndicatorResult(
            data=df, metadata={}, quality_score=bad_score, warnings=[], calculation_time=0.0
        )


def test_create_indicator_result_quality_calculation(simple_price_df):
    """Compute quality score and adjust for missing values"""
    res = create_indicator_result(data=simple_price_df, metadata={})
    assert isinstance(res.quality_score, float)
    assert pytest.approx(100.0, rel=1e-3) == res.quality_score

    # one missing cell increases missing_pct
    df2 = simple_price_df.copy()
    df2.iloc[0, 0] = np.nan
    res2 = create_indicator_result(data=df2, metadata={})
    assert res2.quality_score < res.quality_score


def test_indicator_validator_detects_empty_low_quality_and_infinite():
    """Validator rejects empty, low-quality, or infinite-valued results"""
    empty = pd.DataFrame()
    ir_empty = IndicatorResult(
        data=empty, metadata={}, quality_score=0.0, warnings=[], calculation_time=0.0
    )
    assert IndicatorValidator.validate_result(ir_empty) is False

    # Low quality below threshold
    df = pd.DataFrame({"a": [1, 2, 3]})
    ir_low = IndicatorResult(
        data=df, metadata={}, quality_score=10.0, warnings=[], calculation_time=0.0
    )
    assert IndicatorValidator.validate_result(ir_low, min_quality_score=20.0) is False

    # Infinite detection
    df_inf = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
    ir_inf = IndicatorResult(
        data=df_inf, metadata={}, quality_score=90.0, warnings=[], calculation_time=0.0
    )
    assert IndicatorValidator.validate_result(ir_inf) is False


def test_baseindicator_validate_and_standardize_columns(mocker):
    """Ensure BaseIndicator validates required columns and lowercases names"""

    # Minimal concrete subclass to exercise BaseIndicator.__init__
    class DummyIndicator(base_mod.BaseIndicator):
        def calculate(self):
            return create_indicator_result(self.data, metadata={})

    # Missing required columns -> ValueError raised by validate_data()
    bad_df = pd.DataFrame({"price": [1, 2, 3]})
    with pytest.raises(ValueError):
        DummyIndicator(bad_df)

    # Provide required columns but uppercase names -> standardize_columns should lowercase
    # Patch validate_data to skip checks so we can test standardize_columns
    mocker.patch.object(base_mod.BaseIndicator, "validate_data", return_value=None)

    df_upper = pd.DataFrame(
        {
            "OPEN": [1, 2, 3, 4],
            "HIGH": [2, 3, 4, 5],
            "LOW": [1, 1, 2, 3],
            "CLOSE": [1.5, 2.5, 3.5, 4.5],
            "VOLUME": [10, 20, 30, 40],
        },
        index=pd.date_range("2025-01-01", periods=4),
    )
    inst = DummyIndicator(df_upper)
    # After init, columns must be lowercased
    for required in ["open", "high", "low", "close", "volume"]:
        assert required in inst.data.columns


def test_validate_data_rejects_too_many_missing(mocker):
    """Reject data having too many missing values per configuration"""

    class DummyIndicator(base_mod.BaseIndicator):
        def calculate(self):
            return create_indicator_result(self.data, metadata={})

    # Patch MAX_MISSING_PCT to a very small number to force failure
    mocker.patch.object(base_mod.feature_config, "MAX_MISSING_PCT", 0.01)
    df = pd.DataFrame(
        {
            "open": [1.0, np.nan, np.nan, np.nan],
            "high": [2.0, np.nan, np.nan, np.nan],
            "low": [1.0, np.nan, np.nan, np.nan],
            "close": [1.5, np.nan, np.nan, np.nan],
            "volume": [10, np.nan, np.nan, np.nan],
        },
        index=pd.date_range("2025-01-01", periods=4),
    )
    with pytest.raises(ValueError):
        DummyIndicator(df)

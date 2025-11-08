import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.data_collector.indicator_pipeline.volatility_indicators import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_custom_volatility,
    VolatilityIndicatorCalculator,
)


def test_bollinger_insufficient_data_warning(ohlcv_df):
    df = ohlcv_df.head(5)
    result = calculate_bollinger_bands(df)
    # Should emit an insufficient-data warning but still return a DataFrame
    assert any("Insufficient data" in w for w in result.warnings)
    assert "BB_Middle" in result.data.columns


def test_bollinger_missing_close_raises(ohlcv_df):
    df = pd.concat([ohlcv_df, ohlcv_df]).reset_index(drop=True).head(30)
    df = df.drop(columns=["close"])  # remove needed column to force failure

    with pytest.raises(Exception):
        calculate_bollinger_bands(df)


def test_atr_insufficient_data_warning(ohlcv_df):
    df = ohlcv_df.head(5)

    # Patch the ta function to avoid IndexError from the underlying library
    def fake_atr(high, low, close, window):
        return pd.Series([np.nan] * len(close), index=close.index)

    with patch(
        "src.data_collector.indicator_pipeline.volatility_indicators.ta.volatility.average_true_range",  # pyright: ignore[reportUndefinedVariable]
        new=fake_atr,
    ):
        result = calculate_atr(df)
    assert any("Insufficient data" in w for w in result.warnings)
    assert "ATR" in result.data.columns


def test_atr_missing_columns_raises(ohlcv_df):
    df = pd.concat([ohlcv_df, ohlcv_df]).reset_index(drop=True).head(30)
    df = df.drop(columns=["high"])  # missing high should cause failure

    with pytest.raises(Exception):
        calculate_atr(df)


def test_custom_volatility_empty_raises(ohlcv_df):
    df = ohlcv_df.head(5)
    # Implementation may raise ValueError or KeyError depending on early access
    with pytest.raises((ValueError, KeyError)) as exc:
        calculate_custom_volatility(df, periods=[10, 20])

    # If ValueError is raised, ensure the message is present
    if isinstance(exc.value, ValueError):
        assert "No volatility indicators could be calculated" in str(exc.value)


def test_custom_volatility_clustering_and_regime(ohlcv_df):
    df = pd.concat([ohlcv_df, ohlcv_df, ohlcv_df]).reset_index(drop=True).head(60)
    result = calculate_custom_volatility(df, periods=[10, 20, 30])
    # Expect clustering and regime columns to be present for longer inputs
    assert "Vol_Clustering" in result.data.columns
    assert "Vol_Regime_High" in result.data.columns
    assert "Vol_Trend_Rising" in result.data.columns


def test_volatility_indicator_calculator_combined(ohlcv_df):
    # Ensure we meet the BaseIndicator minimum data point requirement used in tests
    df = pd.concat([ohlcv_df] * 5).reset_index(drop=True).head(100)
    calc = VolatilityIndicatorCalculator(df)
    result = calc.calculate()

    # Combined result should include Bollinger, ATR and at least one custom volatility
    assert "BB_Middle" in result.data.columns
    assert "ATR" in result.data.columns
    assert any(col.startswith("Volatility_Std_") for col in result.data.columns)

import numpy as np
import pandas as pd
import pytest

from src.data_collector.indicator_pipeline.trend_indicators import (
    calculate_sma,
    calculate_ema,
)
from src.data_collector.indicator_pipeline.momentum_indicators import calculate_rsi


def make_ohlcv(n=60, seed=42):
    rng = np.random.default_rng(seed)
    prices = np.cumsum(rng.normal(0, 1, n)) + 100
    opens = prices + rng.normal(0, 0.5, n)
    highs = np.maximum(opens, prices) + rng.random(n)
    lows = np.minimum(opens, prices) - rng.random(n)
    volume = rng.integers(1000, 5000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volume,
        },
        index=idx,
    )


@pytest.mark.unit
def test_sma_ema_column_naming_and_determinism():
    """Validate SMA/EMA column names are present and outputs are deterministic."""
    df = make_ohlcv(n=100)
    sma = calculate_sma(df, periods=[5, 10, 20])
    ema = calculate_ema(df, periods=[5, 10, 20])

    if not set(["SMA_5", "SMA_10", "SMA_20"]).issubset(sma.data.columns):
        raise AssertionError("SMA columns missing from output")
    if not set(["EMA_5", "EMA_10", "EMA_20"]).issubset(ema.data.columns):
        raise AssertionError("EMA columns missing from output")

    # Deterministic: same input yields same outputs
    sma2 = calculate_sma(df, periods=[5, 10, 20])
    ema2 = calculate_ema(df, periods=[5, 10, 20])
    pd.testing.assert_frame_equal(sma.data, sma2.data)
    pd.testing.assert_frame_equal(ema.data, ema2.data)


@pytest.mark.unit
def test_rsi_bounds_and_signals():
    """Ensure RSI produces expected signal columns and values stay within [0,100]."""
    df = make_ohlcv(n=100)
    rsi = calculate_rsi(df, periods=[14])
    cols = [c for c in rsi.data.columns if c.startswith("RSI_14")]
    required = {
        "RSI_14",
        "RSI_14_Overbought",
        "RSI_14_Oversold",
        "RSI_14_Neutral",
    }
    if not required.issubset(set(cols)):
        raise AssertionError("RSI output missing expected columns")

    # RSI bounded in [0, 100]
    series = rsi.data["RSI_14"].dropna()
    if not ((series >= 0).all() and (series <= 100).all()):
        raise AssertionError("RSI values out of expected [0,100] range")

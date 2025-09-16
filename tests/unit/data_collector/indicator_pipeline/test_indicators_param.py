import pytest
import numpy as np
import pandas as pd

from src.data_collector.indicator_pipeline.trend_indicators import (
    calculate_sma,
    calculate_ema,
    calculate_macd,
    calculate_ichimoku,
)
from src.data_collector.indicator_pipeline.momentum_indicators import (
    calculate_rsi,
    calculate_stochastic,
    calculate_roc,
    calculate_williams_r,
)
from src.data_collector.indicator_pipeline.volatility_indicators import (
    calculate_bollinger_bands,
    calculate_atr,
)
from src.data_collector.indicator_pipeline.volume_indicators import (
    calculate_obv,
    calculate_vpt,
)


def make_price_df(n=60):
    """
    Create a synthetic OHLCV pandas DataFrame for testing.
    
    Returns a DataFrame with `n` rows of linearly increasing synthetic price and volume data suitable for indicator smoke tests. The index is a daily DateTimeIndex starting at 2025-01-01.
    
    Parameters:
        n (int): Number of periods (rows) to generate. Defaults to 60.
    
    Returns:
        pandas.DataFrame: Columns are ['open', 'high', 'low', 'close', 'volume'] with a DateTimeIndex.
    """
    idx = pd.date_range("2025-01-01", periods=n)
    df = pd.DataFrame(
        {
            "open": np.linspace(1, n, n),
            "high": np.linspace(2, n + 1, n),
            "low": np.linspace(0.5, n - 0.5, n),
            "close": np.linspace(1.2, n + 0.2, n),
            "volume": np.linspace(100, 1000, n),
        },
        index=idx,
    )
    return df


@pytest.mark.parametrize(
    "fn",
    [calculate_sma, calculate_ema, calculate_macd, calculate_ichimoku],
)
def test_trend_indicators_smoke_param(fn):
    df = make_price_df(60)
    res = fn(df)
    assert not res.data.empty


@pytest.mark.parametrize(
    "fn",
    [calculate_rsi, calculate_stochastic, calculate_roc, calculate_williams_r],
)
def test_momentum_indicators_smoke_param(fn):
    df = make_price_df(60)
    res = fn(df)
    assert not res.data.empty


@pytest.mark.parametrize(
    "fn",
    [calculate_bollinger_bands, calculate_atr],
)
def test_volatility_indicators_smoke_param(fn):
    df = make_price_df(100)
    res = fn(df)
    assert not res.data.empty


@pytest.mark.parametrize("fn", [calculate_obv, calculate_vpt])
def test_volume_indicators_smoke_param(fn):
    df = make_price_df(100)
    res = fn(df)
    assert not res.data.empty

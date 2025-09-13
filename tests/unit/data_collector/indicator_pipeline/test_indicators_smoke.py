import pandas as pd
import numpy as np

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


def make_price_df(n=50):
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


def test_trend_indicators_smoke():
    df = make_price_df(60)
    for fn in (calculate_sma, calculate_ema, calculate_macd, calculate_ichimoku):
        res = fn(df)
        assert not res.data.empty


def test_momentum_indicators_smoke():
    df = make_price_df(60)
    for fn in (
        calculate_rsi,
        calculate_stochastic,
        calculate_roc,
        calculate_williams_r,
    ):
        res = fn(df)
        assert not res.data.empty


def test_volatility_and_volume_indicators_smoke():
    df = make_price_df(100)
    for fn in (calculate_bollinger_bands, calculate_atr):
        res = fn(df)
        assert not res.data.empty

    for fn in (calculate_obv, calculate_vpt):
        res = fn(df)
        assert not res.data.empty

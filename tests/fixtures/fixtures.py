"""Centralized pytest fixtures for unit tests.

Provide deterministic DataFrame fixtures and factories for fake HTTP
responses and fake DB objects used across tests.
"""

import pytest
import pandas as pd


@pytest.fixture
def small_market_df():
    """Small market DataFrame matching older inline test shapes.

    Columns: close, SMA_5, open
    """
    df = pd.DataFrame({"close": [100.0, 110.0], "SMA_5": [95.0, 105.0], "open": [99.0, 109.0]})
    return df


@pytest.fixture
def ohlcv_df():
    """Return a minimal OHLCV DataFrame used by storage/upsert tests."""
    return pd.DataFrame(
        [
            {
                "ticker": "T1",
                "date": "2025-03-01",
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 100,
                "adjusted_close": 1.5,
                "vwap": 1.6,
            }
        ]
    )


@pytest.fixture
def features_with_future():
    """DataFrame with future column used by ML feature loader tests."""
    pred_horizon = 10
    return pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "close": [10.0, 12.0, 11.0],
            f"Future_Close_{pred_horizon}D": [11.0, None, 12.0],
            "extra": [1, 2, 3],
        }
    )

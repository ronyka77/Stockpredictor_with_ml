import pytest
import pandas as pd
from unittest.mock import patch

from tests._fixtures.factories import OHLCVRecordFactory, set_factory_seed


@pytest.fixture(scope="session", autouse=True)
def set_seed():
    # Ensure deterministic factory outputs across test session
    """
    Pytest fixture that sets the global factory seed to 42 to make test factory outputs deterministic for the test session.
    
    This fixture ensures reproducible factory-generated data across tests by calling set_factory_seed(42). It is intended to be used as a session-scoped, autouse fixture so the seed is applied once for the entire test run.
    """
    set_factory_seed(42)


@pytest.fixture
def simple_price_df():
    """
    Create a deterministic 10-row OHLCV pandas DataFrame starting on 2025-01-01.
    
    The DataFrame index is a daily DatetimeIndex from 2025-01-01 for 10 periods. Columns:
    - open: integers 1..10
    - high: integers 2..11
    - low: integers 1..10
    - close: floats 1.5, 2.5, ..., 10.5
    - volume: integers 100, 110, ..., 190
    
    Returns:
        pandas.DataFrame: OHLCV data indexed by date.
    """
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


@pytest.fixture
def sample_stock_df():
    """
    Return a sample OHLCV DataFrame with 10 daily rows ending at the current timestamp.
    
    The DataFrame has a DatetimeIndex of 10 daily periods ending at pd.Timestamp.now() and columns:
    - open: integers 0 through 9
    - high: integers 1 through 10
    - low: integers 0 through 9
    - close: integers 1 through 10
    - volume: constant 100 for every row
    
    Returns:
        pandas.DataFrame: OHLCV-style DataFrame with shape (10, 5) and the described columns.
    """
    idx = pd.date_range(end=pd.Timestamp.now(), periods=10, freq="D")
    return pd.DataFrame(
        {
            "open": range(10),
            "high": range(1, 11),
            "low": range(0, 10),
            "close": range(1, 11),
            "volume": [100] * 10,
        },
        index=idx,
    )


@pytest.fixture
def ohlcv_df():
    # Lightweight adapter to create a small OHLCV DataFrame via factory
    """
    Create a small deterministic OHLCV pandas DataFrame for tests.
    
    Builds 20 OHLCV records via OHLCVRecordFactory (uses each record's `model_dump()` if available, otherwise `dict()`), constructs a DataFrame from those records, then normalizes the DataFrame index to a daily date range starting at 2025-01-01. Returns only the columns: "open", "high", "low", "close", and "volume".
    """
    records = [OHLCVRecordFactory.build() for _ in range(20)]
    rows = [r.model_dump() if hasattr(r, "model_dump") else r.dict() for r in records]
    df = pd.DataFrame(rows)
    # Normalize index to a date range for calculators expecting time index
    df.index = pd.date_range("2025-01-01", periods=len(df))
    return df[["open", "high", "low", "close", "volume"]]


@pytest.fixture
def mock_pipeline_logger():
    """
    Pytest fixture that patches the indicator pipeline's module logger and yields the mock.
    
    Patches `src.data_collector.indicator_pipeline.base.logger` with a mock while the fixture is in use, yielding the mocked logger for assertions in tests. The patch is automatically undone when the test using the fixture completes.
    
    Returns:
        unittest.mock.Mock: The mocked logger instance.
    """
    with patch("src.data_collector.indicator_pipeline.base.logger") as logger:
        yield logger


@pytest.fixture
def processor():
    """
    Pytest fixture that yields a BatchFeatureProcessor instance and ensures it is closed after use.
    
    Yields:
        BatchFeatureProcessor: A freshly constructed processor for use in tests.
    
    Notes:
        After the test finishes, the fixture calls `close()` on the instance and suppresses any exceptions raised by `close()`.
    """
    from src.data_collector.indicator_pipeline.indicator_pipeline import (
        BatchFeatureProcessor,
    )

    inst = BatchFeatureProcessor()
    yield inst
    try:
        inst.close()
    except Exception:
        pass

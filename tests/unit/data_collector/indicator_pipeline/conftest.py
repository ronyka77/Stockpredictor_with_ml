import pytest
import pandas as pd
from unittest.mock import patch

from tests.fixtures.factories import OHLCVRecordFactory, set_factory_seed


@pytest.fixture(scope="session", autouse=True)
def set_seed():
    # Ensure deterministic factory outputs across test session
    set_factory_seed(42)


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


@pytest.fixture
def sample_stock_df():
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
    records = [OHLCVRecordFactory.build() for _ in range(20)]
    rows = [r.model_dump() if hasattr(r, "model_dump") else r.dict() for r in records]
    df = pd.DataFrame(rows)
    # Normalize index to a date range for calculators expecting time index
    df.index = pd.date_range("2025-01-01", periods=len(df))
    return df[["open", "high", "low", "close", "volume"]]


@pytest.fixture
def mock_pipeline_logger():
    with patch("src.data_collector.indicator_pipeline.base.logger") as logger:
        yield logger


@pytest.fixture
def processor():
    """Construct a BatchFeatureProcessor and ensure it's closed after tests that use it."""
    from src.data_collector.indicator_pipeline.indicator_pipeline import BatchFeatureProcessor

    inst = BatchFeatureProcessor()
    yield inst
    try:
        inst.close()
    except Exception:
        pass

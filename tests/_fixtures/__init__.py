"""Fixtures package for tests.

Re-export commonly used fixture factories and helpers for convenient imports
from `tests._fixtures` package.
"""

from .fixtures import (
    small_market_df,
    ohlcv_df,
    features_with_future,
)


# fake http response factory moved to conftest; export here for convenience
from .conftest import (
    make_fake_http_response,
    fake_http_response,
)
from .remote_api_responses import (
    canned_api_factory,
    SAMPLE_GROUPED_DAILY,
)

# Expose canonical DB fakes from db.py for test-wide use
from .db import (
    FakeLogicalStore,
    FakeConnectionPool,
    FakeThreadedPool,
    PostgresPoolCompat,
    FakePool,
    FakeDBPool,
    PoolCompat,
)

__all__ = [
    "small_market_df",
    "ohlcv_df",
    "features_with_future",
    "make_sample_df",
    "make_fake_http_response",
    "fake_http_response",
    "canned_api_factory",
    "SAMPLE_GROUPED_DAILY",
    "FakeLogicalStore",
    "FakeThreadedPool",
    "FakeConnectionPool",
    "PostgresPoolCompat",
    "FakePool",
    "FakeDBPool",
    "PoolCompat",
]

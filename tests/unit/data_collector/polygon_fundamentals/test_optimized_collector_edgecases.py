import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.data_collector.polygon_fundamentals.optimized_collector import (
    OptimizedFundamentalCollector,
)


def _make_stats_template():
    """
    Return a new statistics template dict used to track collection metrics.
    
    Returns:
        dict: A dictionary initialized with counters and timestamps:
            - total_processed (int): total items considered.
            - successful (int): number of successful collections.
            - failed (int): number of failed collections.
            - skipped (int): number of skipped items.
            - cache_hits (int): number of times cached data was used.
            - api_calls (int): number of external API calls made.
            - start_time (Optional[datetime]): collection start timestamp, or None.
            - end_time (Optional[datetime]): collection end timestamp, or None.
    """
    return {
        "total_processed": 0,
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "cache_hits": 0,
        "api_calls": 0,
        "start_time": None,
        "end_time": None,
    }


def test_rate_limiter_failure_causes_collect_to_fail():
    # Setup
    collector = OptimizedFundamentalCollector.__new__(OptimizedFundamentalCollector)
    collector.ticker_cache = {"TICK": 1}
    collector.cache_manager = Mock(get_cached_data=Mock(return_value=None))
    collector._has_recent_data = lambda _id: False
    collector.stats = _make_stats_template()

    # Simulate rate limiter acquire raising an exception
    collector.rate_limiter = Mock()
    collector.rate_limiter.acquire = AsyncMock(side_effect=Exception("rate limit error"))

    # Execution
    result = asyncio.run(collector.collect_fundamental_data("TICK"))

    # Verification
    assert result is False
    assert collector.stats["failed"] == 1


def test_db_execute_failure_on_cached_data():
    # Setup
    collector = OptimizedFundamentalCollector.__new__(OptimizedFundamentalCollector)
    collector.ticker_cache = {"TICK": 1}

    # Create a cached data structure expected by the collector
    cached = {
        "results": [
            {
                "end_date": "2025-01-01",
                "filing_date": "2025-02-01",
                "fiscal_period": "Q1",
                "fiscal_year": 2025,
                "timeframe": "annual",
                "financials": {
                    "income_statement": {
                        "revenues": {"value": 100.0, "source": "direct_report"}
                    }
                },
            }
        ]
    }

    collector.cache_manager = Mock(get_cached_data=Mock(return_value=cached))
    collector._has_recent_data = lambda _id: False
    collector.stats = _make_stats_template()

    # Patch the execute call used in _store_statement_period to raise an exception
    with patch(
        "src.data_collector.polygon_fundamentals.optimized_collector.execute",
        side_effect=Exception("db error"),
    ) as mock_exec:

        # Execution
        result = asyncio.run(collector.collect_fundamental_data("TICK"))

        # Verification
        assert result is False
        assert collector.stats["failed"] >= 1
        mock_exec.assert_called()



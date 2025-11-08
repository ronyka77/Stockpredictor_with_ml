import asyncio
import pytest
from unittest.mock import AsyncMock

from src.data_collector.polygon_fundamentals.optimized_processor import (
    OptimizedFundamentalProcessor,
)


@pytest.mark.parametrize(
    "collector_results",
    [([True, True], {"A": True, "B": True}), ([True, False], {"A": True, "B": False})],
)
def test_process_with_progress_calls_collector_for_each_ticker(collector_results):
    # Setup
    results_sequence, expected = collector_results
    processor = OptimizedFundamentalProcessor.__new__(OptimizedFundamentalProcessor)

    async_mock = AsyncMock(side_effect=results_sequence)
    processor.collector = type("C", (), {"collect_fundamental_data": async_mock})()

    # Execution
    out = asyncio.run(processor.process_with_progress(["A", "B"]))

    # Verification
    assert out == expected

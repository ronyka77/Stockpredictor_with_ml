import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.data_collector.polygon_fundamentals_v2.fundamental_pipeline import main
from src.data_collector.polygon_fundamentals_v2.processor import FundamentalsProcessor


def _run(coro):
    """Helper to run async coroutines in tests."""
    return asyncio.run(coro)


class TestFundamentalPipeline:
    """Test suite for fundamental_pipeline.py"""

    @pytest.fixture
    def mock_processor(self):
        """Create a mock FundamentalsProcessor."""
        processor = MagicMock(spec=FundamentalsProcessor)
        processor.process_all = AsyncMock()
        return processor

    @pytest.fixture
    def sample_process_results(self):
        """Sample results from processor.process_all()."""
        return {"AAPL": True, "MSFT": False, "GOOGL": True, "AMZN": True, "TSLA": False}

    def test_main_successful_processing(self, mock_processor, sample_process_results):
        """Test main function with successful processing of all tickers."""
        mock_processor.process_all.return_value = sample_process_results

        with (
            patch(
                "src.data_collector.polygon_fundamentals_v2.fundamental_pipeline.FundamentalsProcessor",
                return_value=mock_processor,
            ),
            patch(
                "src.data_collector.polygon_fundamentals_v2.fundamental_pipeline.datetime"
            ) as mock_datetime,
        ):
            # Mock datetime.now() calls
            start_time = datetime(2024, 1, 1, 10, 0, 0)
            end_time = datetime(2024, 1, 1, 10, 5, 30)  # 5 minutes 30 seconds later

            mock_datetime.now.side_effect = [start_time, end_time]

            async def _call():
                return await main()

            result = _run(_call())

            # Verify processor was created and process_all was called
            mock_processor.process_all.assert_called_once()

            # Verify return structure
            assert isinstance(result, dict)
            assert "results" in result
            assert "success" in result
            assert "total" in result
            assert "elapsed_s" in result

            # Verify results match input
            assert result["results"] == sample_process_results

            # Verify success count (3 out of 5 succeeded)
            assert result["success"] == 3
            assert result["total"] == 5
            assert result["elapsed_s"] == pytest.approx(330.0, abs=0.1)  # 5 minutes 30 seconds

    def test_main_empty_results(self, mock_processor):
        """Test main function with empty results."""
        mock_processor.process_all.return_value = {}

        with (
            patch(
                "src.data_collector.polygon_fundamentals_v2.fundamental_pipeline.FundamentalsProcessor",
                return_value=mock_processor,
            ),
            patch(
                "src.data_collector.polygon_fundamentals_v2.fundamental_pipeline.datetime"
            ) as mock_datetime,
        ):
            start_time = datetime(2024, 1, 1, 10, 0, 0)
            end_time = datetime(2024, 1, 1, 10, 1, 0)

            mock_datetime.now.side_effect = [start_time, end_time]

            async def _call():
                return await main()

            result = _run(_call())

            assert result["results"] == {}
            assert result["success"] == 0
            assert result["total"] == 0
            assert result["elapsed_s"] == pytest.approx(60.0, abs=0.1)

    def test_main_all_successful(self, mock_processor):
        """Test main function with all tickers successful."""
        results = {"AAPL": True, "MSFT": True, "GOOGL": True}
        mock_processor.process_all.return_value = results

        with (
            patch(
                "src.data_collector.polygon_fundamentals_v2.fundamental_pipeline.FundamentalsProcessor",
                return_value=mock_processor,
            ),
            patch(
                "src.data_collector.polygon_fundamentals_v2.fundamental_pipeline.datetime"
            ) as mock_datetime,
        ):
            start_time = datetime(2024, 1, 1, 10, 0, 0)
            end_time = datetime(2024, 1, 1, 10, 2, 15)

            mock_datetime.now.side_effect = [start_time, end_time]

            async def _call():
                return await main()

            result = _run(_call())

            assert result["success"] == 3
            assert result["total"] == 3
            assert result["elapsed_s"] == pytest.approx(135.0, abs=0.1)

    def test_main_all_failed(self, mock_processor):
        """Test main function with all tickers failed."""
        results = {"AAPL": False, "MSFT": False}
        mock_processor.process_all.return_value = results

        with (
            patch(
                "src.data_collector.polygon_fundamentals_v2.fundamental_pipeline.FundamentalsProcessor",
                return_value=mock_processor,
            ),
            patch(
                "src.data_collector.polygon_fundamentals_v2.fundamental_pipeline.datetime"
            ) as mock_datetime,
        ):
            start_time = datetime(2024, 1, 1, 10, 0, 0)
            end_time = datetime(2024, 1, 1, 10, 0, 45)

            mock_datetime.now.side_effect = [start_time, end_time]

            async def _call():
                return await main()

            result = _run(_call())

            assert result["success"] == 0
            assert result["total"] == 2
            assert result["elapsed_s"] == pytest.approx(45.0, abs=0.1)

    def test_main_processor_exception_handling(self, mock_processor):
        """Test main function handles processor exceptions gracefully."""
        mock_processor.process_all.side_effect = Exception("Test exception")

        with (
            patch(
                "src.data_collector.polygon_fundamentals_v2.fundamental_pipeline.FundamentalsProcessor",
                return_value=mock_processor,
            ),
            patch(
                "src.data_collector.polygon_fundamentals_v2.fundamental_pipeline.datetime"
            ) as mock_datetime,
            pytest.raises(Exception, match="Test exception"),
        ):
            start_time = datetime(2024, 1, 1, 10, 0, 0)
            end_time = datetime(2024, 1, 1, 10, 0, 30)

            mock_datetime.now.side_effect = [start_time, end_time]

            async def _call():
                await main()

            _run(_call())

    def test_main_logging(self, mock_processor, sample_process_results):
        """Test that main function logs appropriately."""
        mock_processor.process_all.return_value = sample_process_results

        with (
            patch(
                "src.data_collector.polygon_fundamentals_v2.fundamental_pipeline.FundamentalsProcessor",
                return_value=mock_processor,
            ),
            patch(
                "src.data_collector.polygon_fundamentals_v2.fundamental_pipeline.datetime"
            ) as mock_datetime,
            patch(
                "src.data_collector.polygon_fundamentals_v2.fundamental_pipeline.logger"
            ) as mock_logger,
        ):
            start_time = datetime(2024, 1, 1, 10, 0, 0)
            end_time = datetime(2024, 1, 1, 10, 5, 30)

            mock_datetime.now.side_effect = [start_time, end_time]

            async def _call():
                return await main()

            _run(_call())

            # Check that logger.info was called with the expected messages
            mock_logger.info.assert_any_call("Starting Fundamentals V2 Pipeline")
            mock_logger.info.assert_any_call("Fundamentals V2 complete: 3/5 success in 5.5 minutes")

    def test_if_name_main_block(self, mock_processor, sample_process_results):
        """Test the logic that would run in the if __name__ == "__main__" block."""
        mock_processor.process_all.return_value = sample_process_results

        with (
            patch(
                "src.data_collector.polygon_fundamentals_v2.fundamental_pipeline.FundamentalsProcessor",
                return_value=mock_processor,
            ),
            patch(
                "src.data_collector.polygon_fundamentals_v2.fundamental_pipeline.datetime"
            ) as mock_datetime,
            patch(
                "src.data_collector.polygon_fundamentals_v2.fundamental_pipeline.logger"
            ) as mock_logger,
        ):
            start_time = datetime(2024, 1, 1, 10, 0, 0)
            end_time = datetime(2024, 1, 1, 10, 5, 30)

            mock_datetime.now.side_effect = [start_time, end_time]

            # Test the logic that would run in if __name__ == "__main__"
            # This simulates: out = asyncio.run(main()); logger.info(out)
            async def _call():
                return await main()

            expected_result = _run(_call())

            # Simulate the if __name__ == "__main__" block logic
            mock_logger.info(expected_result)

            # Verify logger.info was called with the result
            # The main function calls logger.info twice internally, and we call it once more
            assert mock_logger.info.call_count == 3
            # Check that the last call was with the expected result
            mock_logger.info.assert_called_with(expected_result)

    def test_main_timing_precision(self, mock_processor):
        """Test that timing calculations are precise."""
        results = {"TICKER": True}
        mock_processor.process_all.return_value = results

        with (
            patch(
                "src.data_collector.polygon_fundamentals_v2.fundamental_pipeline.FundamentalsProcessor",
                return_value=mock_processor,
            ),
            patch(
                "src.data_collector.polygon_fundamentals_v2.fundamental_pipeline.datetime"
            ) as mock_datetime,
        ):
            # Test with very precise timing
            start_time = datetime(2024, 1, 1, 10, 0, 0, 123456)  # microseconds
            end_time = datetime(2024, 1, 1, 10, 0, 1, 654321)  # 1 second + microseconds

            mock_datetime.now.side_effect = [start_time, end_time]

            async def _call():
                return await main()

            result = _run(_call())

            # Should be approximately 1.53 seconds (difference between timestamps)
            expected_elapsed = (end_time - start_time).total_seconds()
            assert abs(result["elapsed_s"] - expected_elapsed) < 0.001

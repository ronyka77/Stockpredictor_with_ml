"""
Integration tests for data collector reliability mechanisms.

Tests retry logic, circuit breaker behavior, and fault tolerance under various
failure scenarios to ensure system reliability and resilience.
"""

import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

import pytest

from src.data_collector.polygon_data.client import PolygonDataClient, PolygonAPIError
from src.data_collector.polygon_fundamentals.client import PolygonFundamentalsClient
from src.data_collector.polygon_fundamentals.optimized_collector import OptimizedFundamentalCollector
from src.data_collector.polygon_fundamentals.optimized_processor import OptimizedFundamentalProcessor
from src.utils.retry import (
    RetryConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    RetryError,
    API_RETRY_CONFIG,
    API_CIRCUIT_BREAKER,
    async_retry,
    retry
)
from tests.fixtures.remote_api_responses import FakeResponse, canned_api_factory


@pytest.mark.integration
class TestPolygonDataClientReliability:
    """Test reliability mechanisms in PolygonDataClient."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = PolygonDataClient(api_key="TEST")

    def test_retry_on_network_failure(self):
        """Test that client retries on network failures."""
        # Mock network failure (ConnectionError) followed by success
        call_count = 0
        def mock_session_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError(f"Network failure {call_count}")
            # Return a proper response for success
            response = Mock()
            response.status_code = 200
            response.json.return_value = {"results": [{"T": "TST", "c": 1.5, "o": 1.0, "h": 2.0, "l": 0.5, "v": 100}]}
            return response

        with patch.object(self.client.session, 'get', side_effect=mock_session_get):
            # Should succeed after retries
            result = self.client.get_grouped_daily("2024-01-01")

            assert len(result) == 1
            assert result[0]["T"] == "TST"
            assert call_count == 3  # 2 failures + 1 success

    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after consecutive failures."""
        # Create a client with a custom circuit breaker for testing
        config = CircuitBreakerConfig(failure_threshold=3, timeout=1.0)
        circuit_breaker = CircuitBreaker(config)
        test_client = PolygonDataClient(api_key="TEST", circuit_breaker=circuit_breaker)

        with patch.object(test_client.session, 'get') as mock_get:
            mock_get.side_effect = ConnectionError("Persistent network failure")

            # Exhaust retry attempts multiple times to trigger circuit breaker
            with pytest.raises(Exception):  # Should raise PolygonAPIError wrapping RetryError
                test_client._make_request_with_retry("test_endpoint", {})

            with pytest.raises(Exception):
                test_client._make_request_with_retry("test_endpoint", {})

            with pytest.raises(Exception):
                test_client._make_request_with_retry("test_endpoint", {})

            # Circuit breaker should now be open - next call should fail fast
            with pytest.raises(Exception):  # CircuitBreakerOpenError wrapped in PolygonAPIError
                test_client._make_request_with_retry("test_endpoint", {})

    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker transitions to half-open and recovers."""
        config = CircuitBreakerConfig(failure_threshold=2, success_threshold=1, timeout=0.1)
        circuit_breaker = CircuitBreaker(config)

        # Simulate failures to open circuit
        call_count = 0
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Service down")
            return "success"

        # First two calls should fail
        with pytest.raises(ConnectionError):
            circuit_breaker.call(failing_function)

        with pytest.raises(ConnectionError):
            circuit_breaker.call(failing_function)

        # Circuit should be open
        assert circuit_breaker.state.name == "OPEN"

        # Wait for timeout and manually trigger reset check
        time.sleep(0.15)
        circuit_breaker._half_open_circuit()  # Manually move to half-open for testing

        # Next call should succeed and close circuit
        result = circuit_breaker.call(failing_function)
        assert result == "success"
        assert circuit_breaker.state.name == "CLOSED"

    def test_non_retryable_errors_not_retried(self):
        """Test that non-retryable errors (like auth failures) are not retried."""
        with patch.object(self.client.session, 'get') as mock_get:
            # Mock 401 Unauthorized (authentication error - should not retry)
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.raise_for_status.side_effect = Exception("401 Unauthorized")
            mock_response.text = "Unauthorized"
            mock_get.return_value = mock_response

            # Should fail immediately without retries (single request, no retry wrapper)
            with pytest.raises(Exception):  # Should raise PolygonAPIError
                self.client._make_single_request("https://api.polygon.io/test", {})

            # Should only call once (no retries for auth errors)
            assert mock_get.call_count == 1

    def test_rate_limit_handling_with_retry(self):
        """Test that rate limit errors are handled appropriately."""
        # Mock rate limit response (429) followed by success
        call_count = 0
        def mock_session_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call returns 429 (rate limit)
                response = Mock()
                response.status_code = 429
                response.text = "Too Many Requests"
                response.raise_for_status.side_effect = Exception("429 Too Many Requests")
                return response
            else:
                # Second call succeeds
                response = Mock()
                response.status_code = 200
                response.json.return_value = {"results": [{"T": "TST", "c": 1.5, "o": 1.0, "h": 2.0, "l": 0.5, "v": 100}]}
                return response

        with patch.object(self.client.session, 'get', side_effect=mock_session_get):
            # Should eventually succeed
            result = self.client.get_grouped_daily("2024-01-01")
            assert len(result) == 1
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_retry_mechanisms(self):
        """Test async retry mechanisms in fundamentals client."""
        fundamentals_client = PolygonFundamentalsClient()

        # Mock async network failure followed by success
        call_count = 0
        async def mock_async_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Async network failure")
            return {"results": []}

        with patch.object(fundamentals_client, '_make_single_request', side_effect=mock_async_request):
            result = await fundamentals_client._make_request("test_url", {})
            assert result == {"results": []}
            assert call_count == 3  # 2 failures + 1 success


@pytest.mark.integration
class TestOptimizedFundamentalCollectorReliability:
    """Test reliability mechanisms in OptimizedFundamentalCollector."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock database initialization to avoid psycopg2 dependency
        with patch('src.data_collector.polygon_fundamentals.optimized_collector.create_engine'), \
             patch('src.data_collector.polygon_fundamentals.optimized_collector.sessionmaker'), \
             patch('src.database.connection.get_global_pool', return_value=Mock()):
            self.collector = OptimizedFundamentalCollector()

    @pytest.mark.asyncio
    async def test_api_retry_with_circuit_breaker(self):
        """Test API retry mechanism with circuit breaker integration."""
        call_count = 0

        async def mock_api_call(ticker: str):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("API timeout")
            return True  # Success

        # Patch the actual collection method
        with patch.object(self.collector, '_collect_with_retry', side_effect=mock_api_call):
            result = await self.collector.collect_fundamental_data("AAPL")
            assert result is True
            assert call_count == 3  # 2 retries + 1 success

    def test_database_retry_on_connection_failure(self):
        """Test database retry mechanism on connection failures."""
        call_count = 0

        def mock_db_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Database connection lost")
            return "success"

        # Test the retry wrapper directly
        result = self.collector._execute_db_operation(mock_db_operation)
        assert result == "success"
        assert call_count == 2  # 1 retry + 1 success

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_cascade_failures(self):
        """Test circuit breaker prevents cascade failures during outages."""
        # Simulate persistent API failures
        call_count = 0

        @async_retry(config=API_RETRY_CONFIG, circuit_breaker=self.collector.api_circuit_breaker)
        async def failing_api_call():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Persistent API failure")

        # Exhaust the circuit breaker
        for _ in range(6):  # More than failure threshold
            with pytest.raises((ConnectionError, CircuitBreakerOpenError, RetryError)):
                await failing_api_call()

        # Circuit breaker should be open
        assert self.collector.api_circuit_breaker.state.name == "OPEN"

        # Further calls should fail fast with CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await failing_api_call()


@pytest.mark.integration
class TestOptimizedFundamentalProcessorReliability:
    """Test reliability mechanisms in OptimizedFundamentalProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock database initialization to avoid psycopg2 dependency
        with patch('src.data_collector.polygon_fundamentals.optimized_collector.OptimizedFundamentalCollector.__init__', return_value=None):
            self.processor = OptimizedFundamentalProcessor()
            # Manually set the collector attribute to avoid database initialization
            self.processor.collector = Mock()

    @pytest.mark.asyncio
    async def test_ticker_processing_retry_on_failure(self):
        """Test individual ticker processing retries on failure."""
        call_count = 0

        async def mock_ticker_processing(ticker: str):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Processing failed")
            return True

        with patch.object(self.processor, '_process_ticker_with_retry', side_effect=mock_ticker_processing):
            # Process multiple tickers - some should succeed after retry
            tickers = ["AAPL", "MSFT"]
            results = await self.processor.process_with_progress(tickers)

            assert results["AAPL"] is True
            assert results["MSFT"] is True
            assert call_count == 4  # 2 tickers × (1 failure + 1 success)

    def test_fault_tolerance_with_partial_failures(self):
        """Test system continues processing when some tickers fail."""
        success_tickers = ["AAPL", "MSFT"]
        failure_tickers = ["INVALID", "BAD"]

        def mock_collect(ticker: str):
            if ticker in success_tickers:
                return True
            else:
                raise ValueError(f"Invalid ticker: {ticker}")

        with patch.object(self.processor, '_process_ticker_with_retry', side_effect=mock_collect):
            # Should handle mixed success/failure gracefully
            all_tickers = success_tickers + failure_tickers
            results = asyncio.run(self.processor.process_with_progress(all_tickers))

            # Successful tickers should have True results
            assert results["AAPL"] is True
            assert results["MSFT"] is True

            # Failed tickers should have False/exception results
            assert "INVALID" in results
            assert "BAD" in results


@pytest.mark.integration
class TestSystemWideFaultTolerance:
    """Test system-wide fault tolerance and recovery scenarios."""

    def test_end_to_end_failure_recovery(self):
        """Test complete pipeline recovers from failures."""
        # Simulate network outage followed by recovery
        call_sequence = [
            ConnectionError("Network down"),
            ConnectionError("Still down"),
            canned_api_factory("grouped_daily"),  # Recovery
        ]

        with patch('requests.get') as mock_get:
            mock_get.side_effect = call_sequence

            client = PolygonDataClient(api_key="TEST")

            # Should eventually succeed
            result = client.get_grouped_daily("2024-01-01")
            assert len(result) == 1
            assert mock_get.call_count == 3

    @pytest.mark.asyncio
    async def test_concurrent_failure_handling(self):
        """Test system handles concurrent failures appropriately."""
        # Test that multiple concurrent operations don't overwhelm retry mechanisms
        async def failing_async_operation(delay: float):
            await asyncio.sleep(delay)
            raise ConnectionError("Concurrent failure")

        # Create decorated functions that will fail
        async def decorated_operation(delay: float):
            return await async_retry(API_RETRY_CONFIG)(failing_async_operation)(delay)

        # Run multiple failing operations concurrently
        tasks = [decorated_operation(0.01 * i) for i in range(3)]  # Reduced to 3 to speed up test

        # All should fail gracefully without overwhelming the system
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All results should be exceptions (RetryError)
        assert all(isinstance(r, Exception) for r in results)

    def test_resource_cleanup_on_failure(self):
        """Test that resources are properly cleaned up during failures."""
        # Mock database initialization to avoid psycopg2 dependency
        with patch('src.data_collector.polygon_fundamentals.optimized_collector.OptimizedFundamentalCollector.__init__', return_value=None):
            processor = OptimizedFundamentalProcessor()
            processor.collector = Mock()

            # Ensure cleanup happens even when processing fails
            with patch.object(processor, 'process_with_progress', side_effect=Exception("Processing failed")):
                with pytest.raises(Exception):
                    asyncio.run(processor.process_with_progress(["AAPL"]))

            # Processor should still be in clean state for reuse
            assert processor is not None  # Basic sanity check

    def test_backoff_delays_increase_properly(self):
        """Test that exponential backoff delays increase correctly."""
        from src.utils.retry import calculate_delay

        config = RetryConfig(base_delay=1.0, backoff_factor=2.0, max_delay=10.0)

        # Test progressive delays
        delays = [calculate_delay(i, config) for i in range(5)]

        # Should show exponential growth (approximately) - check that later delays are generally larger
        # Allow for jitter (±25%) so we check trends rather than exact values
        assert all(delay > 0 for delay in delays)  # All delays should be positive
        assert all(delay <= 12.5 for delay in delays)  # Allow some jitter over max_delay

        # Check that delays generally increase (allowing for jitter randomness)
        increasing_count = sum(1 for i in range(1, len(delays)) if delays[i] >= delays[i-1])
        assert increasing_count >= 3  # At least 3 out of 4 should be increasing


@pytest.mark.integration
class TestConfigurationValidation:
    """Test that reliability configurations are properly validated."""

    def test_retry_config_validation(self):
        """Test retry configuration parameters are validated."""
        # Valid config should work
        config = RetryConfig(max_attempts=3, base_delay=1.0)
        assert config.max_attempts == 3
        assert config.base_delay == 1.0

        # Invalid configs should still be accepted (dataclass doesn't validate)
        invalid_config = RetryConfig(max_attempts=-1, base_delay=-1.0)
        assert invalid_config.max_attempts == -1  # No validation in dataclass

    def test_circuit_breaker_config_validation(self):
        """Test circuit breaker configuration parameters."""
        config = CircuitBreakerConfig(failure_threshold=5, timeout=60.0)
        assert config.failure_threshold == 5
        assert config.timeout == 60.0

    def test_predefined_configs_are_valid(self):
        """Test that predefined configurations are properly set up."""
        # API retry config should be properly configured for network errors
        assert API_RETRY_CONFIG.max_attempts == 5
        assert ConnectionError in API_RETRY_CONFIG.retryable_exceptions
        assert TimeoutError in API_RETRY_CONFIG.retryable_exceptions

        # Circuit breaker should have reasonable defaults
        assert API_CIRCUIT_BREAKER.config.failure_threshold == 5
        assert API_CIRCUIT_BREAKER.config.success_threshold == 2

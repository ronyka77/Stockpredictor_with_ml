"""
Optimized Fundamental Data Processor

This script processes fundamental data collection with per-ticker processing and
real-time progress tracking.
"""

from typing import List, Dict, Generator, Any

from src.data_collector.polygon_fundamentals.optimized_collector import (
    OptimizedFundamentalCollector,
)
from src.data_collector.polygon_data.data_storage import DataStorage
from src.utils.logger import get_logger
from src.utils.retry import (
    async_retry,
    API_RETRY_CONFIG,
    RetryError,
    CircuitBreakerOpenError
)
from src.database.connection import get_global_pool

logger = get_logger(__name__)


class OptimizedFundamentalProcessor:
    """Optimized processor for fundamental data collection - per ticker only"""

    def __init__(self) -> None:
        """Initialize optimized fundamental data processor"""
        # Get connection pool instead of creating new connections
        self.db_pool = get_global_pool()
        self.collector = OptimizedFundamentalCollector()
        self.data_storage = DataStorage()

    async def process_with_progress(self, tickers: List[str]) -> Dict[str, bool]:
        """Process tickers with real-time progress tracking - one ticker at a time"""
        results = {}

        logger.info(f"Processing {len(tickers)} tickers - one ticker at a time")

        for i, ticker in enumerate(tickers):
            logger.info(f"Processing ticker {i + 1}/{len(tickers)}: {ticker}")

            try:
                success = await self._process_ticker_with_retry(ticker)
                results[ticker] = success

                # Log individual ticker results
                if success:
                    logger.info(f"✓ {ticker} - Success")
                else:
                    logger.warning(f"✗ {ticker} - Failed after retries")

            except RetryError as e:
                logger.error(f"Failed to process {ticker} after {e.attempts} attempts. Last error: {e.last_exception}")
                results[ticker] = False
            except CircuitBreakerOpenError:
                logger.error(f"Circuit breaker open for {ticker} - service temporarily unavailable")
                results[ticker] = False
            except Exception as e:
                logger.error(f"Unexpected exception processing {ticker}: {e}")
                results[ticker] = False

        return results

    def chunk_tickers(self, tickers: List[str], chunk_size: int) -> Generator[List[str], None, None]:
        """
        Split a list of tickers into chunks for memory-efficient processing

        Args:
            tickers: List of ticker symbols
            chunk_size: Number of tickers per chunk

        Yields:
            Chunks of ticker symbols
        """
        for i in range(0, len(tickers), chunk_size):
            yield tickers[i:i + chunk_size]

    async def process_in_chunks(
        self,
        tickers: List[str],
        chunk_size: int = 50,
        max_concurrent_chunks: int = 2
    ) -> Dict[str, bool]:
        """
        Process tickers in memory-efficient chunks instead of all at once

        This method processes tickers in batches to reduce memory usage and provide
        better progress tracking for large ticker lists.

        Args:
            tickers: List of ticker symbols to process
            chunk_size: Number of tickers to process per chunk
            max_concurrent_chunks: Maximum number of chunks to process concurrently

        Returns:
            Dictionary mapping ticker symbols to success status
        """
        all_results = {}
        total_chunks = (len(tickers) + chunk_size - 1) // chunk_size

        logger.info(f"Processing {len(tickers)} tickers in {total_chunks} chunks (size: {chunk_size})")

        chunk_count = 0
        for ticker_chunk in self.chunk_tickers(tickers, chunk_size):
            chunk_count += 1
            logger.info(f"Processing chunk {chunk_count}/{total_chunks} ({len(ticker_chunk)} tickers)")

            # Process chunk sequentially to avoid overwhelming the API
            chunk_results = await self.process_with_progress(ticker_chunk)
            all_results.update(chunk_results)

            # Log chunk summary
            successful = sum(1 for success in chunk_results.values() if success)
            failed = len(chunk_results) - successful
            logger.info(f"Chunk {chunk_count} complete: {successful} successful, {failed} failed")

        logger.info(f"All chunks processed. Total: {len(all_results)} tickers")
        return all_results

    async def process_with_callback(
        self,
        tickers: List[str],
        callback: Any,
        chunk_size: int = 100,
        **callback_kwargs: Any
    ) -> int:
        """
        Process tickers in chunks and call a callback function for each chunk

        This method allows processing large ticker lists with custom callback logic,
        reducing memory usage by processing in batches.

        Args:
            tickers: List of ticker symbols to process
            callback: Function to call for each processed chunk
            chunk_size: Number of tickers to process per chunk before calling callback
            **callback_kwargs: Additional keyword arguments to pass to callback

        Returns:
            Total number of tickers processed
        """
        total_processed = 0
        total_chunks = (len(tickers) + chunk_size - 1) // chunk_size

        logger.info(f"Processing {len(tickers)} tickers with callback in {total_chunks} chunks")

        chunk_count = 0
        for ticker_chunk in self.chunk_tickers(tickers, chunk_size):
            chunk_count += 1
            logger.debug(f"Processing callback chunk {chunk_count}/{total_chunks}")

            # Process the chunk
            chunk_results = await self.process_with_progress(ticker_chunk)

            # Call the callback with the results
            callback(chunk_results, chunk_number=chunk_count, **callback_kwargs)

            total_processed += len(ticker_chunk)
            logger.debug(f"Callback chunk {chunk_count} complete. Processed: {total_processed}")

        logger.info(f"Callback processing complete. Total tickers processed: {total_processed}")
        return total_processed

    @async_retry(config=API_RETRY_CONFIG, circuit_breaker=None)
    async def _process_ticker_with_retry(self, ticker: str) -> bool:
        """
        Process a single ticker with retry logic

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if successful, False otherwise
        """
        try:
            success = await self.collector.collect_fundamental_data(ticker)
            return success
        except Exception as e:
            logger.warning(f"Ticker processing failed for {ticker}: {e}")
            raise  # Let retry framework handle this

    def close(self) -> None:
        """Close the processor and cleanup resources"""
        try:
            # Close the collector
            if hasattr(self, "collector"):
                self.collector.close()

            logger.info("OptimizedFundamentalProcessor closed")

        except Exception as e:
            logger.error(f"Error during processor cleanup: {e}")

    def __del__(self) -> None:
        """Destructor to ensure cleanup"""
        self.close()

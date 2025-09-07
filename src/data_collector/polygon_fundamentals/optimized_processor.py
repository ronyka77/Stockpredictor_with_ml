"""
Optimized Fundamental Data Processor

This script processes fundamental data collection with per-ticker processing and
real-time progress tracking.
"""

from typing import List, Dict

from src.data_collector.polygon_fundamentals.optimized_collector import (
    OptimizedFundamentalCollector,
)
from src.data_collector.polygon_data.data_storage import DataStorage
from src.utils.logger import get_logger
from src.database.connection import fetch_all, get_global_pool

logger = get_logger(__name__)


class OptimizedFundamentalProcessor:
    """Optimized processor for fundamental data collection - per ticker only"""

    def __init__(self):
        # Get connection pool instead of creating new connections
        self.db_pool = get_global_pool()
        self.collector = OptimizedFundamentalCollector()
        self.data_storage = DataStorage()

    def _get_tickers_from_cache(self, filter_active: bool = True) -> List[str]:
        """Get tickers from database cache"""
        tickers = []
        try:
            rows = fetch_all(
                "SELECT ticker FROM tickers" + (" WHERE active = true" if filter_active else "")
            )
            for row in (rows or []):
                tickers.append(row["ticker"])

            logger.info(f"Loaded {len(tickers)} tickers from database")
            return tickers
        except Exception as e:
            logger.error(f"Failed to load tickers: {e}")
            return []

    async def process_with_progress(self, tickers: List[str]) -> Dict[str, bool]:
        """Process tickers with real-time progress tracking - one ticker at a time"""
        results = {}

        logger.info(f"Processing {len(tickers)} tickers - one ticker at a time")

        for i, ticker in enumerate(tickers):
            logger.info(f"Processing ticker {i + 1}/{len(tickers)}: {ticker}")

            try:
                success = await self.collector.collect_fundamental_data(ticker)
                results[ticker] = success

                # Log individual ticker results
                if success:
                    logger.info(f"✓ {ticker} - Success")
                else:
                    logger.warning(f"✗ {ticker} - Failed")

            except Exception as e:
                logger.error(f"Exception processing {ticker}: {e}")
                results[ticker] = False

        return results

    def close(self):
        """Close the processor and cleanup resources"""
        try:
            # Close the collector
            if hasattr(self, "collector"):
                self.collector.close()

            logger.info("OptimizedFundamentalProcessor closed")

        except Exception as e:
            logger.error(f"Error during processor cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.close()

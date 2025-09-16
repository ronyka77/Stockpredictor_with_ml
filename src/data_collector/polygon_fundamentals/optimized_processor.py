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
from src.database.connection import get_global_pool

logger = get_logger(__name__)


class OptimizedFundamentalProcessor:
    """Optimized processor for fundamental data collection - per ticker only"""

    def __init__(self):
        # Get connection pool instead of creating new connections
        """
        Initialize the OptimizedFundamentalProcessor.
        
        Sets up a global database connection pool and creates the collector and storage components used for per-ticker fundamental data collection:
        - self.db_pool: shared connection pool from get_global_pool()
        - self.collector: OptimizedFundamentalCollector instance
        - self.data_storage: DataStorage instance
        """
        self.db_pool = get_global_pool()
        self.collector = OptimizedFundamentalCollector()
        self.data_storage = DataStorage()

    async def process_with_progress(self, tickers: List[str]) -> Dict[str, bool]:
        """Process a list of ticker symbols sequentially, collecting fundamental data with real-time progress logging.
        
        Each ticker is processed one at a time by calling the collector's collect_fundamental_data method. Any exception raised while processing a ticker is caught and treated as a failure for that ticker.
        
        Parameters:
            tickers (List[str]): Sequence of ticker symbols to process in order.
        
        Returns:
            Dict[str, bool]: Mapping from each ticker to a boolean indicating success (True) or failure (False).
        """
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

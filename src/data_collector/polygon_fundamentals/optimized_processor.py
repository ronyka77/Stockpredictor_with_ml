"""
Optimized Fundamental Data Processor

This script processes fundamental data collection with per-ticker processing and
real-time progress tracking.
"""

import asyncio
from typing import List, Dict, Any, Optional

from src.data_collector.polygon_fundamentals.optimized_collector import OptimizedFundamentalCollector
from src.data_collector.polygon_data.data_storage import DataStorage
from src.utils.logger import get_logger
from src.data_collector.polygon_fundamentals.db_pool import get_connection_pool, close_connection_pool

logger = get_logger(__name__)

class OptimizedFundamentalProcessor:
    """Optimized processor for fundamental data collection - per ticker only"""
    
    def __init__(self):
        # Get connection pool instead of creating new connections
        self.db_pool = get_connection_pool()
        self.collector = OptimizedFundamentalCollector()
        self.data_storage = DataStorage()
    
    def _get_tickers_from_cache(self, filter_active: bool = True) -> List[str]:
        """Get tickers from database cache"""
        tickers = []
        try:
            with self.db_pool.get_connection() as conn:
                with conn.cursor() as cursor:
                    query = "SELECT ticker FROM tickers"
                    if filter_active:
                        query += " WHERE active = true"
                    
                    cursor.execute(query)
                    for row in cursor.fetchall():
                        tickers.append(row['ticker'])
        
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
    
    async def process_all_fundamentals(self) -> Dict[str, bool]:
        """Process fundamental data for all active stocks - one ticker at a time"""
        tickers = self._get_tickers_from_cache(filter_active=True)
        
        if not tickers:
            logger.error("No active tickers found")
            return {}
        
        logger.info(f"Starting fundamental data collection for {len(tickers)} active tickers - one ticker at a time")
        return await self.process_with_progress(tickers)
    
    async def process_custom_tickers(self, tickers: List[str]) -> Dict[str, bool]:
        """Process fundamental data for custom ticker list - one ticker at a time"""
        if not tickers:
            logger.error("No tickers provided")
            return {}
        
        logger.info(f"Starting fundamental data collection for {len(tickers)} custom tickers - one ticker at a time")
        return await self.process_with_progress(tickers)
    
    def get_collection_stats(self, results: Dict[str, bool]) -> Dict[str, Any]:
        """Get statistics from collection results"""
        total = len(results)
        successful = sum(1 for success in results.values() if success)
        failed = total - successful
        
        # Get failed tickers
        failed_tickers = [ticker for ticker, success in results.items() if not success]
        
        return {
            'total': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'failed_tickers': failed_tickers,
            'collector_stats': self.collector.stats
        }

async def main():
    """Main execution function"""
    processor = OptimizedFundamentalProcessor()
    
    # Process all fundamentals - one ticker at a time
    logger.info("Starting fundamental data collection - one ticker at a time...")
    results = await processor.process_all_fundamentals()
    
    # Get and display statistics
    stats = processor.get_collection_stats(results)
    
    logger.info("=== Collection Statistics ===")
    logger.info(f"Total tickers processed: {stats['total']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Success rate: {stats['success_rate']:.2%}")
    
    if stats['failed_tickers']:
        logger.warning(f"Failed tickers: {stats['failed_tickers']}")
    
    # Collector stats
    collector_stats = stats['collector_stats']
    if collector_stats['start_time'] and collector_stats['end_time']:
        elapsed = (collector_stats['end_time'] - collector_stats['start_time']).total_seconds()
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        logger.info(f"Average rate: {collector_stats['total_processed']/elapsed*60:.1f} tickers/minute")
        logger.info(f"Skipped: {collector_stats['skipped']} tickers")
    
    logger.info("Collection complete!")

if __name__ == "__main__":
    asyncio.run(main()) 
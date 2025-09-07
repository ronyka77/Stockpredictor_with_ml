"""
Main data pipeline orchestrator for Polygon.io data acquisition
"""

from typing import Dict, Optional, Any, Union
from datetime import datetime, date, timedelta
import time
import json
from pathlib import Path

from src.utils.logger import get_logger
from src.data_collector.polygon_data.client import PolygonDataClient
from src.data_collector.ticker_manager import TickerManager
from src.data_collector.polygon_data.data_fetcher import HistoricalDataFetcher
from src.data_collector.polygon_data.data_storage import DataStorage
from src.data_collector.polygon_data.data_validator import DataValidator
from src.data_collector.config import config

logger = get_logger(__name__, utility="data_collector")


class PipelineStats:
    """Statistics tracking for pipeline execution"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.end_time = None
        self.tickers_processed = 0
        self.tickers_successful = 0
        self.tickers_failed = 0
        self.total_records_fetched = 0
        self.total_records_stored = 0
        self.total_api_calls = 0
        self.errors = []
        
    def add_ticker_result(self, ticker: str, success: bool, records_count: int = 0, error: str = None):
        """Add result for a processed ticker"""
        self.tickers_processed += 1
        if success:
            self.tickers_successful += 1
            self.total_records_fetched += records_count
        else:
            self.tickers_failed += 1
            if error:
                self.errors.append(f"{ticker}: {error}")
    
    def finish(self):
        """Mark pipeline as finished"""
        self.end_time = datetime.now()
    
    @property
    def duration(self) -> timedelta:
        """Get pipeline duration"""
        end = self.end_time or datetime.now()
        return end - self.start_time
    
    @property
    def success_rate(self) -> float:
        """Get success rate percentage"""
        if self.tickers_processed == 0:
            return 0.0
        return (self.tickers_successful / self.tickers_processed) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary"""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration.total_seconds(),
            'tickers_processed': self.tickers_processed,
            'tickers_successful': self.tickers_successful,
            'tickers_failed': self.tickers_failed,
            'success_rate': self.success_rate,
            'total_records_fetched': self.total_records_fetched,
            'total_records_stored': self.total_records_stored,
            'total_api_calls': self.total_api_calls,
            'error_count': len(self.errors),
            'errors': self.errors[:10]  # Limit to first 10 errors
        }


class DataPipeline:
    """
    Main data acquisition pipeline that orchestrates all components
    
    Coordinates ticker discovery, data fetching, validation, and storage
    for efficient and reliable data acquisition from Polygon.io.
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                    db_connection: Optional[str] = None,
                    requests_per_minute: int = 5):
        """
        Initialize the data pipeline
        
        Args:
            api_key: Polygon.io API key (defaults to config)
            db_connection: Database connection string (defaults to config)
            requests_per_minute: Rate limit for API requests
        """
        # Initialize components
        self.client = PolygonDataClient(api_key, requests_per_minute)
        self.ticker_manager = TickerManager(self.client)
        self.data_fetcher = HistoricalDataFetcher(self.client)
        self.storage = DataStorage(db_connection)
        self.validator = DataValidator(strict_mode=False)
        
        # Pipeline state
        self.stats = PipelineStats()
        
        logger.info("Data pipeline initialized")
    
    def run_grouped_daily_pipeline(self, start_date: Union[str, date], 
                                    end_date: Union[str, date],
                                    validate_data: bool = True,
                                    save_stats: bool = True) -> PipelineStats:
        """
        Run pipeline to fetch grouped daily data for each day in the date range
        
        This method loops through each day between start_date and end_date,
        fetches grouped daily data for all stocks on that day using get_grouped_daily_data,
        and stores the results in the database.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            validate_data: Whether to validate the data
            save_stats: Whether to save pipeline statistics
            
        Returns:
            Pipeline statistics
        """
        # Convert string dates to date objects if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        logger.info(f"Starting grouped daily pipeline: {start_date} to {end_date}")
        
        try:
            # Step 1: Health checks
            self._perform_health_checks()
            
            # Step 2: Generate list of dates to process
            current_date = start_date
            dates_to_process = []
            
            while current_date <= end_date:
                # Skip weekends (Saturday=5, Sunday=6)
                if current_date.weekday() < 5:
                    dates_to_process.append(current_date)
                current_date += timedelta(days=1)
            
            logger.info(f"Processing {len(dates_to_process)} trading days")
            
            # Step 3: Process each date
            total_records_processed = 0
            
            for i, target_date in enumerate(dates_to_process, 1):
                logger.info(f"Processing date {i}/{len(dates_to_process)}: {target_date}")
                
                try:
                    # Fetch grouped daily data for this date
                    grouped_data = self.data_fetcher.get_grouped_daily_data(
                        target_date=target_date,
                        validate_data=validate_data
                    )
                    
                    if not grouped_data:
                        logger.warning(f"No grouped data returned for {target_date}")
                        self.stats.add_ticker_result(f"date_{target_date}", False, 0, "No data returned")
                        continue
                    
                    # Convert grouped data to list of records for storage
                    records_to_store = list(grouped_data.values())
                    
                    if records_to_store:
                        storage_result = self.storage.store_historical_data(records_to_store)
                        stored_count = storage_result.get('stored_count', 0)
                        
                        self.stats.total_records_stored += stored_count
                        total_records_processed += len(records_to_store)
                        
                        # Update stats - treat each date as a successful "ticker"
                        self.stats.add_ticker_result(
                            f"date_{target_date}", 
                            True, 
                            len(records_to_store)
                        )
                        
                        logger.info(f"Stored {stored_count} records for {len(grouped_data)} tickers on {target_date}")
                    else:
                        self.stats.add_ticker_result(f"date_{target_date}", False, 0, "No valid records")
                
                except Exception as e:
                    logger.error(f"Error processing date {target_date}: {e}")
                    self.stats.add_ticker_result(f"date_{target_date}", False, 0, str(e))
                
                # Brief pause between dates – skip when rate limiting disabled
                if i < len(dates_to_process):
                    if not getattr(config, 'DISABLE_RATE_LIMITING', False):
                        time.sleep(0.5)
            
            # Step 4: Finalize
            self.stats.finish()
            
            logger.info(f"Grouped daily pipeline completed: {self.stats.tickers_successful}/{self.stats.tickers_processed} "
                        f"dates successful ({self.stats.success_rate:.1f}%)")
            logger.info(f"Total records processed: {total_records_processed}, stored: {self.stats.total_records_stored}")
            
            # Save statistics if requested
            if save_stats:
                self._save_pipeline_stats()
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Grouped daily pipeline failed: {e}")
            self.stats.finish()
            raise
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on all components"""
        logger.info("Performing health checks...")
        
        # Check API connectivity
        if not self.client.health_check():
            raise RuntimeError("Polygon.io API health check failed")
        
        # Check database connectivity
        db_health = self.storage.health_check()
        if db_health['status'] != 'healthy':
            raise RuntimeError(f"Database health check failed: {db_health.get('error')}")
        
        # Ensure database tables exist
        self.storage.create_tables()
        
        logger.info("All health checks passed")
    
    def _save_pipeline_stats(self) -> None:
        """Save pipeline statistics to file"""
        try:
            stats_dir = Path("pipeline_stats")
            stats_dir.mkdir(exist_ok=True)
            
            timestamp = self.stats.start_time.strftime("%Y%m%d_%H%M%S")
            stats_file = stats_dir / f"pipeline_stats_{timestamp}.json"
            
            with open(stats_file, 'w') as f:
                json.dump(self.stats.to_dict(), f, indent=2)
            
            logger.info(f"Pipeline statistics saved to {stats_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save pipeline statistics: {e}")
    
    def cleanup(self) -> None:
        """Cleanup pipeline resources"""
        try:
            if hasattr(self.client, 'session'):
                self.client.session.close()
            
            if hasattr(self.storage, 'engine'):
                self.storage.engine.dispose()
                
            logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup() 

if __name__ == "__main__":
    pipeline = DataPipeline()
    from datetime import datetime, timedelta
    
    # Calculate last 1 week from today
    end_date = datetime.now().date()
    start_date = datetime.now() - timedelta(days=7)
    
    pipeline.run_grouped_daily_pipeline(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        validate_data=True,
        save_stats=True)
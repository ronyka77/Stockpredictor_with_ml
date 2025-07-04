"""
Example usage of the Polygon.io data acquisition pipeline

This script demonstrates how to use the various components of the data pipeline
to fetch, validate, and store stock market data from Polygon.io.
"""

import logging
from datetime import date, timedelta

# Use relative imports since we're in the polygon package
from src.data_collector.polygon_data import DataPipeline, PolygonDataClient, TickerManager
from src.data_collector.config import LOGGING_CONFIG
import logging.config

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def example_single_ticker():
    """Example: Fetch data for a single ticker"""
    print("\n=== Single Ticker Example ===")
    
    with DataPipeline() as pipeline:
        # Fetch recent data for Apple
        result = pipeline.run_single_ticker(
            ticker="AAPL",
            start_date=date.today() - timedelta(days=30),
            end_date=date.today() - timedelta(days=1),
            timespan="day",
            validate_data=True
        )
        
        print(f"Result: {result}")


def example_popular_tickers():
    """Example: Fetch data for popular tickers"""
    print("\n=== Popular Tickers Example ===")
    
    with DataPipeline() as pipeline:
        # Run pipeline for popular tickers (last 7 days)
        stats = pipeline.run_incremental_update(
            days_back=7,
            ticker_source="popular",
            max_tickers=10  # Limit for demo
        )
        
        print("Pipeline completed:")
        print(f"- Tickers processed: {stats.tickers_processed}")
        print(f"- Success rate: {stats.success_rate:.1f}%")
        print(f"- Records fetched: {stats.total_records_fetched}")
        print(f"- Duration: {stats.duration}")


def example_sp500_historical():
    """Example: Fetch historical data for S&P 500 stocks"""
    print("\n=== S&P 500 Historical Data Example ===")
    
    with DataPipeline() as pipeline:
        # Fetch 3 months of data for S&P 500 stocks
        stats = pipeline.run_full_pipeline(
            start_date=date.today() - timedelta(days=90),
            end_date=date.today(),
            ticker_source="sp500",
            max_tickers=5,  # Limit for demo
            timespan="day",
            validate_data=True,
            batch_size=5
        )
        
        print("S&P 500 pipeline completed:")
        print(f"- Tickers processed: {stats.tickers_processed}")
        print(f"- Success rate: {stats.success_rate:.1f}%")
        print(f"- Records stored: {stats.total_records_stored}")


def example_ticker_discovery():
    """Example: Discover and filter tickers"""
    print("\n=== Ticker Discovery Example ===")
    
    client = PolygonDataClient()
    ticker_manager = TickerManager(client)
    
    # Get S&P 500 tickers
    sp500_tickers = ticker_manager.get_sp500_tickers()
    print(f"S&P 500 tickers available: {len(sp500_tickers)}")
    print(f"First 10: {sp500_tickers[:10]}")
    
    # Get popular tickers
    popular_tickers = ticker_manager.get_popular_tickers(20)
    print(f"Popular tickers: {popular_tickers}")
    
    # Get cache statistics
    cache_stats = ticker_manager.get_cache_stats()
    print(f"Cache statistics: {cache_stats}")


def example_data_validation():
    """Example: Data validation and quality checks"""
    print("\n=== Data Validation Example ===")
    
    with DataPipeline() as pipeline:
        # Fetch data with detailed validation
        records, metrics = pipeline.data_fetcher.get_historical_data(
            ticker="MSFT",
            start_date=date.today() - timedelta(days=30),
            end_date=date.today(),
            validate_data=True
        )
        
        print("Validation results for MSFT:")
        print(f"- Total records: {metrics.total_records}")
        print(f"- Valid records: {metrics.valid_records}")
        print(f"- Success rate: {metrics.success_rate:.1f}%")
        print(f"- Validation errors: {len(metrics.validation_errors)}")
        print(f"- Data gaps: {len(metrics.data_gaps)}")
        print(f"- Outliers detected: {len(metrics.outliers)}")
        
        if metrics.validation_errors:
            print(f"Sample errors: {metrics.validation_errors[:3]}")


def example_database_operations():
    """Example: Database operations"""
    print("\n=== Database Operations Example ===")
    
    with DataPipeline() as pipeline:
        storage = pipeline.storage
        
        # Get database statistics
        stats = storage.get_data_stats()
        print("Database statistics:")
        print(f"- Total records: {stats['total_records']}")
        print(f"- Unique tickers: {stats['unique_tickers']}")
        print(f"- Date range: {stats['min_date']} to {stats['max_date']}")
        
        # Get available tickers
        available_tickers = storage.get_available_tickers()
        print(f"Available tickers in database: {len(available_tickers)}")
        if available_tickers:
            print(f"Sample tickers: {available_tickers[:10]}")
        
        # Health check
        health = storage.health_check()
        print(f"Database health: {health['status']}")


def example_rate_limiting():
    """Example: Rate limiting and API management"""
    print("\n=== Rate Limiting Example ===")
    
    client = PolygonDataClient(requests_per_minute=5)  # Free tier limit
    
    # Check rate limit status
    status = client.get_rate_limit_status()
    print("Rate limit status:")
    print(f"- Requests per minute: {status['requests_per_minute']}")
    print(f"- Remaining requests: {status['remaining_requests']}")
    print(f"- Time until reset: {status['time_until_reset']:.1f}s")
    
    # Make a few API calls to demonstrate rate limiting
    print("Making API calls...")
    for i in range(3):
        try:
            tickers = client.get_tickers(limit=1)
            print(f"Call {i+1}: Retrieved {len(tickers)} ticker(s)")
            
            # Check status after each call
            status = client.get_rate_limit_status()
            print(f"  Remaining: {status['remaining_requests']}")
            
        except Exception as e:
            print(f"Call {i+1} failed: {e}")


def main():
    """Run all examples"""
    print("Polygon.io Data Acquisition Pipeline Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_ticker_discovery()
        example_single_ticker()
        example_data_validation()
        example_database_operations()
        example_rate_limiting()
        
        # Uncomment these for longer-running examples
        # example_popular_tickers()
        # example_sp500_historical()
        
        print("\n=== All Examples Completed Successfully! ===")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 
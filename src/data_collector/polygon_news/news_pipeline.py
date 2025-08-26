"""
Main news collection orchestrator for Polygon news data
Handles incremental updates, historical backfill, and comprehensive news collection
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from src.data_collector.polygon_news.news_client import PolygonNewsClient
from src.data_collector.polygon_news.storage import PolygonNewsStorage
from src.data_collector.polygon_news.ticker_integration import NewsTickerIntegration
from src.data_collector.polygon_news.processor import NewsProcessor
from src.data_collector.polygon_news.validator import NewsValidator
from src.utils.logger import get_logger

logger = get_logger(__name__, utility="data_collector")


class PolygonNewsCollector:
    """
    Main orchestrator for Polygon news data collection
    
    Provides three main collection modes:
    1. Incremental Update: From last stored date to current
    2. Historical Backfill: 2-year historical data collection  
    3. Targeted Collection: Specific tickers and date ranges
    """
    
    def __init__(self, 
                    db_session: Session,
                    polygon_api_key: Optional[str] = None,
                    requests_per_minute: int = 5):
        """
        Initialize the news collector
        
        Args:
            db_session: Database session for storage operations
            polygon_api_key: Polygon.io API key
            requests_per_minute: Rate limit for API requests
        """
        self.db_session = db_session
        self.logger = get_logger(self.__class__.__name__, utility="data_collector")
        
        # Initialize components
        self.news_client = PolygonNewsClient(polygon_api_key, requests_per_minute)
        self.storage = PolygonNewsStorage(db_session)
        self.ticker_integration = NewsTickerIntegration()
        self.processor = NewsProcessor()
        self.validator = NewsValidator()
        
        # Collection statistics
        self.stats = {
            'total_api_calls': 0,
            'total_articles_fetched': 0,
            'total_articles_stored': 0,
            'total_articles_updated': 0,
            'total_articles_skipped': 0,
            'failed_tickers': [],
            'processing_errors': [],
            'start_time': None,
            'end_time': None
        }
        
        self.logger.info("Polygon News Collector initialized")
    
    def collect_historical_news(self, 
                                max_tickers: int = 50,
                                years_back: int = 2,
                                batch_size_days: int = 30) -> Dict[str, Any]:
        """
        Collect historical news data (2-year backfill)
        
        Args:
            max_tickers: Maximum number of tickers to process
            years_back: Number of years to go back
            batch_size_days: Process in batches of this many days
            
        Returns:
            Collection statistics
        """
        self.logger.info(f"Starting historical news collection ({years_back} years)")
        self._reset_stats()
        from datetime import timezone
        self.stats['start_time'] = datetime.now(timezone.utc)
        
        try:
            # Get high-priority tickers for historical collection
            ticker_info_list = self.ticker_integration.get_prioritized_tickers(max_tickers)
            major_tickers = [info for info in ticker_info_list if info['is_major']][:20]
            
            self.logger.info(f"Processing {len(major_tickers)} major tickers for historical backfill")
            
            # Calculate date range (ensure timezone consistency)
            from datetime import timezone
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=years_back * 365)
            
            # Process in batches to avoid overwhelming the API
            current_date = start_date
            
            while current_date < end_date:
                batch_end = min(current_date + timedelta(days=batch_size_days), end_date)
                
                self.logger.info(f"Processing batch: {current_date.date()} to {batch_end.date()}")
                
                for ticker_info in major_tickers:
                    ticker = ticker_info['ticker']
                    
                    try:
                        # Check if we already have data for this period
                        existing_articles = self.storage.get_articles_for_ticker(
                            ticker, current_date, batch_end, limit=1
                        )
                        
                        if existing_articles:
                            self.logger.info(f"Ticker {ticker}: data exists for {current_date.date()}-{batch_end.date()}")
                            continue
                        
                        # Collect news for this ticker and date range
                        ticker_stats = self._collect_ticker_news(
                            ticker, current_date, batch_end, ticker_info['priority_score']
                        )
                        
                        # Update overall stats
                        self._update_stats(ticker_stats)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing ticker {ticker} for {current_date.date()}: {e}")
                        self.stats['failed_tickers'].append(f"{ticker}_{current_date.date()}")
                        self.stats['processing_errors'].append(f"{ticker}_{current_date.date()}: {str(e)}")
                
                current_date = batch_end
            
            self.stats['end_time'] = datetime.now(timezone.utc)
            self.logger.info(f"Historical collection completed: {self._format_stats()}")
            
            return self.stats.copy()
            
        except Exception as e:
            self.logger.error(f"Historical collection failed: {e}")
            self.stats['end_time'] = datetime.now(timezone.utc)
            raise
    
    def collect_targeted_news(self,
                                tickers: List[str],
                                start_date: datetime,
                                end_date: datetime,
                                limit_per_ticker: int = 100) -> Dict[str, Any]:
        """
        Collect news for specific tickers and date range
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for collection
            end_date: End date for collection
            limit_per_ticker: Maximum articles per ticker
            
        Returns:
            Collection statistics
        """
        self.logger.info(f"Starting targeted collection for {len(tickers)} tickers")
        self._reset_stats()
        from datetime import timezone
        self.stats['start_time'] = datetime.now(timezone.utc)
        
        try:
            # Validate tickers
            valid_tickers, invalid_tickers = self.ticker_integration.validate_ticker_list(tickers)
            
            if invalid_tickers:
                self.logger.warning(f"Invalid tickers skipped: {invalid_tickers}")
            
            self.logger.info(f"Processing {len(valid_tickers)} valid tickers")
            
            for ticker in valid_tickers:
                try:
                    # Get ticker info for priority scoring
                    ticker_info = self.ticker_integration.get_ticker_info(ticker)
                    priority_score = ticker_info.get('priority_score', 50)
                    
                    # Collect news for this ticker
                    ticker_stats = self._collect_ticker_news(
                        ticker, start_date, end_date, priority_score, limit_per_ticker
                    )
                    
                    # Update overall stats
                    self._update_stats(ticker_stats)
                    
                except Exception as e:
                    self.logger.error(f"Error processing ticker {ticker}: {e}")
                    self.stats['failed_tickers'].append(ticker)
                    self.stats['processing_errors'].append(f"{ticker}: {str(e)}")
            
            self.stats['end_time'] = datetime.now(timezone.utc)
            self.logger.info(f"Targeted collection completed: {self._format_stats()}")
            
            return self.stats.copy()
            
        except Exception as e:
            self.logger.error(f"Targeted collection failed: {e}")
            self.stats['end_time'] = datetime.now()
            raise
    
    def _collect_ticker_news(self,
                            ticker: str,
                            start_date: datetime,
                            end_date: datetime,
                            priority_score: float,
                            limit: int = 1000) -> Dict[str, int]:
        """
        Collect news for a single ticker within date range
        
        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date
            priority_score: Priority score for processing
            limit: Maximum articles to fetch
            
        Returns:
            Statistics for this ticker
        """
        ticker_stats = {
            'api_calls': 0,
            'articles_fetched': 0,
            'articles_stored': 0,
            'articles_updated': 0,
            'articles_skipped': 0
        }
        
        try:
            # Fetch news from Polygon API
            self.logger.info(f"Fetching news for {ticker} from {start_date.date()} to {end_date.date()}")
            
            raw_articles = self.news_client.get_news_for_ticker(
                ticker=ticker,
                published_utc_gte=start_date,
                published_utc_lte=end_date,
                limit=limit
            )
            
            ticker_stats['api_calls'] = 1
            ticker_stats['articles_fetched'] = len(raw_articles)
            
            if not raw_articles:
                self.logger.info(f"No articles found for {ticker}")
                return ticker_stats
            
            # Process articles
            processed_articles = []
            
            for raw_article in raw_articles:
                try:
                    # Extract metadata
                    article_metadata = self.news_client.extract_article_metadata(raw_article)
                    
                    # Process content
                    processed_article = self.processor.process_article(article_metadata)
                    
                    # Validate quality
                    is_valid, quality_score, issues = self.validator.validate_article(processed_article)
                    
                    if is_valid:
                        processed_article['quality_score'] = quality_score
                        processed_article['relevance_score'] = priority_score / 100.0  # Normalize to 0-1
                        processed_articles.append(processed_article)
                    else:
                        self.logger.info(f"Article failed validation: {issues}")
                        ticker_stats['articles_skipped'] += 1
                
                except Exception as e:
                    self.logger.warning(f"Error processing article: {e}")
                    ticker_stats['articles_skipped'] += 1
            
            # Store articles in batch
            if processed_articles:
                storage_stats = self.storage.store_articles_batch(processed_articles)
                ticker_stats['articles_stored'] = storage_stats['new_articles']
                ticker_stats['articles_updated'] = storage_stats['updated_articles']
                ticker_stats['articles_skipped'] += storage_stats['skipped_articles']
            
            return ticker_stats
            
        except Exception as e:
            self.logger.error(f"Error collecting news for {ticker}: {e}")
            raise
    
    def _reset_stats(self):
        """Reset collection statistics"""
        self.stats = {
            'total_api_calls': 0,
            'total_articles_fetched': 0,
            'total_articles_stored': 0,
            'total_articles_updated': 0,
            'total_articles_skipped': 0,
            'failed_tickers': [],
            'processing_errors': [],
            'start_time': None,
            'end_time': None
        }
    
    def _update_stats(self, ticker_stats: Dict[str, int]):
        """Update overall statistics with ticker statistics"""
        self.stats['total_api_calls'] += ticker_stats.get('api_calls', 0)
        self.stats['total_articles_fetched'] += ticker_stats.get('articles_fetched', 0)
        self.stats['total_articles_stored'] += ticker_stats.get('articles_stored', 0)
        self.stats['total_articles_updated'] += ticker_stats.get('articles_updated', 0)
        self.stats['total_articles_skipped'] += ticker_stats.get('articles_skipped', 0)
    
    def _format_stats(self) -> str:
        """Format statistics for logging"""
        duration = None
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
        
        return (f"API calls: {self.stats['total_api_calls']}, "
                f"Fetched: {self.stats['total_articles_fetched']}, "
                f"Stored: {self.stats['total_articles_stored']}, "
                f"Updated: {self.stats['total_articles_updated']}, "
                f"Skipped: {self.stats['total_articles_skipped']}, "
                f"Failed tickers: {len(self.stats['failed_tickers'])}, "
                f"Duration: {duration}")
    
    def get_collection_status(self) -> Dict[str, Any]:
        """Get current collection status and statistics"""
        try:
            health_check = self.storage.health_check()
            latest_date = self.storage.get_latest_date_overall()
            
            # Get recent statistics
            from datetime import timezone
            recent_stats = self.storage.get_article_statistics(
                start_date=datetime.now(timezone.utc) - timedelta(days=30)
            )
            
            return {
                'status': 'healthy' if health_check['status'] == 'healthy' else 'unhealthy',
                'database_health': health_check,
                'latest_article_date': latest_date.isoformat() if latest_date else None,
                'recent_statistics': recent_stats,
                'last_collection_stats': self.stats
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'last_collection_stats': self.stats
            }
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # Cleanup if needed
        pass


def main():
    """
    Main function to run incremental news collection
    
    This function sets up the database connection, initializes the news collector,
    and runs incremental news collection with proper error handling and logging.
    
    Environment Variables Required:
        - POLYGON_API_KEY: Your Polygon.io API key
        - DATABASE_URL: PostgreSQL connection string (optional, uses centralized config)
        - NEWS_MAX_TICKERS: Maximum tickers to process (optional, default: 100)
        - NEWS_DAYS_LOOKBACK: Days to look back for new tickers (optional, default: 7)
        - NEWS_RETENTION_YEARS: Years to retain news data (optional, default: 2)
    
    Usage:
        python -m src.data_collector.polygon_news.news_collector
        
    Or from another script:
        from src.data_collector.polygon_news.news_collector import main
        main()
    """
    import os
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from src.data_collector.polygon_news.models import create_tables
    from src.data_collector.config import config
    
    try:
        logger.info("=" * 60)
        logger.info("Starting Polygon News Collection - Incremental Update")
        logger.info("=" * 60)
        
        # Check for required API key from config
        if not config.API_KEY:
            logger.error("POLYGON_API_KEY environment variable not set")
            logger.error("Please set your Polygon.io API key:")
            logger.error("export POLYGON_API_KEY='your_api_key_here'")
            return False
        
        # Database connection using centralized config
        database_url = os.getenv('DATABASE_URL', config.database_url)
        logger.info("Connecting to database...")
        logger.info(f"Using database: {config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}")
        
        try:
            engine = create_engine(database_url)
            
            # Create tables if they don't exist (safe for existing tables)
            logger.info("Ensuring database tables exist...")
            create_tables(engine)
            
            # Create session
            Session = sessionmaker(bind=engine)
            session = Session()
            
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            logger.error("Please check your DATABASE_URL environment variable")
            return False
        
        # Initialize news collector
        try:
            logger.info("Initializing Polygon News Collector...")
            
            collector = PolygonNewsCollector(
                db_session=session,
                polygon_api_key=config.API_KEY,
                requests_per_minute=config.REQUESTS_PER_MINUTE
            )
            
            logger.info("News collector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize news collector: {e}")
            session.close()
            engine.dispose()
            return False
        
        # Run incremental news collection
        try:
            logger.info("Starting incremental news collection...")
            logger.info("This will collect news from the last stored date to current date")
            
            # Configuration for incremental collection from centralized config
            max_tickers = config.NEWS_MAX_TICKERS
            days_lookback = config.NEWS_DAYS_LOOKBACK
            
            logger.info("Configuration:")
            logger.info(f"  - Max tickers: {max_tickers}")
            logger.info(f"  - Days lookback (for new tickers): {days_lookback}")
            logger.info(f"  - Rate limit: {config.REQUESTS_PER_MINUTE} requests/minute")
            logger.info(f"  - Retention period: {config.NEWS_RETENTION_YEARS} years")
            
            # Execute collection
            stats_historical = collector.collect_historical_news(
                max_tickers=3000,
                years_back=1,
                batch_size_days=30
            )
            
            # Log results
            logger.info("=" * 60)
            logger.info("COLLECTION COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info("Collection Statistics:")
            logger.info(f"  - Total API calls: {stats_historical['total_api_calls']}")
            logger.info(f"  - Articles fetched: {stats_historical['total_articles_fetched']}")
            logger.info(f"  - Articles stored: {stats_historical['total_articles_stored']}")
            logger.info(f"  - Articles updated: {stats_historical['total_articles_updated']}")
            logger.info(f"  - Articles skipped: {stats_historical['total_articles_skipped']}")
            logger.info(f"  - Failed tickers: {len(stats_historical['failed_tickers'])}")
            
            if stats_historical['failed_tickers']:
                logger.warning(f"Failed tickers: {stats_historical['failed_tickers']}")
            
            if stats_historical['processing_errors']:
                logger.warning(f"Processing errors: {len(stats_historical['processing_errors'])}")
                for error in stats_historical['processing_errors'][:5]:  # Show first 5 errors
                    logger.warning(f"  - {error}")
            
            # Duration
            if stats_historical['start_time'] and stats_historical['end_time']:
                duration = stats_historical['end_time'] - stats_historical['start_time']
                logger.info(f"  - Duration: {duration}")
            
            # Get system status
            try:
                status = collector.get_collection_status()
                logger.info("System Status:")
                logger.info(f"  - Database health: {status.get('status', 'unknown')}")
                logger.info(f"  - Latest article date: {status.get('latest_article_date', 'None')}")
                
                if 'recent_statistics' in status:
                    recent_stats = status['recent_statistics']
                    logger.info(f"  - Total articles in DB: {recent_stats.get('total_articles', 0)}")
                    
                    top_tickers = recent_stats.get('top_tickers', {})
                    if top_tickers:
                        logger.info(f"  - Top tickers: {dict(list(top_tickers.items())[:5])}")
                    
                    sentiment_dist = recent_stats.get('sentiment_distribution', {})
                    if sentiment_dist:
                        logger.info(f"  - Sentiment distribution: {sentiment_dist}")
                        
            except Exception as e:
                logger.warning(f"Could not retrieve system status: {e}")
            
            logger.info("=" * 60)
            return True
        except Exception as e:
            logger.error(f"News collection failed: {e}")
            logger.error("Check the error logs for detailed information")
            return False
            
        finally:
            # Cleanup
            try:
                session.close()
                engine.dispose()
                logger.info("Database connection closed")
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
    
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error in main function: {e}")
        return False


if __name__ == "__main__":
    """
    Entry point when running the module directly
    
    Usage:
        python -m src.data_collector.polygon_news.news_collector
        
    Or:
        cd src/data_collector/polygon_news
        python news_collector.py
    """
    import sys
    
    # Run main function and exit with appropriate code
    success = main()
    sys.exit(0 if success else 1) 
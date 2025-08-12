"""
Ticker management and discovery functionality for Polygon.io data
"""

from typing import List, Dict, Optional

from src.utils.logger import get_polygon_logger
from src.data_collector.polygon_data.client import PolygonDataClient
from src.data_collector.polygon_data.data_validator import DataValidator
from src.data_collector.polygon_data.data_storage import DataStorage

logger = get_polygon_logger(__name__)


class TickerManager:
    """
    Manages ticker discovery, filtering, and caching for efficient data acquisition
    """
    
    def __init__(self, client: PolygonDataClient, storage: Optional[DataStorage] = None):
        """
        Initialize the ticker manager
        
        Args:
            client: Polygon.io API client
            storage: Database storage instance (optional, will create if not provided)
        """
        self.client = client
        self.storage = storage or DataStorage()
        self.validator = DataValidator(strict_mode=False)
        
        logger.info("Ticker manager initialized with database storage")
    
    def get_all_active_tickers(self, market: str = "stocks") -> List[str]:
        """
        Get all active stock tickers from database
        
        Args:
            market: Market type (stocks, crypto, fx, etc.)
            
        Returns:
            List of active ticker symbols
        """
        try:
            db_tickers = self.storage.get_ticker_symbols(market=market, active=True)
            if db_tickers:
                logger.info(f"Retrieved {len(db_tickers)} {market} tickers from database")
                return db_tickers
        except Exception as e:
            logger.error(f"Failed to get tickers from database: {e}")
            raise
        
        logger.warning(f"No {market} tickers found in database")
        return []
    
    def get_ticker_details(self, ticker: str) -> Optional[Dict]:
        """
        Get detailed information about a specific ticker from database
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Ticker details dictionary or None if not found
        """
        try:
            # Get ticker from database using a more efficient query
            with self.storage.engine.connect() as conn:
                from sqlalchemy import text
                result = conn.execute(
                    text("SELECT * FROM tickers WHERE ticker = :ticker LIMIT 1"),
                    {'ticker': ticker.upper()}
                )
                row = result.fetchone()
                
                if row:
                    # Convert database row to dictionary
                    details = dict(row._mapping)
                    logger.info(f"Retrieved details for {ticker} from database")
                    return details
                
        except Exception as e:
            logger.error(f"Failed to get ticker details from database: {e}")
        
        logger.warning(f"No details found for ticker {ticker}")
        return None
    
    def filter_tickers_by_criteria(self, tickers: List[str], 
                                    min_market_cap: Optional[float] = None,
                                    min_volume: Optional[int] = None,
                                    exchanges: Optional[List[str]] = None,
                                    exclude_otc: bool = True,
                                    max_tickers: Optional[int] = None) -> List[str]:
        """
        Filter tickers based on various criteria
        
        Args:
            tickers: List of ticker symbols to filter
            min_market_cap: Minimum market capitalization
            min_volume: Minimum average daily volume
            exchanges: List of allowed exchanges
            exclude_otc: Whether to exclude OTC stocks
            max_tickers: Maximum number of tickers to return
            
        Returns:
            Filtered list of ticker symbols
        """
        logger.info(f"Filtering {len(tickers)} tickers with criteria...")
        
        filtered_tickers = []
        
        for ticker in tickers:
            try:
                # Basic filtering based on ticker format
                if not self._is_valid_ticker_format(ticker):
                    continue
                
                # Exclude OTC stocks if requested
                if exclude_otc and self._is_otc_ticker(ticker):
                    continue
                
                # Get ticker details for advanced filtering
                if min_market_cap or min_volume or exchanges:
                    details = self.get_ticker_details(ticker)
                    if not details:
                        continue
                    
                    # Filter by exchange
                    if exchanges:
                        ticker_exchange = details.get('primary_exchange', '')
                        if ticker_exchange not in exchanges:
                            continue
                    
                    # Note: Market cap and volume filtering would require additional API calls
                    # For now, we'll implement basic filtering
                
                filtered_tickers.append(ticker)
                
                # Limit number of tickers if specified
                if max_tickers and len(filtered_tickers) >= max_tickers:
                    break
                    
            except Exception as e:
                logger.info(f"Error filtering ticker {ticker}: {e}")
                continue
        
        logger.info(f"Filtered to {len(filtered_tickers)} tickers")
        return filtered_tickers
    
    def _is_valid_ticker_format(self, ticker: str) -> bool:
        """
        Check if ticker has a valid format
        
        Args:
            ticker: Ticker symbol to validate
            
        Returns:
            True if ticker format is valid
        """
        if not ticker or len(ticker) < 1 or len(ticker) > 10:
            return False
        
        # Basic validation - alphanumeric with some special characters
        cleaned = ticker.replace('.', '').replace('-', '').replace('/', '')
        return cleaned.isalnum()
    
    def _is_otc_ticker(self, ticker: str) -> bool:
        """
        Check if ticker is an OTC (Over-The-Counter) stock
        
        Args:
            ticker: Ticker symbol to check
            
        Returns:
            True if ticker appears to be OTC
        """
        # OTC stocks often have certain patterns
        otc_patterns = [
            ticker.endswith('F'),  # Foreign stocks
            len(ticker) == 5 and ticker.endswith('Y'),  # ADRs
            '.' in ticker and ticker.split('.')[1] in ['PK', 'OB'],  # Pink sheets, OTC Bulletin Board
        ]
        
        return any(otc_patterns)
    
    def refresh_ticker_data(self, market: str = "stocks", batch_size: int = 100, 
                            force_refresh: bool = False) -> Dict[str, int]:
        """
        Refresh ticker information in database by fetching from Polygon.io API
        
        Args:
            market: Market type to refresh (stocks, crypto, fx, etc.)
            batch_size: Number of tickers to process in each batch
            force_refresh: If True, refresh all tickers. If False, only refresh missing/outdated ones
            
        Returns:
            Dictionary with refresh statistics
        """
        logger.info(f"🔄 Starting ticker data refresh for {market} market")
        
        stats = {
            'fetched_from_api': 0,
            'updated_in_db': 0,
            'errors': 0,
            'total_processed': 0
        }
        
        try:
            # Step 1: Fetch all active tickers from Polygon.io API
            logger.info("📡 Fetching active tickers from Polygon.io API...")
            ticker_data = self.client.get_tickers(
                market=market,
                active=True,
                limit=10000  # Get more tickers in one call
            )
            
            if not ticker_data:
                logger.warning("No ticker data received from API")
                return stats
            
            logger.info(f"📊 Received {len(ticker_data)} tickers from API")
            
            # Step 2: Get existing tickers from database for comparison
            existing_tickers = {}
            if not force_refresh:
                try:
                    db_tickers = self.storage.get_tickers(market=market)
                    existing_tickers = {t['ticker']: t for t in db_tickers}
                    logger.info(f"📋 Found {len(existing_tickers)} existing tickers in database")
                except Exception as e:
                    logger.warning(f"Failed to get existing tickers: {e}")
            
            # Step 3: Process tickers in batches
            tickers_to_update = []
            
            for item in ticker_data:
                if 'ticker' not in item:
                    continue
                    
                ticker_symbol = item['ticker'].upper()
                stats['total_processed'] += 1
                
                # Check if we need to update this ticker
                should_update = force_refresh
                
                if not should_update:
                    existing = existing_tickers.get(ticker_symbol)
                    if not existing:
                        # Missing ticker - needs to be added
                        should_update = True
                    else:
                        # Check if data has changed (compare key fields)
                        if (existing.get('name') != item.get('name') or
                            existing.get('active') != item.get('active', True) or
                            existing.get('primary_exchange') != item.get('primary_exchange') or
                            existing.get('type') != item.get('type') or
                            existing.get('currency_name') != item.get('currency_name') or
                            existing.get('market_cap') != item.get('market_cap') or
                            existing.get('weighted_shares_outstanding') != item.get('weighted_shares_outstanding') or
                            existing.get('round_lot') != item.get('round_lot') or
                            existing.get('cik') != item.get('cik') or
                            existing.get('composite_figi') != item.get('composite_figi') or
                            existing.get('share_class_figi') != item.get('share_class_figi') or
                            existing.get('sic_code') != item.get('sic_code') or
                            existing.get('sic_description') != item.get('sic_description') or
                            existing.get('ticker_root') != item.get('ticker_root') or
                            existing.get('total_employees') != item.get('total_employees') or
                            existing.get('list_date') != item.get('list_date')):
                            should_update = True
                
                if should_update:
                    # Prepare ticker data for database
                    ticker_info = {
                        'ticker': ticker_symbol,
                        'name': item.get('name'),
                        'market': market,
                        'locale': item.get('locale', 'us'),
                        'primary_exchange': item.get('primary_exchange'),
                        'currency_name': item.get('currency_name'),
                        'active': item.get('active', True),
                        'type': item.get('type'),
                        'market_cap': item.get('market_cap'),
                        'weighted_shares_outstanding': item.get('weighted_shares_outstanding'),
                        'round_lot': item.get('round_lot'),
                        'cik': item.get('cik'),
                        'composite_figi': item.get('composite_figi'),
                        'share_class_figi': item.get('share_class_figi'),
                        'sic_code': item.get('sic_code'),
                        'sic_description': item.get('sic_description'),
                        'ticker_root': item.get('ticker_root'),
                        'total_employees': item.get('total_employees'),
                        'list_date': item.get('list_date')
                    }
                    tickers_to_update.append(ticker_info)
                    stats['fetched_from_api'] += 1
                
                # Process in batches
                if len(tickers_to_update) >= batch_size:
                    try:
                        result = self.storage.store_tickers(tickers_to_update)
                        stats['updated_in_db'] += result['stored_count']
                        logger.info(f"✅ Updated batch of {len(tickers_to_update)} tickers")
                        tickers_to_update = []
                    except Exception as e:
                        logger.error(f"❌ Failed to update batch: {e}")
                        stats['errors'] += len(tickers_to_update)
                        tickers_to_update = []
            
            # Process remaining tickers
            if tickers_to_update:
                try:
                    result = self.storage.store_tickers(tickers_to_update)
                    stats['updated_in_db'] += result['stored_count']
                    logger.info(f"✅ Updated final batch of {len(tickers_to_update)} tickers")
                except Exception as e:
                    logger.error(f"❌ Failed to update final batch: {e}")
                    stats['errors'] += len(tickers_to_update)
            
            # Step 4: Mark inactive tickers
            if not force_refresh:
                try:
                    api_tickers = {item['ticker'].upper() for item in ticker_data if 'ticker' in item}
                    inactive_count = 0
                    
                    with self.storage.engine.connect() as conn:
                        from sqlalchemy import text
                        # Mark tickers as inactive if they're not in the API response
                        result = conn.execute(text("""
                            UPDATE tickers 
                            SET active = false, last_updated = CURRENT_TIMESTAMP
                            WHERE market = :market AND active = true 
                            AND ticker NOT IN :api_tickers
                        """), {
                            'market': market,
                            'api_tickers': tuple(api_tickers) if api_tickers else ('',)
                        })
                        inactive_count = result.rowcount
                        conn.commit()
                    
                    if inactive_count > 0:
                        logger.info(f"📉 Marked {inactive_count} tickers as inactive")
                        
                except Exception as e:
                    logger.warning(f"Failed to mark inactive tickers: {e}")
            
            logger.info("🎉 Ticker refresh completed successfully!")
            logger.info("📊 Refresh Summary:")
            logger.info(f"   Total processed: {stats['total_processed']}")
            logger.info(f"   Fetched from API: {stats['fetched_from_api']}")
            logger.info(f"   Updated in DB: {stats['updated_in_db']}")
            logger.info(f"   Errors: {stats['errors']}")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Ticker refresh failed: {e}")
            stats['errors'] += 1
            raise
    
    def refresh_ticker_details(self, tickers: List[str], batch_size: int = 50) -> Dict[str, int]:
        """
        Refresh detailed information for specific tickers
        
        Args:
            tickers: List of ticker symbols to refresh
            batch_size: Number of tickers to process in each batch
            
        Returns:
            Dictionary with refresh statistics
        """
        logger.info(f"🔍 Refreshing detailed information for {len(tickers)} tickers")
        
        stats = {
            'processed': 0,
            'updated': 0,
            'errors': 0,
            'not_found': 0
        }
        
        try:
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i + batch_size]
                logger.info(f"📋 Processing batch {i//batch_size + 1}: {len(batch)} tickers")
                
                for ticker in batch:
                    try:
                        stats['processed'] += 1
                        
                        # Fetch details from API
                        details = self.client.get_ticker_details(ticker)
                        
                        if details:
                            # Prepare ticker data for database
                            ticker_data = {
                                'ticker': details.get('ticker', ticker).upper(),
                                'name': details.get('name'),
                                'market': details.get('market', 'stocks'),
                                'locale': details.get('locale', 'us'),
                                'primary_exchange': details.get('primary_exchange'),
                                'currency_name': details.get('currency_name'),
                                'active': details.get('active', True),
                                'type': details.get('type'),
                                'market_cap': details.get('market_cap'),
                                'weighted_shares_outstanding': details.get('weighted_shares_outstanding'),
                                'round_lot': details.get('round_lot'),
                                'cik': details.get('cik'),
                                'composite_figi': details.get('composite_figi'),
                                'share_class_figi': details.get('share_class_figi'),
                                'sic_code': details.get('sic_code'),
                                'sic_description': details.get('sic_description'),
                                'ticker_root': details.get('ticker_root'),
                                'total_employees': details.get('total_employees'),
                                'list_date': details.get('list_date')
                            }
                            
                            # Store in database
                            result = self.storage.store_tickers([ticker_data])
                            if result['stored_count'] > 0:
                                stats['updated'] += 1
                                logger.info(f"✅ Updated details for {ticker}")
                        else:
                            stats['not_found'] += 1
                            logger.warning(f"⚠️ No details found for {ticker}")
                            
                    except Exception as e:
                        stats['errors'] += 1
                        logger.error(f"❌ Error refreshing {ticker}: {e}")
                        continue
                
                # Add small delay between batches to respect rate limits
                import time
                time.sleep(0.1)
            
            logger.info("🎉 Ticker details refresh completed!")
            logger.info("📊 Details Refresh Summary:")
            logger.info(f"   Processed: {stats['processed']}")
            logger.info(f"   Updated: {stats['updated']}")
            logger.info(f"   Not found: {stats['not_found']}")
            logger.info(f"   Errors: {stats['errors']}")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Ticker details refresh failed: {e}")
            raise
    
    def get_database_stats(self) -> Dict:
        """
        Get statistics about ticker data in database
        
        Returns:
            Dictionary with database statistics
        """
        try:
            with self.storage.engine.connect() as conn:
                from sqlalchemy import text
                
                # Get basic counts
                total_result = conn.execute(text("SELECT COUNT(*) FROM tickers"))
                total_tickers = total_result.fetchone()[0]
                
                active_result = conn.execute(text("SELECT COUNT(*) FROM tickers WHERE active = true"))
                active_tickers = active_result.fetchone()[0]
                
                # Get market breakdown
                market_result = conn.execute(text("""
                    SELECT market, COUNT(*) as count 
                    FROM tickers 
                    WHERE active = true 
                    GROUP BY market 
                    ORDER BY count DESC
                """))
                markets = {row[0]: row[1] for row in market_result}
                
                return {
                    'total_tickers': total_tickers,
                    'active_tickers': active_tickers,
                    'markets': markets
                }
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {
                'total_tickers': 0,
                'active_tickers': 0,
                'markets': {},
                'error': str(e)
            }

def main():
    """
    Main function to refresh ticker data in database
    """
    from src.utils.logger import get_general_logger
    
    logger = get_general_logger(__name__)
    
    try:
        from src.data_collector.polygon_data import PolygonDataClient, DataStorage
        from src.data_collector.config import config
        
        logger.info("🚀 Starting ticker refresh")
        
        # Initialize components
        storage = DataStorage()
        client = PolygonDataClient(api_key=config.API_KEY)
        ticker_manager = TickerManager(client, storage)
        
        # Refresh ticker data
        # ticker_manager.refresh_ticker_data(
        #     market="stocks",
        #     batch_size=1000,
        #     force_refresh=True
        # )

        ticker_manager.refresh_ticker_details(
            tickers=ticker_manager.get_all_active_tickers(market="stocks"),
            batch_size=50
        )
        
        logger.info("🎉 Ticker refresh completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Ticker refresh failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 
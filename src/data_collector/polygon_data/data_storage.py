"""
Data storage functionality for PostgreSQL database operations
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, date
import pandas as pd
from sqlalchemy import create_engine, text

from src.utils.logger import get_polygon_logger
from src.data_collector.polygon_data.data_validator import OHLCVRecord
from src.data_collector.config import config

logger = get_polygon_logger(__name__)


class DataStorage:
    """
    Handles storage and retrieval of stock market data in PostgreSQL database
    
    Provides efficient bulk operations, data integrity checks, and query capabilities.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize the data storage handler
        
        Args:
            connection_string: PostgreSQL connection string (defaults to config)
        """
        self.connection_string = connection_string or config.database_url
        self.engine = create_engine(self.connection_string, echo=False)
        
        # Test connection
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def store_historical_data(self, records: List[OHLCVRecord], 
                            batch_size: int = 1000, 
                            on_conflict: str = 'update') -> Dict[str, Any]:
        """
        Store validated OHLCV data to PostgreSQL database
        
        Args:
            records: List of validated OHLCV records
            batch_size: Number of records to insert in each batch
            on_conflict: How to handle conflicts ('update', 'ignore', 'error')
            
        Returns:
            Dictionary with storage statistics
        """
        if not records:
            logger.warning("No records to store")
            return {'stored_count': 0, 'error_count': 0, 'updated_count': 0}
        
        logger.info(f"Storing {len(records)} records to database")
        
        # Convert records to database format
        data_rows = []
        for record in records:
            row_data = record.to_dict()
            data_rows.append(row_data)
        
        # Create DataFrame for efficient processing
        df = pd.DataFrame(data_rows)
        
        # Ensure proper data types
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['volume'] = df['volume'].astype('int64')
        
        stored_count = 0
        updated_count = 0
        error_count = 0
        
        try:
            # Process in batches
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i + batch_size]
                
                try:
                    if on_conflict == 'update':
                        batch_stored, batch_updated = self._upsert_batch(batch_df)
                        stored_count += batch_stored
                        updated_count += batch_updated
                    elif on_conflict == 'ignore':
                        batch_stored = self._insert_ignore_batch(batch_df)
                        stored_count += batch_stored
                    else:  # error
                        batch_stored = self._insert_batch(batch_df)
                        stored_count += batch_stored
                        
                    logger.info(f"Processed batch {i//batch_size + 1}: "
                                f"{len(batch_df)} records")
                    
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                    error_count += len(batch_df)
                    continue
            
            logger.info(f"Storage complete: {stored_count} stored, "
                        f"{updated_count} updated, {error_count} errors")
            
            return {
                'stored_count': stored_count,
                'updated_count': updated_count,
                'error_count': error_count,
                'total_processed': len(records)
            }
            
        except Exception as e:
            logger.error(f"Database storage failed: {e}")
            raise
    
    def _upsert_batch(self, batch_df: pd.DataFrame) -> tuple[int, int]:
        """
        Perform upsert (insert or update) operation for a batch
        
        Args:
            batch_df: DataFrame with batch data
            
        Returns:
            Tuple of (inserted_count, updated_count)
        """
        with self.engine.connect() as conn:
            # Use PostgreSQL's ON CONFLICT clause for upsert
            upsert_query = text("""
                INSERT INTO historical_prices 
                (ticker, date, open, high, low, close, volume, adjusted_close, vwap)
                VALUES (:ticker, :date, :open, :high, :low, :close, :volume, :adjusted_close, :vwap)
                ON CONFLICT (ticker, date) 
                DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    adjusted_close = EXCLUDED.adjusted_close,
                    vwap = EXCLUDED.vwap
            """)
            
            # Execute batch insert
            conn.execute(upsert_query, batch_df.to_dict('records'))
            conn.commit()
            
            # For simplicity, return the batch size as inserted
            # In a real implementation, you might want to track actual inserts vs updates
            return len(batch_df), 0
    
    def _insert_ignore_batch(self, batch_df: pd.DataFrame) -> int:
        """
        Insert batch with conflict ignore
        
        Args:
            batch_df: DataFrame with batch data
            
        Returns:
            Number of records inserted
        """
        with self.engine.connect() as conn:
            insert_query = text("""
                INSERT INTO historical_prices 
                (ticker, date, open, high, low, close, volume, adjusted_close, vwap)
                VALUES (:ticker, :date, :open, :high, :low, :close, :volume, :adjusted_close, :vwap)
                ON CONFLICT (ticker, date) DO NOTHING
            """)
            
            conn.execute(insert_query, batch_df.to_dict('records'))
            conn.commit()
            
            return len(batch_df)
    
    def _insert_batch(self, batch_df: pd.DataFrame) -> int:
        """
        Insert batch without conflict handling
        
        Args:
            batch_df: DataFrame with batch data
            
        Returns:
            Number of records inserted
        """
        batch_df.to_sql(
            'historical_prices',
            self.engine,
            if_exists='append',
            index=False,
            method='multi'
        )
        
        return len(batch_df)
    
    def get_historical_data(self, ticker: str, start_date: Optional[date] = None,
                            end_date: Optional[date] = None, 
                            limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve historical data from database
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of records to return
            
        Returns:
            List of historical data records
        """
        query = "SELECT * FROM historical_prices WHERE ticker = :ticker"
        params = {'ticker': ticker.upper()}
        
        if start_date:
            query += " AND date >= :start_date"
            params['start_date'] = start_date
        
        if end_date:
            query += " AND date <= :end_date"
            params['end_date'] = end_date
        
        query += " ORDER BY date ASC"
        
        if limit:
            query += " LIMIT :limit"
            params['limit'] = limit
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                records = [dict(row._mapping) for row in result]
                
            logger.info(f"Retrieved {len(records)} records for {ticker}")
            return records
            
        except Exception as e:
            logger.error(f"Error retrieving data for {ticker}: {e}")
            raise
    
    def get_available_tickers(self) -> List[str]:
        """
        Get list of all tickers available in the database
        
        Returns:
            List of ticker symbols
        """
        query = "SELECT DISTINCT ticker FROM historical_prices ORDER BY ticker"
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                tickers = [row[0] for row in result]
                
            logger.info(f"Found {len(tickers)} unique tickers in database")
            return tickers
            
        except Exception as e:
            logger.error(f"Error retrieving available tickers: {e}")
            raise
    
    def get_date_range(self, ticker: Optional[str] = None) -> Dict[str, date]:
        """
        Get the date range of available data
        
        Args:
            ticker: Optional ticker to filter by
            
        Returns:
            Dictionary with min_date and max_date
        """
        query = "SELECT MIN(date) as min_date, MAX(date) as max_date FROM historical_prices"
        params = {}
        
        if ticker:
            query += " WHERE ticker = :ticker"
            params['ticker'] = ticker.upper()
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                row = result.fetchone()
                
                if row and row[0] and row[1]:
                    return {
                        'min_date': row[0],
                        'max_date': row[1]
                    }
                else:
                    return {'min_date': None, 'max_date': None}
                    
        except Exception as e:
            logger.error(f"Error retrieving date range: {e}")
            raise
    
    def get_data_stats(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about stored data
        
        Args:
            ticker: Optional ticker to filter by
            
        Returns:
            Dictionary with data statistics
        """
        base_query = "FROM historical_prices"
        where_clause = ""
        params = {}
        
        if ticker:
            where_clause = " WHERE ticker = :ticker"
            params['ticker'] = ticker.upper()
        
        queries = {
            'total_records': f"SELECT COUNT(*) {base_query}{where_clause}",
            'unique_tickers': f"SELECT COUNT(DISTINCT ticker) {base_query}{where_clause}",
            'date_range': f"SELECT MIN(date), MAX(date) {base_query}{where_clause}",
            'avg_volume': f"SELECT AVG(volume) {base_query}{where_clause}",
            'latest_update': f"SELECT MAX(created_at) {base_query}{where_clause}"
        }
        
        try:
            stats = {}
            
            with self.engine.connect() as conn:
                for stat_name, query in queries.items():
                    result = conn.execute(text(query), params)
                    row = result.fetchone()
                    
                    if stat_name == 'date_range':
                        stats['min_date'] = row[0] if row else None
                        stats['max_date'] = row[1] if row else None
                    else:
                        stats[stat_name] = row[0] if row else 0
            
            logger.info(f"Retrieved data statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error retrieving data statistics: {e}")
            raise
    
    def create_tables(self) -> None:
        """Create database tables if they don't exist"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS historical_prices (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            open DECIMAL(10,4) NOT NULL,
            high DECIMAL(10,4) NOT NULL,
            low DECIMAL(10,4) NOT NULL,
            close DECIMAL(10,4) NOT NULL,
            volume BIGINT NOT NULL,
            adjusted_close DECIMAL(10,4),
            vwap DECIMAL(10,4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, date)
        );
        
        CREATE TABLE IF NOT EXISTS tickers (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10) NOT NULL UNIQUE,
            name VARCHAR(255),
            market VARCHAR(50) DEFAULT 'stocks',
            locale VARCHAR(10) DEFAULT 'us',
            primary_exchange VARCHAR(50),
            currency_name VARCHAR(10),
            active BOOLEAN DEFAULT true,
            type VARCHAR(50),
            market_cap DOUBLE PRECISION,
            weighted_shares_outstanding DOUBLE PRECISION,
            round_lot INTEGER,
            cik VARCHAR(20),
            composite_figi VARCHAR(20),
            share_class_figi VARCHAR(20),
            sic_code VARCHAR(10),
            sic_description VARCHAR(255),
            ticker_root VARCHAR(10),
            total_employees INTEGER,
            list_date DATE,
            last_updated_utc TIMESTAMP,
            is_sp500 BOOLEAN DEFAULT false,
            is_popular BOOLEAN DEFAULT false,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS ticker_cache (
            id SERIAL PRIMARY KEY,
            cache_key VARCHAR(255) NOT NULL UNIQUE,
            cache_data JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL
        );
        
        -- Historical prices indexes
        CREATE INDEX IF NOT EXISTS idx_historical_prices_ticker_date 
        ON historical_prices(ticker, date);
        
        CREATE INDEX IF NOT EXISTS idx_historical_prices_date 
        ON historical_prices(date);
        
        CREATE INDEX IF NOT EXISTS idx_historical_prices_ticker 
        ON historical_prices(ticker);
        
        -- Tickers indexes
        CREATE INDEX IF NOT EXISTS idx_tickers_ticker 
        ON tickers(ticker);
        
        CREATE INDEX IF NOT EXISTS idx_tickers_market 
        ON tickers(market);
        
        CREATE INDEX IF NOT EXISTS idx_tickers_active 
        ON tickers(active);
        
        CREATE INDEX IF NOT EXISTS idx_tickers_sp500 
        ON tickers(is_sp500);
        
        CREATE INDEX IF NOT EXISTS idx_tickers_popular 
        ON tickers(is_popular);
        
        CREATE INDEX IF NOT EXISTS idx_tickers_cik 
        ON tickers(cik);
        
        CREATE INDEX IF NOT EXISTS idx_tickers_composite_figi 
        ON tickers(composite_figi);
        
        CREATE INDEX IF NOT EXISTS idx_tickers_share_class_figi 
        ON tickers(share_class_figi);
        
        CREATE INDEX IF NOT EXISTS idx_tickers_sic_code 
        ON tickers(sic_code);
        
        CREATE INDEX IF NOT EXISTS idx_tickers_list_date 
        ON tickers(list_date);
        
        CREATE INDEX IF NOT EXISTS idx_tickers_last_updated_utc 
        ON tickers(last_updated_utc);
        
        -- Ticker cache indexes
        CREATE INDEX IF NOT EXISTS idx_ticker_cache_key 
        ON ticker_cache(cache_key);
        
        CREATE INDEX IF NOT EXISTS idx_ticker_cache_expires 
        ON ticker_cache(expires_at);
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
                
            logger.info("Database tables created/verified successfully")
            
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the database connection
        
        Returns:
            Dictionary with health check results
        """
        try:
            with self.engine.connect() as conn:
                # Test basic connectivity
                result = conn.execute(text("SELECT 1"))
                connectivity = result.fetchone()[0] == 1
                
                # Test table existence
                table_check = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'historical_prices'
                    )
                """))
                table_exists = table_check.fetchone()[0]
                
                # Get basic stats
                if table_exists:
                    stats_result = conn.execute(text(
                        "SELECT COUNT(*) FROM historical_prices"
                    ))
                    record_count = stats_result.fetchone()[0]
                else:
                    record_count = 0
                
                return {
                    'status': 'healthy',
                    'connectivity': connectivity,
                    'table_exists': table_exists,
                    'record_count': record_count,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def store_tickers(self, tickers_data: List[Dict[str, Any]], 
                        batch_size: int = 1000) -> Dict[str, Any]:
        """
        Store ticker information to database
        
        Args:
            tickers_data: List of ticker dictionaries
            batch_size: Number of records to insert in each batch
            
        Returns:
            Dictionary with storage statistics
        """
        if not tickers_data:
            logger.warning("No ticker data to store")
            return {'stored_count': 0, 'updated_count': 0, 'error_count': 0}
        
        logger.info(f"Storing {len(tickers_data)} tickers to database")
        
        stored_count = 0
        updated_count = 0
        error_count = 0
        
        try:
            # Process in batches
            for i in range(0, len(tickers_data), batch_size):
                batch_data = tickers_data[i:i + batch_size]
                
                try:
                    with self.engine.connect() as conn:
                        for ticker_data in batch_data:
                            
                            # Prepare ticker data with all available fields
                            upsert_data = {
                                'ticker': ticker_data.get('ticker', '').upper(),
                                'name': ticker_data.get('name'),
                                'market': ticker_data.get('market', 'stocks'),
                                'locale': ticker_data.get('locale', 'us'),
                                'primary_exchange': ticker_data.get('primary_exchange'),
                                'currency_name': ticker_data.get('currency_name'),
                                'active': ticker_data.get('active', True),
                                'type': ticker_data.get('type'),
                                'market_cap': ticker_data.get('market_cap'),
                                'weighted_shares_outstanding': ticker_data.get('weighted_shares_outstanding'),
                                'round_lot': ticker_data.get('round_lot'),
                                'cik': ticker_data.get('cik'),
                                'composite_figi': ticker_data.get('composite_figi'),
                                'share_class_figi': ticker_data.get('share_class_figi'),
                                'sic_code': ticker_data.get('sic_code'),
                                'sic_description': ticker_data.get('sic_description'),
                                'ticker_root': ticker_data.get('ticker_root'),
                                'total_employees': ticker_data.get('total_employees'),
                                'list_date': ticker_data.get('list_date'),
                                'last_updated_utc': ticker_data.get('last_updated_utc'),
                                'is_sp500': ticker_data.get('is_sp500', False),
                                'is_popular': ticker_data.get('is_popular', False)
                            }
                            print(upsert_data)
                            # Remove None values
                            upsert_data = {k: v for k, v in upsert_data.items() if v is not None}
                            
                            # Build dynamic query based on available fields
                            columns = list(upsert_data.keys())
                            placeholders = [f":{col}" for col in columns]
                            
                            # Add last_updated to columns and values
                            columns.append('last_updated')
                            placeholders.append('CURRENT_TIMESTAMP')
                            
                            # Build the INSERT part
                            insert_columns = ', '.join(columns)
                            insert_values = ', '.join(placeholders)
                            
                            # Build the UPDATE part (exclude ticker and timestamps)
                            update_columns = [col for col in upsert_data.keys() if col not in ['ticker', 'created_at']]
                            update_set = ', '.join([f"{col} = EXCLUDED.{col}" for col in update_columns])
                            update_set += ', last_updated = CURRENT_TIMESTAMP'
                            
                            upsert_query = text(f"""
                                INSERT INTO tickers ({insert_columns})
                                VALUES ({insert_values})
                                ON CONFLICT (ticker) 
                                DO UPDATE SET {update_set}
                            """)
                            
                            conn.execute(upsert_query, upsert_data)
                            stored_count += 1
                        
                        conn.commit()
                        
                    logger.info(f"Processed ticker batch {i//batch_size + 1}: {len(batch_data)} tickers")
                    
                except Exception as e:
                    logger.error(f"Error processing ticker batch {i//batch_size + 1}: {e}")
                    error_count += len(batch_data)
                    continue
            
            logger.info(f"Ticker storage complete: {stored_count} processed, {error_count} errors")
            
            return {
                'stored_count': stored_count,
                'updated_count': updated_count,
                'error_count': error_count,
                'total_processed': len(tickers_data)
            }
            
        except Exception as e:
            logger.error(f"Ticker storage failed: {e}")
            raise
    
    def get_tickers(self, market: str = 'stocks', active: bool = True, 
                    is_sp500: Optional[bool] = None, is_popular: Optional[bool] = None,
                    limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get tickers from database
        
        Args:
            market: Market type filter
            active: Active status filter
            is_sp500: S&P 500 filter
            is_popular: Popular stocks filter
            limit: Maximum number of results
            
        Returns:
            List of ticker dictionaries
        """
        query = "SELECT * FROM tickers WHERE 1=1"
        params = {}
        
        if market:
            query += " AND market = :market"
            params['market'] = market
        
        if active is not None:
            query += " AND active = :active"
            params['active'] = active
        
        if is_sp500 is not None:
            query += " AND is_sp500 = :is_sp500"
            params['is_sp500'] = is_sp500
        
        if is_popular is not None:
            query += " AND is_popular = :is_popular"
            params['is_popular'] = is_popular
        
        query += " ORDER BY ticker"
        
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                tickers = []
                
                for row in result:
                    ticker_dict = dict(row._mapping)
                    tickers.append(ticker_dict)
                
                logger.info(f"Retrieved {len(tickers)} tickers from database")
                return tickers
                
        except Exception as e:
            logger.error(f"Error retrieving tickers: {e}")
            raise
    
    def get_ticker_symbols(self, market: str = 'stocks', active: bool = True,
                            is_sp500: Optional[bool] = None, is_popular: Optional[bool] = None,
                            limit: Optional[int] = None) -> List[str]:
        """
        Get ticker symbols only (for performance)
        
        Args:
            market: Market type filter
            active: Active status filter
            is_sp500: S&P 500 filter
            is_popular: Popular stocks filter
            limit: Maximum number of results
            
        Returns:
            List of ticker symbols
        """
        query = "SELECT ticker FROM tickers WHERE 1=1"
        params = {}
        
        if market:
            query += " AND market = :market"
            params['market'] = market
        
        if active is not None:
            query += " AND active = :active"
            params['active'] = active
        
        if is_sp500 is not None:
            query += " AND is_sp500 = :is_sp500"
            params['is_sp500'] = is_sp500
        
        if is_popular is not None:
            query += " AND is_popular = :is_popular"
            params['is_popular'] = is_popular

        # Add type filter for common stock
        query += " AND type = 'CS' AND market_cap is null" 
        
        query += " ORDER BY ticker"
        
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                tickers = [row[0] for row in result]
                
                logger.info(f"Retrieved {len(tickers)} ticker symbols from database")
                return tickers
                
        except Exception as e:
            logger.error(f"Error retrieving ticker symbols: {e}")
            raise
    
    def store_cache(self, cache_key: str, data: Any, expires_hours: int = 24) -> None:
        """
        Store cache data in database
        
        Args:
            cache_key: Unique cache key
            data: Data to cache (will be JSON serialized)
            expires_hours: Hours until cache expires
        """
        try:
            import json
            from datetime import timedelta
            
            expires_at = datetime.now() + timedelta(hours=expires_hours)
            
            with self.engine.connect() as conn:
                upsert_query = text("""
                    INSERT INTO ticker_cache (cache_key, cache_data, expires_at)
                    VALUES (:cache_key, :cache_data, :expires_at)
                    ON CONFLICT (cache_key)
                    DO UPDATE SET
                        cache_data = EXCLUDED.cache_data,
                        expires_at = EXCLUDED.expires_at,
                        created_at = CURRENT_TIMESTAMP
                """)
                
                conn.execute(upsert_query, {
                    'cache_key': cache_key,
                    'cache_data': json.dumps(data),
                    'expires_at': expires_at
                })
                conn.commit()
                
            logger.info(f"Stored cache for key: {cache_key}")
            
        except Exception as e:
            logger.error(f"Error storing cache: {e}")
            raise
    
    def get_cache(self, cache_key: str) -> Optional[Any]:
        """
        Get cache data from database
        
        Args:
            cache_key: Cache key to retrieve
            
        Returns:
            Cached data or None if not found/expired
        """
        try:
            import json
            
            with self.engine.connect() as conn:
                query = text("""
                    SELECT cache_data FROM ticker_cache 
                    WHERE cache_key = :cache_key AND expires_at > CURRENT_TIMESTAMP
                """)
                
                result = conn.execute(query, {'cache_key': cache_key})
                row = result.fetchone()
                
                if row:
                    return json.loads(row[0])
                
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving cache: {e}")
            return None
    
    def clear_expired_cache(self) -> int:
        """
        Clear expired cache entries
        
        Returns:
            Number of entries cleared
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    DELETE FROM ticker_cache WHERE expires_at <= CURRENT_TIMESTAMP
                """))
                cleared_count = result.rowcount
                conn.commit()
                
            logger.info(f"Cleared {cleared_count} expired cache entries")
            return cleared_count
            
        except Exception as e:
            logger.error(f"Error clearing expired cache: {e}")
            raise

    def __exit__(self):
        """Context manager exit"""
        if hasattr(self, 'engine'):
            self.engine.dispose() 
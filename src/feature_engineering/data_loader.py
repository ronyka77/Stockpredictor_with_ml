"""
Stock Data Loader

This module provides functionality to load and validate stock data
from the database for feature engineering calculations.
"""

import pandas as pd
from typing import List, Optional, Dict, Any, Union
from datetime import date
import os
from sqlalchemy import create_engine, text

from src.utils.logger import get_logger
from src.feature_engineering.config import config

logger = get_logger(__name__, utility='feature_engineering')

class StockDataLoader:
    """
    Loads and validates stock data from the database for feature engineering
    """
    
    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data loader
        
        Args:
            db_config: Database configuration dictionary
        """
        if db_config:
            self.config = db_config
        else:
            # Default configuration from environment variables
            self.config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', 5432)),
                'database': os.getenv('DB_NAME', 'stock_data'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', '')
            }
        # Fail-fast validation for required credentials
        if not self.config.get('password'):
            raise ValueError("DB_PASSWORD environment variable is required for feature engineering data access")
        
        # Create SQLAlchemy engine for pandas compatibility
        connection_string = f"postgresql://{self.config['user']}:{self.config['password']}@{self.config['host']}:{self.config['port']}/{self.config['database']}"
        self.engine = create_engine(connection_string)
        self.feature_config = config
        
        logger.info(f"Initialized StockDataLoader with database: {self.config['host']}:{self.config['port']}/{self.config['database']}")
    
    def load_stock_data(self, ticker: str, start_date: Union[str, date], 
                        end_date: Union[str, date]) -> pd.DataFrame:
        """
        Load stock data for a single ticker
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data loading
            end_date: End date for data loading
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Loading data for {ticker} from {start_date} to {end_date}")
        
        try:
            # Convert dates to strings if needed
            if isinstance(start_date, date):
                start_date = start_date.strftime('%Y-%m-%d')
            if isinstance(end_date, date):
                end_date = end_date.strftime('%Y-%m-%d')
            
            # SQL query for historical_prices table
            query = text("""
            SELECT date, "open", high, low, "close", volume, adjusted_close, vwap
            FROM historical_prices 
            WHERE ticker = :ticker
                AND date >= :start_date AND date <= :end_date 
            ORDER BY date ASC
            """)
            
            params = {
                'ticker': ticker.upper(),
                'start_date': start_date,
                'end_date': end_date
            }
            
            # Use SQLAlchemy engine with pandas
            df = pd.read_sql_query(
                query, 
                self.engine, 
                params=params,
                parse_dates=['date']
            )
            
            if df.empty:
                logger.warning(f"No data found for {ticker} between {start_date} and {end_date}")
                return df
            
            # Set date as index
            df.set_index('date', inplace=True)
            
            # Convert numeric columns to float (handle PostgreSQL numeric type)
            numeric_columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'vwap']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert volume to int64
            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('int64')
            
            # Validate and clean the data
            df = self._validate_and_clean_data(df, ticker)
            
            logger.info(f"Loaded {len(df)} records for {ticker}")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {str(e)}")
            raise
    
    def get_available_tickers(self, min_data_points: Optional[int] = None, 
                                active_only: bool = True, market: str = 'stocks',
                                ) -> List[str]:
        """
        Get list of available tickers with sufficient data
        
        Args:
            min_data_points: Minimum number of data points required
            active_only: Only include active tickers
            market: Market type filter (default: 'stocks')
            
        Returns:
            List of ticker symbols
        """
        if min_data_points is None:
            min_data_points = self.feature_config.data_quality.MIN_DATA_POINTS
        
        logger.info(f"Getting available tickers with at least {min_data_points} data points")
        
        try:
            # Join tickers table with historical_prices to get active tickers with sufficient data
            query = text("""
            SELECT t.ticker, COUNT(hp.*) as data_points, t.name, t.market
            FROM tickers t
            INNER JOIN historical_prices hp ON t.ticker = hp.ticker
            WHERE (:active_only = false OR t.active = true)
                AND (:market = 'all' OR t.market = :market)
                AND t."type" ='CS'
            GROUP BY t.ticker, t.name, t.market
            HAVING COUNT(hp.*) >= :min_data_points
            ORDER BY COUNT(hp.*) DESC, t.ticker
            """)
            
            params = {
                'min_data_points': min_data_points,
                'active_only': active_only,
                'market': market if market != 'all' else 'all',
            }
            
            df = pd.read_sql_query(query, self.engine, params=params)
            
            tickers = df['ticker'].tolist()
            logger.info(f"Found {len(tickers)} tickers with at least {min_data_points} data points")
            
            return tickers
            
        except Exception as e:
            logger.error(f"Error getting available tickers: {str(e)}")
            raise

    def get_ticker_metadata(self, ticker: Optional[str] = None) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Get metadata for a specific ticker or all tickers from the tickers table
        
        Args:
            ticker: Stock ticker symbol (None for all tickers)
            
        Returns:
            Dictionary with ticker metadata (if ticker specified) or DataFrame with all tickers metadata
        """
        try:
            # Build base query
            query = text("""
            SELECT id, ticker, "name", market, locale, primary_exchange, currency_name, 
                active, "type", market_cap, weighted_shares_outstanding, round_lot, 
                last_updated, created_at, cik, composite_figi, 
                share_class_figi, sic_code, sic_description, ticker_root, total_employees, 
                list_date
            FROM tickers 
            """ + ("WHERE ticker = :ticker" if ticker is not None else "") + """
            ORDER BY ticker
            """)
            
            # Execute query with or without parameters
            if ticker is not None:
                df = pd.read_sql_query(query, self.engine, params={'ticker': ticker.upper()})
            else:
                df = pd.read_sql_query(query, self.engine)
            
            if df.empty:
                if ticker is None:
                    logger.warning("No ticker metadata found")
                    return pd.DataFrame()
                else:
                    logger.warning(f"No metadata found for ticker {ticker}")
                    return {}
            
            if ticker is None:
                logger.info(f"Retrieved metadata for {len(df)} tickers")
                return df
            else:
                return df.iloc[0].to_dict()
            
        except Exception as e:
            logger.error(f"Error getting metadata for {'all tickers' if ticker is None else ticker}: {str(e)}")
            return pd.DataFrame() if ticker is None else {}

    def get_data_summary(self, ticker: str) -> Dict[str, Any]:
        """
        Get data summary for a ticker including metadata and price data statistics
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with data summary
        """
        logger.info(f"Getting data summary for {ticker}")
        
        try:
            # Get ticker metadata
            metadata = self.get_ticker_metadata(ticker)
            
            # Get price data statistics
            query = text("""
            SELECT 
                COUNT(*) as total_records,
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                AVG("close") as avg_close,
                MIN("close") as min_close,
                MAX("close") as max_close,
                AVG(volume) as avg_volume,
                MIN(volume) as min_volume,
                MAX(volume) as max_volume,
                COUNT(CASE WHEN adjusted_close IS NULL THEN 1 END) as missing_adj_close,
                COUNT(CASE WHEN vwap IS NULL THEN 1 END) as missing_vwap
            FROM historical_prices 
            WHERE ticker = :ticker
            """)
            
            df = pd.read_sql_query(query, self.engine, params={'ticker': ticker.upper()})
            
            if df.empty:
                logger.warning(f"No price data found for {ticker}")
                return metadata
            
            price_stats = df.iloc[0].to_dict()
            
            # Combine metadata and price statistics
            summary = {
                'ticker': ticker.upper(),
                'metadata': metadata,
                'price_data': price_stats,
                'data_quality': {
                    'has_metadata': bool(metadata),
                    'total_records': price_stats.get('total_records', 0),
                    'missing_adj_close': price_stats.get('missing_adj_close', 0),
                    'missing_vwap': price_stats.get('missing_vwap', 0),
                    'data_completeness': 1.0 - (price_stats.get('missing_adj_close', 0) + price_stats.get('missing_vwap', 0)) / (2 * price_stats.get('total_records', 1))
                }
            }
            
            logger.info(f"Data summary for {ticker}: {price_stats.get('total_records', 0)} records")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting data summary for {ticker}: {str(e)}")
            return {'ticker': ticker, 'error': str(e)}
    
    def _validate_and_clean_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Validate and clean the loaded data
        
        Args:
            df: Raw DataFrame from database
            ticker: Ticker symbol for logging
            
        Returns:
            Cleaned DataFrame
        """
        original_length = len(df)
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns for {ticker}: {missing_columns}")
        
        # Remove rows with missing OHLCV data
        df = df.dropna(subset=required_columns)
        
        # Ensure all price columns are numeric
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                # Convert to numeric, replacing any non-numeric values with NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows where price conversion failed
        df = df.dropna(subset=price_columns)
        
        # Validate OHLC relationships and positive values
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close']) |
            (df['open'] <= 0) |
            (df['high'] <= 0) |
            (df['low'] <= 0) |
            (df['close'] <= 0) |
            (df['volume'] < 0)
        )
        
        if invalid_ohlc.any():
            invalid_count = invalid_ohlc.sum()
            logger.warning(f"Removing {invalid_count} rows with invalid OHLC data for {ticker}")
            df = df[~invalid_ohlc]
        
        # Check for extreme outliers (more than 50% price change in one day)
        if len(df) > 1:
            price_changes = df['close'].pct_change().abs()
            extreme_changes = price_changes > 0.5
            
            if extreme_changes.any():
                extreme_count = extreme_changes.sum()
                logger.warning(f"Found {extreme_count} extreme price changes for {ticker}")
        
        # Sort by date to ensure chronological order
        df = df.sort_index()
        
        # Check for data continuity
        if len(df) > 1:
            date_gaps = df.index.to_series().diff().dt.days
            large_gaps = date_gaps > 7  # More than 7 days gap
            
            if large_gaps.any():
                gap_count = large_gaps.sum()
                logger.warning(f"Found {gap_count} large date gaps (>7 days) for {ticker}")
        
        cleaned_length = len(df)
        removed_count = original_length - cleaned_length
        
        if removed_count > 0:
            logger.info(f"Cleaned data for {ticker}: removed {removed_count} invalid rows")
        
        # Check if we still have enough data
        if len(df) < self.feature_config.data_quality.MIN_DATA_POINTS:
            logger.warning(f"Insufficient data for {ticker} after cleaning: {len(df)} < {self.feature_config.data_quality.MIN_DATA_POINTS}")
        
        return df
    
    def execute_query(self, query: str, params: Optional[List[Any]] = None) -> List[tuple]:
        """
        Execute a raw SQL query and return results
        
        Args:
            query: SQL query string (with %s placeholders for parameters)
            params: List of parameters for the query
            
        Returns:
            List of tuples containing query results
        """
        try:
            # Convert %s placeholders to :param format for SQLAlchemy
            param_count = query.count('%s')
            if param_count > 0 and params:
                # Create named parameters
                param_dict = {f'param_{i}': params[i] for i in range(min(param_count, len(params)))}
                
                # Replace %s with :param_n
                formatted_query = query
                for i in range(param_count):
                    formatted_query = formatted_query.replace('%s', f':param_{i}', 1)
                
                # Execute with named parameters
                with self.engine.connect() as conn:
                    result = conn.execute(text(formatted_query), param_dict)
                    return [tuple(row) for row in result]
            else:
                # Execute without parameters
                with self.engine.connect() as conn:
                    result = conn.execute(text(query))
                    return [tuple(row) for row in result]
                    
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.info(f"Query: {query}")
            logger.info(f"Params: {params}")
            raise

    def close(self) -> None:
        """Close database connection"""
        if hasattr(self, 'engine') and self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close() 
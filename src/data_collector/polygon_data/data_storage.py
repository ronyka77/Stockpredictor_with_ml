"""
Data storage functionality for PostgreSQL database operations
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, date
import pandas as pd

from src.utils.logger import get_logger
from src.data_collector.polygon_data.data_validator import OHLCVRecord
from src.database.connection import (
    init_global_pool,
    fetch_all,
    fetch_one,
    execute,
    execute_values,
)

logger = get_logger(__name__, utility="data_collector")


class DataStorage:
    """
    Handles storage and retrieval of stock market data in PostgreSQL database

    Provides efficient bulk operations, data integrity checks, and query capabilities.
    """

    def __init__(self):
        """
        Initialize the data storage handler

        Args:
        """
        init_global_pool()

    def store_historical_data(
        self,
        records: List[OHLCVRecord],
        batch_size: int = 5000,
    ) -> Dict[str, Any]:
        """
        Store validated OHLCV data to PostgreSQL database

        Args:
            records: List of validated OHLCV records
            batch_size: Number of records to insert in each batch

        Returns:
            Dictionary with storage statistics
        """
        if not records:
            logger.warning("No records to store")
            return {"stored_count": 0, "error_count": 0, "updated_count": 0}

        logger.info(f"Storing {len(records)} records to database")

        # Convert records to database format
        data_rows = []
        for record in records:
            row_data = record.to_dict()
            data_rows.append(row_data)

        # Create DataFrame for efficient processing
        df = pd.DataFrame(data_rows).drop_duplicates(subset=["ticker", "date"])
        # print(df.head())

        # Ensure proper data types
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["volume"] = df["volume"].astype("int64")

        stored_count = 0
        updated_count = 0
        error_count = 0

        try:
            # Process in batches
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i : i + batch_size]

                try:
                    batch_stored, batch_updated = self._upsert_batch(batch_df)
                    stored_count += batch_stored
                    updated_count += batch_updated
                    logger.info(
                        f"Processed batch {i // batch_size + 1}: "
                        f"{len(batch_df)} records"
                    )
                except Exception as e:
                    logger.error(f"Error processing batch {i // batch_size + 1}: {e}")
                    error_count += len(batch_df)
                    continue

            logger.info(
                f"Storage complete: {stored_count} stored, "
                f"{updated_count} updated, {error_count} errors"
            )

            return {
                "stored_count": stored_count,
                "updated_count": updated_count,
                "error_count": error_count,
                "total_processed": len(records),
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
        # Use execute_values for performant batched upserts
        cols = [
            "ticker",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adjusted_close",
            "vwap",
        ]
        values = [tuple(row[col] for col in cols) for _, row in batch_df.iterrows()]

        insert_sql = (
            "INSERT INTO historical_prices (" + ",".join(cols) + ") VALUES %s "
            "ON CONFLICT (ticker, date) DO UPDATE SET "
            "open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, "
            "close = EXCLUDED.close, volume = EXCLUDED.volume, "
            "adjusted_close = EXCLUDED.adjusted_close, vwap = EXCLUDED.vwap"
        )

        try:
            # Use helper that manages connection, cursor and commit
            execute_values(insert_sql, values, page_size=1000)
            return len(batch_df), 0
        except Exception as e:
            logger.error(f"_upsert_batch failed: {e}")
            raise

    def _insert_ignore_batch(self, batch_df: pd.DataFrame) -> int:
        """
        Insert batch with conflict ignore

        Args:
            batch_df: DataFrame with batch data

        Returns:
            Number of records inserted
        """
        cols = [
            "ticker",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adjusted_close",
            "vwap",
        ]
        values = [tuple(row[col] for col in cols) for _, row in batch_df.iterrows()]
        insert_sql = (
            "INSERT INTO historical_prices (" + ",".join(cols) + ") VALUES %s "
            "ON CONFLICT (ticker, date) DO NOTHING"
        )

        try:
            execute_values(insert_sql, values, page_size=1000)
            return len(batch_df)
        except Exception as e:
            logger.error(f"_insert_ignore_batch failed: {e}")
            raise

    def _insert_batch(self, batch_df: pd.DataFrame) -> int:
        """
        Insert batch without conflict handling

        Args:
            batch_df: DataFrame with batch data

        Returns:
            Number of records inserted
        """
        # Generic insert using execute_values
        cols = list(batch_df.columns)
        values = [tuple(row[col] for col in cols) for _, row in batch_df.iterrows()]
        insert_sql = "INSERT INTO historical_prices (" + ",".join(cols) + ") VALUES %s"

        try:
            execute_values(insert_sql, values, page_size=1000)
            return len(batch_df)
        except Exception as e:
            logger.error(f"_insert_batch failed: {e}")
            raise

    def get_historical_data(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
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
        query = "SELECT * FROM historical_prices WHERE ticker = %s"
        params: list[Any] = [ticker.upper()]

        if start_date:
            query += " AND date >= %s"
            params.append(start_date)

        if end_date:
            query += " AND date <= %s"
            params.append(end_date)

        query += " ORDER BY date ASC"

        if limit:
            query += " LIMIT %s"
            params.append(limit)

        try:
            rows = fetch_all(query, tuple(params))
            # fetch_all returns list of dict rows by default
            records = rows or []
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
            rows = fetch_all(query, None, dict_cursor=False)
            tickers = [r[0] for r in (rows or [])]
            logger.info(f"Found {len(tickers)} unique tickers in database")
            return tickers
        except Exception as e:
            logger.error(f"Error retrieving available tickers: {e}")
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
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
        """

        try:
            execute(create_table_sql)

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
            connectivity = bool(fetch_one("SELECT 1") is not None)
            table_exists = bool(
                fetch_one(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'historical_prices')"
                )
            )
            record_count = 0
            if table_exists:
                rc = fetch_one("SELECT COUNT(*) as cnt FROM historical_prices")
                record_count = (
                    rc.get("cnt") if isinstance(rc, dict) else (rc[0] if rc else 0)
                )

            return {
                "status": "healthy",
                "connectivity": connectivity,
                "table_exists": table_exists,
                "record_count": record_count,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def __enter__(self):
        """Context manager entry"""
        return self

    def store_tickers(
        self, tickers_data: List[Dict[str, Any]], batch_size: int = 1000
    ) -> Dict[str, Any]:
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
            return {"stored_count": 0, "updated_count": 0, "error_count": 0}

        logger.info(f"Storing {len(tickers_data)} tickers to database")

        stored_count = 0
        updated_count = 0
        error_count = 0

        try:
            # Process in batches
            for i in range(0, len(tickers_data), batch_size):
                batch_data = tickers_data[i : i + batch_size]

                try:
                    # Prepare rows for execute_values
                    cols = None
                    rows = []
                    for ticker_data in batch_data:
                        upsert_data = {
                            "ticker": ticker_data.get("ticker", "").upper(),
                            "name": ticker_data.get("name"),
                            "market": ticker_data.get("market", "stocks"),
                            "locale": ticker_data.get("locale", "us"),
                            "primary_exchange": ticker_data.get("primary_exchange"),
                            "currency_name": ticker_data.get("currency_name"),
                            "active": ticker_data.get("active", True),
                            "type": ticker_data.get("type"),
                            "market_cap": ticker_data.get("market_cap"),
                            "weighted_shares_outstanding": ticker_data.get(
                                "weighted_shares_outstanding"
                            ),
                            "round_lot": ticker_data.get("round_lot"),
                            "cik": ticker_data.get("cik"),
                            "composite_figi": ticker_data.get("composite_figi"),
                            "share_class_figi": ticker_data.get("share_class_figi"),
                            "sic_code": ticker_data.get("sic_code"),
                            "sic_description": ticker_data.get("sic_description"),
                            "ticker_root": ticker_data.get("ticker_root"),
                            "total_employees": ticker_data.get("total_employees"),
                            "list_date": ticker_data.get("list_date"),
                        }
                        # Remove None values
                        upsert_data = {
                            k: v for k, v in upsert_data.items() if v is not None
                        }

                        if cols is None:
                            cols = list(upsert_data.keys())
                        # maintain column order for rows
                        rows.append(tuple(upsert_data.get(c) for c in cols))

                    if not cols:
                        continue

                    # Build insert SQL
                    insert_columns = ",".join(cols)
                    insert_sql = (
                        f"INSERT INTO tickers ({insert_columns}) VALUES %s "
                        f"ON CONFLICT (ticker) DO UPDATE SET "
                        + ", ".join(
                            [f"{c}=EXCLUDED.{c}" for c in cols if c != "ticker"]
                        )
                    )

                    # use centralized helper for batched upsert
                    execute_values(insert_sql, rows, page_size=500)
                    stored_count += len(batch_data)

                    logger.info(
                        f"Processed ticker batch {i // batch_size + 1}: {len(batch_data)} tickers"
                    )

                except Exception as e:
                    logger.error(
                        f"Error processing ticker batch {i // batch_size + 1}: {e}"
                    )
                    error_count += len(batch_data)
                    continue

            logger.info(
                f"Ticker storage complete: {stored_count} processed, {error_count} errors"
            )

            return {
                "stored_count": stored_count,
                "updated_count": updated_count,
                "error_count": error_count,
                "total_processed": len(tickers_data),
            }

        except Exception as e:
            logger.error(f"Ticker storage failed: {e}")
            raise

    def get_tickers(self, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get tickers from database"""

        query = "SELECT * FROM tickers WHERE 1=1"
        query += " AND active = true"
        if ticker:
            query += f" AND ticker = '{ticker}'"
        query += " ORDER BY ticker"

        try:
            result = fetch_all(query, None, dict_cursor=True)
            tickers = []

            for row in result:
                tickers.append(row)

            logger.info(f"Retrieved {len(tickers)} tickers from database")
            return tickers

        except Exception as e:
            logger.error(f"Error retrieving tickers: {e}")
            raise

    def __exit__(self):
        """Context manager exit"""
        pass

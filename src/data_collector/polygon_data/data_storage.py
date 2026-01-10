"""Data storage functionality for PostgreSQL database operations."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Callable, Dict, Iterator, List, Optional

import pandas as pd
import polars as pl

from src.utils.core.logger import get_logger
from src.data_collector.polygon_data.data_validator import OHLCVRecord
from cytoolz import valfilter
from src.database.db_utils import DBBatch
from src.database.connection import init_global_pool, fetch_all, fetch_one, execute, execute_values

logger = get_logger(__name__)


OHLCV_POLARS_SCHEMA: dict[str, pl.DataType] = {
    "ticker": pl.Utf8,
    "date": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "vwap": pl.Float64,
}
OHLCV_COLUMNS: tuple[str, ...] = tuple(OHLCV_POLARS_SCHEMA)
OHLCV_UNIQUE_SUBSET: tuple[str, ...] = ("ticker", "date")


class DataStorage:
    """
    Handles storage and retrieval of stock market data in PostgreSQL database

    Provides efficient bulk operations, data integrity checks, and query capabilities.
    """

    _DIVIDEND_COLUMNS = (
        "ex_dividend_date",
        "cash_amount",
        "pay_date",
        "record_date",
        "id",
        "currency",
    )

    def __init__(self, execute_values_fn: Optional[Callable[[str, list[tuple], int], None]] = None):
        """Initialize the data storage handler.

        Args:
            execute_values_fn: Optional injected bulk-executor for tests/smoke.
                When provided, DB pool initialization is skipped.
        """
        self._execute_values_fn = execute_values_fn or execute_values
        if execute_values_fn is None:
            init_global_pool()

    def _execute_values(self, sql: str, rows: list[tuple], page_size: int = 1000) -> None:
        """Execute a batched insert/upsert via injected executor or default helper."""
        self._execute_values_fn(sql, rows, page_size)

    @staticmethod
    def _append_date_filters(
        query: str,
        params: list[Any],
        field: str,
        start_date: Optional[date],
        end_date: Optional[date],
    ) -> str:
        if start_date:
            query += f" AND {field} >= %s"
            params.append(start_date)
        if end_date:
            query += f" AND {field} <= %s"
            params.append(end_date)
        return query

    @staticmethod
    def _coerce_dividend_df(df: pd.DataFrame) -> None:
        df["ex_dividend_date"] = pd.to_datetime(df["ex_dividend_date"]).dt.date
        df["cash_amount"] = df["cash_amount"].astype(float)
        if "pay_date" in df.columns and df["pay_date"].notna().any():
            df["pay_date"] = pd.to_datetime(df["pay_date"]).dt.date
        if "record_date" in df.columns and df["record_date"].notna().any():
            df["record_date"] = pd.to_datetime(df["record_date"]).dt.date

    @staticmethod
    def _coerce_historical_chunk(df: pd.DataFrame) -> None:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date

    @staticmethod
    def _yield_sql_chunks(
        query: str,
        params: tuple[Any, ...],
        chunk_size: int,
        postprocess: Callable[[pd.DataFrame], None],
    ) -> Iterator[pd.DataFrame]:
        from src.database.connection import get_connection

        with get_connection() as conn:
            for chunk in pd.read_sql_query(query, conn, params=params, chunksize=chunk_size):
                postprocess(chunk)
                yield chunk

    def _ohlcv_records_to_polars(self, records: List[OHLCVRecord]) -> pl.DataFrame:
        """Convert validated OHLCV records to a Polars DataFrame for DB upsert.

        Preconditions:
            - records is non-empty
            - each record is a validated OHLCVRecord

        Postconditions:
            - output is deduplicated on (ticker, date)
            - date is pl.Date
            - volume is pl.Int64
        """
        if not records:
            raise ValueError("records must be non-empty")

        data_rows: list[dict[str, Any]] = []
        for record in records:
            row_data = record.to_dict()
            # OHLCVRecord.to_dict() is based on a model_dump() that serializes
            # timestamp/date values into ISO strings. Normalize back to Python
            # date objects before building a typed Polars frame.
            raw_date = row_data.get("date")
            if isinstance(raw_date, str):
                try:
                    if "T" in raw_date:
                        row_data["date"] = datetime.fromisoformat(
                            raw_date.replace("Z", "+00:00")
                        ).date()
                    else:
                        row_data["date"] = date.fromisoformat(raw_date)
                except Exception as exc:
                    raise ValueError(f"Invalid date value for OHLCV record: {raw_date}") from exc
            data_rows.append(row_data)

        df = pl.DataFrame(data_rows, schema=OHLCV_POLARS_SCHEMA)

        # Normalize types and dedupe on conflict key
        df = df.with_columns(
            [
                pl.col("date").cast(pl.Date, strict=False),
                pl.col("volume").cast(pl.Int64, strict=False),
            ]
        ).unique(subset=list(OHLCV_UNIQUE_SUBSET), keep="first", maintain_order=True)

        # Ensure stable column set/order for downstream tuple materialization
        return df.select(list(OHLCV_COLUMNS))

    def _polars_to_db_batch(self, df: pl.DataFrame, columns: tuple[str, ...]) -> DBBatch:
        """Materialize DB tuples from a Polars DataFrame without pandas/iterrows."""
        rows = df.select(list(columns)).rows()
        return DBBatch(columns=columns, rows=rows)

    def store_historical_data(
        self, records: List[OHLCVRecord], batch_size: int = 5000
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

        df = self._ohlcv_records_to_polars(records)

        stored_count = 0
        updated_count = 0
        error_count = 0

        try:
            # Process in batches
            for i in range(0, df.height, batch_size):
                batch_df = df.slice(i, batch_size)
                batch = self._polars_to_db_batch(batch_df, OHLCV_COLUMNS)

                try:
                    batch_stored, batch_updated = self._upsert_batch(batch)
                    stored_count += batch_stored
                    updated_count += batch_updated
                    logger.info(f"Processed batch {i // batch_size + 1}: {len(batch.rows)} records")
                except Exception as e:
                    logger.error(f"Error processing batch {i // batch_size + 1}: {e}")
                    error_count += len(batch.rows)
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
                "total_deduped": df.height,
            }

        except Exception as e:
            logger.error(f"Database storage failed: {e}")
            raise

    def _upsert_batch(self, batch: DBBatch) -> tuple[int, int]:
        """Perform upsert (insert or update) operation for a batch.

        Args:
            batch: DBBatch containing DB column order and row tuples.

        Returns:
            Tuple of (inserted_count, updated_count)
        """
        cols = list(batch.columns)
        values = batch.rows

        insert_sql = (
            "INSERT INTO historical_prices (" + ",".join(cols) + ") VALUES %s "
            "ON CONFLICT (ticker, date) DO UPDATE SET "
            "open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, "
            "close = EXCLUDED.close, volume = EXCLUDED.volume, "
            "vwap = EXCLUDED.vwap"
        )

        try:
            # Use helper that manages connection, cursor and commit
            self._execute_values(insert_sql, values, page_size=1000)
            return len(values), 0
        except Exception as e:
            logger.error(f"_upsert_batch failed: {e}")
            raise

    def _insert_ignore_batch(self, batch: DBBatch) -> int:
        """
        Insert batch with conflict ignore

        Args:
            batch: DBBatch containing DB column order and row tuples.

        Returns:
            Number of records inserted
        """
        cols = list(batch.columns)
        values = batch.rows
        insert_sql = (
            "INSERT INTO historical_prices (" + ",".join(cols) + ") VALUES %s "
            "ON CONFLICT (ticker, date) DO NOTHING"
        )

        try:
            self._execute_values(insert_sql, values, page_size=1000)
            return len(values)
        except Exception as e:
            logger.error(f"_insert_ignore_batch failed: {e}")
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
        query = self._append_date_filters(query, params, "date", start_date, end_date)
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
                record_count = rc.get("cnt") if isinstance(rc, dict) else (rc[0] if rc else 0)

            return {
                "status": "healthy",
                "connectivity": connectivity,
                "table_exists": table_exists,
                "record_count": record_count,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {"status": "unhealthy", "error": str(e), "timestamp": datetime.now().isoformat()}

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
                        upsert_data = dict(valfilter(lambda v: v is not None, upsert_data))

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
                        + ", ".join([f"{c}=EXCLUDED.{c}" for c in cols if c != "ticker"])
                    )

                    # use centralized helper for batched upsert
                    execute_values(insert_sql, rows, page_size=500)
                    stored_count += len(batch_data)

                    logger.info(
                        f"Processed ticker batch {i // batch_size + 1}: {len(batch_data)} tickers"
                    )

                except Exception as e:
                    logger.error(f"Error processing ticker batch {i // batch_size + 1}: {e}")
                    error_count += len(batch_data)
                    continue

            logger.info(f"Ticker storage complete: {stored_count} processed, {error_count} errors")

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

        query = "SELECT * FROM tickers WHERE active = true"
        params = []

        if ticker:
            query += " AND ticker = %s"
            params.append(ticker)

        query += " ORDER BY ticker"

        try:
            result = fetch_all(query, tuple(params) if params else None, dict_cursor=True)
            tickers = list(result)

            logger.info(f"Retrieved {len(tickers)} tickers from database")
            return tickers

        except Exception as e:
            logger.error(f"Error retrieving tickers: {e}")
            raise

    def load_dividends_for_ticker(
        self, ticker: str, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Load dividend data for a specific ticker within a date range.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date filter (ex_dividend_date >= start_date)
            end_date: End date filter (ex_dividend_date <= end_date)

        Returns:
            DataFrame with columns: ex_dividend_date, cash_amount, pay_date, record_date, id, currency
        """
        query = """
            SELECT
                d.ex_dividend_date,
                d.cash_amount,
                d.pay_date,
                d.record_date,
                d.id,
                d.currency
            FROM dividends d
            JOIN tickers t ON d.ticker_id = t.id
            WHERE t.ticker = %s
        """
        params = [ticker.upper()]
        query = self._append_date_filters(query, params, "d.ex_dividend_date", start_date, end_date)
        query += " ORDER BY d.ex_dividend_date ASC"

        try:
            rows = fetch_all(query, tuple(params), dict_cursor=True)
            if not rows:
                # logger.info(f"No dividend data found for {ticker}")
                return pd.DataFrame(columns=list(self._DIVIDEND_COLUMNS))

            df = pd.DataFrame(rows)
            self._coerce_dividend_df(df)

            return df

        except Exception as e:
            logger.error(f"Error loading dividend data for {ticker}: {e}")
            raise

    def load_dividends_for_ticker_chunked(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        chunk_size: int = 10000,
    ) -> Iterator[pd.DataFrame]:
        """
        Load dividend data for a specific ticker in chunks using pandas chunking

        This method loads large dividend datasets in memory-efficient chunks,
        suitable for processing historical dividend data for tickers with many records.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date filter (ex_dividend_date >= start_date)
            end_date: End date filter (ex_dividend_date <= end_date)
            chunk_size: Number of records per chunk

        Yields:
            DataFrame chunks with dividend data
        """
        query = """
            SELECT
                d.ex_dividend_date,
                d.cash_amount,
                d.pay_date,
                d.record_date,
                d.id,
                d.currency
            FROM dividends d
            JOIN tickers t ON d.ticker_id = t.id
            WHERE t.ticker = %s
        """
        params = [ticker.upper()]
        query = self._append_date_filters(query, params, "d.ex_dividend_date", start_date, end_date)
        query += " ORDER BY d.ex_dividend_date ASC"

        try:
            for chunk in self._yield_sql_chunks(
                query=query,
                params=tuple(params),
                chunk_size=chunk_size,
                postprocess=self._coerce_dividend_df,
            ):
                logger.debug(f"Yielded dividend chunk with {len(chunk)} records for {ticker}")
                yield chunk

        except Exception as e:
            logger.error(f"Error loading chunked dividend data for {ticker}: {e}")
            raise

    def get_historical_data_chunked(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        chunk_size: int = 50000,
    ) -> Iterator[pd.DataFrame]:
        """
        Retrieve historical data in chunks for memory-efficient processing

        This method loads large historical datasets in chunks to avoid loading
        entire datasets into memory at once.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date filter
            end_date: End date filter
            chunk_size: Number of records per chunk

        Yields:
            DataFrame chunks with historical data
        """
        query = "SELECT * FROM historical_prices WHERE ticker = %s"
        params: List[Any] = [ticker.upper()]
        query = self._append_date_filters(query, params, "date", start_date, end_date)
        query += " ORDER BY date ASC"

        try:
            for chunk in self._yield_sql_chunks(
                query=query,
                params=tuple(params),
                chunk_size=chunk_size,
                postprocess=self._coerce_historical_chunk,
            ):
                logger.debug(
                    f"Yielded historical data chunk with {len(chunk)} records for {ticker}"
                )
                yield chunk

        except Exception as e:
            logger.error(f"Error retrieving chunked data for {ticker}: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        pass

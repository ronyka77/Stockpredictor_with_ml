"""
Stock Data Loader

This module provides functionality to load and validate stock data
from the database for feature engineering calculations.
"""

import pandas as pd
from typing import List, Optional, Dict, Any, Union
from datetime import date
from src.database.connection import fetch_all

from src.utils.logger import get_logger
from src.feature_engineering.config import config

logger = get_logger(__name__, utility="feature_engineering")


class StockDataLoader:
    """
    Loads and validates stock data from the database for feature engineering
    """

    def __init__(self):
        """
        Initialize the data loader
        """
        self.feature_config = config

    def load_stock_data(
        self, ticker: str, start_date: Union[str, date], end_date: Union[str, date]
    ) -> pd.DataFrame:
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
                start_date = start_date.strftime("%Y-%m-%d")
            if isinstance(end_date, date):
                end_date = end_date.strftime("%Y-%m-%d")

            # SQL query for historical_prices table (use positional params)
            query = (
                'SELECT date, "open", high, low, "close", volume, adjusted_close, vwap '
                "FROM historical_prices "
                "WHERE ticker = %s AND date >= %s AND date <= %s "
                "ORDER BY date ASC"
            )

            params = (ticker.upper(), start_date, end_date)

            rows = fetch_all(query, params=params, dict_cursor=False)
            # Construct DataFrame from rows; columns match the SELECT order
            df = (
                pd.DataFrame(
                    rows,
                    columns=[
                        "date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "adjusted_close",
                        "vwap",
                    ],
                )
                if rows
                else pd.DataFrame()
            )
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])

            if df.empty:
                logger.warning(f"No data found for {ticker} between {start_date} and {end_date}")
                return df

            # Set date as index
            df.set_index("date", inplace=True)

            # Convert numeric columns to float (handle PostgreSQL numeric type)
            numeric_columns = ["open", "high", "low", "close", "adjusted_close", "vwap"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Convert volume to int64
            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("int64")

            # Validate and clean the data
            df = self._validate_and_clean_data(df, ticker)

            return df

        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {str(e)}")
            raise

    def get_available_tickers(
        self, min_data_points: Optional[int] = None, market: str = "stocks"
    ) -> List[str]:
        """
        Get list of available tickers with sufficient data

        Args:
            min_data_points: Minimum number of data points required
            market: Market type filter (default: 'stocks')

        Returns:
            List of ticker symbols
        """
        if min_data_points is None:
            min_data_points = self.feature_config.data_quality.MIN_DATA_POINTS

        logger.info(f"Getting available tickers with at least {min_data_points} data points")

        try:
            # Join tickers table with historical_prices to get active tickers with sufficient data
            query = (
                "SELECT t.ticker, COUNT(hp.*) as data_points, t.name, t.market "
                "FROM tickers t "
                "INNER JOIN historical_prices hp ON t.ticker = hp.ticker "
                "WHERE t.active = true AND ( %s = 'all' OR t.market = %s ) AND t.\"type\" = 'CS' "
                "GROUP BY t.ticker, t.name, t.market "
                "HAVING COUNT(hp.*) >= %s "
                "ORDER BY COUNT(hp.*) DESC, t.ticker"
            )

            params = (
                market if market != "all" else "all",
                market if market != "all" else "all",
                min_data_points,
            )

            rows = fetch_all(query, params=params, dict_cursor=False)
            df = (
                pd.DataFrame(rows, columns=["ticker", "data_points", "name", "market"])
                if rows
                else pd.DataFrame()
            )

            tickers = df["ticker"].tolist()
            logger.info(f"Found {len(tickers)} tickers with at least {min_data_points} data points")

            return tickers

        except Exception as e:
            logger.error(f"Error getting available tickers: {str(e)}")
            raise

    def get_ticker_metadata(
        self, ticker: Optional[str] = None
    ) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Get metadata for a specific ticker or all tickers from the tickers table

        Args:
            ticker: Stock ticker symbol (None for all tickers)

        Returns:
            Dictionary with ticker metadata (if ticker specified) or DataFrame with all tickers metadata
        """
        try:
            # Build base query for ticker metadata
            base_query = (
                'SELECT id, ticker, "name", market, locale, primary_exchange, currency_name, '
                'active, "type", market_cap, weighted_shares_outstanding, round_lot, '
                "last_updated, created_at, cik, composite_figi, share_class_figi, sic_code, sic_description, ticker_root, total_employees, list_date "
                "FROM tickers "
            )

            if ticker is not None:
                query = base_query + "WHERE ticker = %s ORDER BY ticker"
                params = (ticker.upper(),)
            else:
                query = base_query + "ORDER BY ticker"
                params = None

            rows = fetch_all(query, params=params, dict_cursor=False)
            if not rows:
                if ticker is None:
                    logger.warning("No ticker metadata found")
                    return pd.DataFrame()
                else:
                    logger.warning(f"No metadata found for ticker {ticker}")
                    return {}

            df = pd.DataFrame(
                rows,
                columns=[
                    "id",
                    "ticker",
                    "name",
                    "market",
                    "locale",
                    "primary_exchange",
                    "currency_name",
                    "active",
                    "type",
                    "market_cap",
                    "weighted_shares_outstanding",
                    "round_lot",
                    "last_updated",
                    "created_at",
                    "cik",
                    "composite_figi",
                    "share_class_figi",
                    "sic_code",
                    "sic_description",
                    "ticker_root",
                    "total_employees",
                    "list_date",
                ],
            )

            if ticker is None:
                logger.info(f"Retrieved metadata for {len(df)} tickers")
                return df
            else:
                return df.iloc[0].to_dict()

        except Exception as e:
            logger.error(
                f"Error getting metadata for {'all tickers' if ticker is None else ticker}: {str(e)}"
            )
            return pd.DataFrame() if ticker is None else {}

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
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns for {ticker}: {missing_columns}")

        # Remove rows with missing OHLCV data
        df = df.dropna(subset=required_columns)

        # Ensure all price columns are numeric
        price_columns = ["open", "high", "low", "close"]
        for col in price_columns:
            if col in df.columns:
                # Convert to numeric, replacing any non-numeric values with NaN
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove rows where price conversion failed
        df = df.dropna(subset=price_columns)

        # Validate OHLC relationships and positive values
        invalid_ohlc = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
            | (df["open"] <= 0)
            | (df["high"] <= 0)
            | (df["low"] <= 0)
            | (df["close"] <= 0)
            | (df["volume"] < 0)
        )

        if invalid_ohlc.any():
            invalid_count = invalid_ohlc.sum()
            logger.warning(f"Removing {invalid_count} rows with invalid OHLC data for {ticker}")
            df = df[~invalid_ohlc]

        # Check for extreme outliers (more than 50% price change in one day)
        if len(df) > 1:
            price_changes = df["close"].pct_change().abs()
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

        return df

    def __enter__(self):
        """Context manager entry"""
        return self

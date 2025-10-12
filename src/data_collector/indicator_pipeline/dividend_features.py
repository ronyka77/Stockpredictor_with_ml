"""
Dividend feature calculation utilities for indicator pipeline.

This module provides functions to compute dividend-derived features
from price data and dividend records, with proper alignment to trading dates.
"""

import pandas as pd
import numpy as np
from datetime import timedelta

from src.utils.logger import get_logger

logger = get_logger(__name__, utility="feature_engineering")


def compute_dividend_features(
    price_df: pd.DataFrame, dividends_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute dividend-derived features aligned to price data index.

    Args:
        price_df: DataFrame with price data indexed by date, must contain 'close' column
        dividends_df: DataFrame with dividend data, must contain 'ex_dividend_date' and 'cash_amount' columns

    Returns:
        DataFrame indexed by same dates as price_df with columns:
        - dividend_amount: cash amount on ex-dividend date (NaN otherwise)
        - dividend_yield: dividend_amount / close on ex-dividend date (NaN if close missing)
        - is_ex_dividend: 1 if date is an ex-dividend date, 0 otherwise
        - days_since_last_dividend: days since previous ex-dividend date (NaN before first)
        - days_to_next_dividend: days until next ex-dividend date (NaN after last)
    """
    if price_df.empty:
        logger.warning("Empty price DataFrame provided")
        return pd.DataFrame(index=price_df.index)

    if dividends_df.empty:
        logger.info("No dividend data available, returning empty features")
        return pd.DataFrame(
            index=price_df.index,
            data={
                "dividend_amount": np.nan,
                "dividend_yield": np.nan,
                "is_ex_dividend": 0,
                "days_since_last_dividend": np.nan,
                "days_to_next_dividend": np.nan,
            },
        )

    # Ensure dividends_df has required columns
    required_cols = ["ex_dividend_date", "cash_amount"]
    missing_cols = [col for col in required_cols if col not in dividends_df.columns]
    if missing_cols:
        raise ValueError(f"dividends_df missing required columns: {missing_cols}")

    # Ensure price_df has close column
    if "close" not in price_df.columns:
        raise ValueError("price_df must contain 'close' column")

    # Initialize result DataFrame with same index as price_df
    result = pd.DataFrame(
        index=price_df.index,
        data={
            "dividend_amount": np.nan,
            "dividend_yield": np.nan,
            "is_ex_dividend": 0,
            "days_since_last_dividend": np.nan,
            "days_to_next_dividend": np.nan,
        },
    )

    # Get unique ex-dividend dates and map to trading dates
    ex_dates = dividends_df["ex_dividend_date"].dropna().unique()
    trading_dates = price_df.index

    # Convert trading dates to date objects for comparison
    trading_date_objects = set(trading_dates.date)

    mapped_dates = {}
    warnings = []

    for ex_date in ex_dates:
        if isinstance(ex_date, str):
            ex_date = pd.to_datetime(ex_date).date()

        # Find exact match first
        if ex_date in trading_date_objects:
            mapped_dates[ex_date] = ex_date
        else:
            # Find nearest prior trading date within 3 days
            candidates = []
            for i in range(1, 4):  # 1, 2, 3 days prior
                candidate = ex_date - timedelta(days=i)
                if candidate in trading_date_objects:
                    candidates.append(candidate)

            if candidates:
                # Use the closest (most recent) candidate
                mapped_dates[ex_date] = min(
                    candidates, key=lambda x: abs((x - ex_date).days)
                )
            else:
                warnings.append(
                    f"Could not map ex-dividend date {ex_date} to any trading date within 3 days"
                )
                continue

    # Log warnings for unmapped dates
    for warning in warnings:
        logger.warning(warning)

    # Process each mapped ex-dividend date
    for ex_date, trade_date in mapped_dates.items():
        # Convert trade_date back to Timestamp for DataFrame indexing
        trade_timestamp = pd.Timestamp(trade_date)

        # Get dividend amount (sum if multiple dividends on same ex-date)
        dividend_rows = dividends_df[dividends_df["ex_dividend_date"] == ex_date]
        total_amount = dividend_rows["cash_amount"].sum()

        # Set dividend_amount
        result.loc[trade_timestamp, "dividend_amount"] = total_amount

        # Set is_ex_dividend
        result.loc[trade_timestamp, "is_ex_dividend"] = 1

        # Calculate dividend_yield if close price is available
        if (
            pd.notna(price_df.loc[trade_timestamp, "close"])
            and price_df.loc[trade_timestamp, "close"] != 0
        ):
            result.loc[trade_timestamp, "dividend_yield"] = (
                total_amount / price_df.loc[trade_timestamp, "close"]
            )
        else:
            logger.warning(
                f"Missing or zero close price for {trade_date}, cannot calculate dividend_yield"
            )

    # Calculate days_since_last_dividend and days_to_next_dividend
    ex_timestamps_mapped = sorted(
        [pd.Timestamp(d) for d in mapped_dates.values()]
    )  # Trading timestamps that are ex-dividend dates

    if ex_timestamps_mapped:
        # For each trading date, find previous and next ex-dividend dates
        for current_date in result.index:
            # Find previous ex-dividend date
            prev_dates = [d for d in ex_timestamps_mapped if d < current_date]
            if prev_dates:
                prev_date = max(prev_dates)
                result.loc[current_date, "days_since_last_dividend"] = (
                    current_date - prev_date
                ).days

            # Find next ex-dividend date
            next_dates = [d for d in ex_timestamps_mapped if d > current_date]
            if next_dates:
                next_date = min(next_dates)
                result.loc[current_date, "days_to_next_dividend"] = (
                    next_date - current_date
                ).days

    logger.info(
        f"Computed dividend features for {len(result)} trading dates, {len(mapped_dates)} ex-dividend dates mapped"
    )
    return result

"""
Pandas-first dividend feature utilities with inline sanitization.

Computes dividend-derived features aligned to trading dates: maps ex-dividend dates
to nearest trading date, aggregates cash amounts, computes yields/flags, and
calculates days-since/days-to ex-dividend using vectorized operations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.core.logger import get_logger

logger = get_logger(__name__)


def compute_dividend_features(price_df: pd.DataFrame, dividends_df: pd.DataFrame) -> pd.DataFrame:
    """Compute dividend features aligned to price data index with inline sanitization.

    Args:
        price_df: DataFrame with price data indexed by date, must contain `close` column.
        dividends_df: DataFrame with `ex_dividend_date` and `cash_amount` columns.

    Returns:
        DataFrame with columns: dividend_amount, dividend_yield, is_ex_dividend,
        days_since_last_dividend, days_to_next_dividend (all properly typed).
    """
    if price_df.empty:
        return pd.DataFrame(index=price_df.index)

    if dividends_df.empty:
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

    # Validate required columns
    if not all(c in dividends_df.columns for c in ["ex_dividend_date", "cash_amount"]):
        raise ValueError("dividends_df missing ex_dividend_date or cash_amount")
    if "close" not in price_df.columns:
        raise ValueError("price_df must contain 'close' column")

    trading_index = pd.to_datetime(price_df.index).normalize()
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

    ex_dates = pd.to_datetime(dividends_df["ex_dividend_date"]).dropna().dt.normalize()
    ex_dates = ex_dates.drop_duplicates().sort_values().reset_index(drop=True)

    if ex_dates.empty:
        return result

    ex_df = pd.DataFrame({"ex_date": ex_dates})
    trades_df = (
        pd.DataFrame({"trade_date": trading_index}).drop_duplicates().sort_values("trade_date")
    )

    # Map ex-dates to nearest trading date within ±3 days
    mapped = pd.merge_asof(
        ex_df,
        trades_df,
        left_on="ex_date",
        right_on="trade_date",
        direction="nearest",
        tolerance=pd.Timedelta(days=3),
    )

    mapped["trade_date"].notna()
    # if not valid_mapped.all():
    #     for d in mapped.loc[~valid_mapped, "ex_date"]:
    # logger.warning(f"Could not map ex-dividend date {d.date()} to trading date")

    mapped = mapped.dropna(subset=["trade_date"]).copy()

    # Populate amounts and flags
    for _, row in mapped.iterrows():
        ex_date = row["ex_date"].date()
        trade_ts = pd.Timestamp(row["trade_date"])
        total_amount = dividends_df.loc[
            pd.to_datetime(dividends_df["ex_dividend_date"]).dt.normalize()
            == pd.Timestamp(ex_date),
            "cash_amount",
        ].sum()

        if trade_ts not in price_df.index:
            matched = price_df.index.normalize() == trade_ts.normalize()
            if matched.any():
                trade_ts = price_df.index[matched.argmax()]
            else:
                continue

        result.loc[trade_ts, "dividend_amount"] = total_amount
        result.loc[trade_ts, "is_ex_dividend"] = 1

        close_val = price_df.reindex([trade_ts])["close"].iloc[0]
        if pd.notna(close_val) and close_val != 0:
            result.loc[trade_ts, "dividend_yield"] = total_amount / close_val

    # Vectorized days since/to next dividend calculation
    ex_arr = np.array(sorted(pd.to_datetime(mapped["trade_date"]).tolist()), dtype="datetime64[ns]")
    dates = pd.to_datetime(result.index).values.astype("datetime64[ns]")
    pos = np.searchsorted(ex_arr, dates)

    prev_idx = pos - 1
    next_idx = pos
    prev_dates = np.where(prev_idx >= 0, ex_arr[prev_idx], np.datetime64("NaT"))
    next_dates = np.where(next_idx < len(ex_arr), ex_arr[next_idx], np.datetime64("NaT"))

    days_since = (pd.to_datetime(dates) - pd.to_datetime(prev_dates)).days
    days_to = (pd.to_datetime(next_dates) - pd.to_datetime(dates)).days

    result["days_since_last_dividend"] = days_since
    result["days_to_next_dividend"] = days_to

    # Inline sanitization: coerce types and handle empty strings
    result.replace("", np.nan, inplace=True)

    type_map = {
        "dividend_amount": "float",
        "dividend_yield": "float",
        "days_since_last_dividend": "Int64",
        "days_to_next_dividend": "Int64",
    }
    for col, dtype in type_map.items():
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").astype(dtype)

    if "is_ex_dividend" in result.columns:
        orig = result["is_ex_dividend"]
        result["is_ex_dividend"] = orig.map({True: 1, False: 0}).combine_first(orig)
        result["is_ex_dividend"] = (
            pd.to_numeric(result["is_ex_dividend"], errors="coerce").fillna(0).astype("int8")
        )

    logger.info(f"Computed dividend features: {len(result)} dates, {len(mapped)} ex-dates mapped")

    return result

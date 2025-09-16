"""Database utility helpers.

Contains bulk upsert helpers used by feature engineering and other modules.
"""

from typing import Dict, Any, Iterable, Tuple, List
import json
from psycopg2.extras import execute_values
from src.database.connection import get_global_pool
from src.utils.logger import get_logger

logger = get_logger(__name__, utility="database")


def _row_tuple_from_dict(d: Dict[str, Any]) -> Tuple:
    return (
        d["ticker"],
        d["date"],
        d["feature_category"],
        d["feature_name"],
        d["feature_value"],
        d.get("quality_score", None),
    )


def bulk_upsert_technical_features(
    rows: Iterable[Dict[str, Any]], page_size: int = 1000, overwrite: bool = True
) -> int:
    """
    Bulk upsert technical feature records into the `technical_features` table.
    
    Expects `rows` to be an iterable of dicts containing the keys:
    `ticker`, `date`, `feature_category`, `feature_name`, `feature_value`, and optional `quality_score`.
    Performs a single multi-row INSERT using psycopg2.extras.execute_values for performance.
    
    Parameters:
        rows: Iterable of feature dictionaries as described above.
        page_size: Number of rows per batch passed to execute_values (default 1000).
        overwrite: If True, conflicts on (ticker, date, feature_category, feature_name) will update
                   feature_value, quality_score, and calculation_timestamp. If False, conflicting rows
                   are ignored.
    
    Returns:
        int: Number of rows attempted (returns cursor.rowcount if available, otherwise len(rows)).
    
    Raises:
        Exception: Any database error is rolled back, logged, and re-raised.
    """
    rows = list(rows)
    if not rows:
        return 0

    if overwrite:
        conflict_clause = (
            "ON CONFLICT (ticker, date, feature_category, feature_name) "
            "DO UPDATE SET feature_value = EXCLUDED.feature_value, "
            "quality_score = EXCLUDED.quality_score, calculation_timestamp = CURRENT_TIMESTAMP"
        )
    else:
        conflict_clause = (
            "ON CONFLICT (ticker, date, feature_category, feature_name) DO NOTHING"
        )

    insert_sql = (
        "INSERT INTO technical_features "
        "(ticker, date, feature_category, feature_name, feature_value, quality_score) "
        "VALUES %s " + conflict_clause
    )

    tuples = [_row_tuple_from_dict(r) for r in rows]

    pool = get_global_pool()
    # Use the pool's connection context to acquire a raw psycopg2 connection
    with pool.connection() as conn:
        cur = conn.cursor()
        try:
            execute_values(cur, insert_sql, tuples, page_size=page_size)
            try:
                rowcount = cur.rowcount
            except Exception:
                rowcount = None
            conn.commit()
            return rowcount or len(tuples)
        except Exception:
            conn.rollback()
            logger.exception("bulk_upsert_technical_features failed")
            raise
        finally:
            try:
                cur.close()
            except Exception:
                pass


def _dividend_row_tuple_from_dict(d: Dict[str, Any]) -> Tuple:
    """
    Convert a dividend record dictionary into a tuple ordered for DB insertion.
    
    The returned tuple matches the dividends table columns:
      (id, ticker_id, cash_amount, currency, declaration_date,
       ex_dividend_date, pay_date, record_date, frequency,
       dividend_type, raw_payload)
    
    Notes:
    - Optional fields are taken via dict.get and will be None if missing.
    - `raw_payload`, if present, is JSON-serialized with `json.dumps(..., default=str)` so non-serializable values are stringified; if absent, the tuple value is None.
    """
    return (
        d["id"],
        d["ticker_id"],
        d["cash_amount"],
        d.get("currency", None),
        d.get("declaration_date", None),
        d.get("ex_dividend_date", None),
        d.get("pay_date", None),
        d.get("record_date", None),
        d.get("frequency", None),
        d.get("dividend_type", None),
        # Ensure raw_payload is JSON-serializable for DB insertion
        (json.dumps(d.get("raw_payload", None), default=str) if d.get("raw_payload", None) is not None else None),
    )

def _drop_dividend_duplicates(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return a list of dividend rows with duplicates removed.
    
    Rows are considered duplicates when all of (ticker_id, cash_amount, ex_dividend_date, pay_date) are identical.
    Keeps the first occurrence for each unique combination and preserves the original input order. The input may be any iterable of dict-like rows; a list is returned.
    """
    seen = set()
    unique_rows = []
    for row in rows:
        key = (
            row.get("ticker_id"),
            row.get("cash_amount"),
            row.get("ex_dividend_date"),
            row.get("pay_date"),
        )
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    return unique_rows


def _upsert_dividends_batch(rows: Iterable[Dict[str, Any]], page_size: int = 500) -> int:
    """
    Bulk upsert dividend records into the `dividends` table.
    
    This function deduplicates the input rows (keeps the first occurrence for each unique
    combination of ticker_id, cash_amount, ex_dividend_date, pay_date), then performs a
    batched INSERT using psycopg2.extras.execute_values. On conflict by `id` the row is
    updated with the incoming ex_dividend_date, pay_date, cash_amount, raw_payload and
    updated_at = now().
    
    Parameters:
        rows: Iterable of dicts representing dividend records. Expected keys include
            at minimum: id, ticker_id, cash_amount, currency, declaration_date,
            ex_dividend_date, pay_date, record_date, frequency, dividend_type,
            and (optionally) raw_payload. Raw payload JSON serialization is handled
            by the helper that builds DB tuples.
        page_size: Number of rows per batch passed to execute_values (default 500).
    
    Returns:
        The number of rows the function attempted to insert/update. If the DB cursor
        provides a rowcount that value is returned; otherwise the number of tuples
        actually sent to the database (after deduplication) is returned.
    
    Raises:
        Propagates any exception raised by the database operations (the transaction
        will be rolled back before re-raising).
    """
    rows = list(rows)
    if not rows:
        return 0
    rows = _drop_dividend_duplicates(rows)
    insert_sql = (
        "INSERT INTO dividends "
        "(id, ticker_id, cash_amount, currency, declaration_date, ex_dividend_date, pay_date, record_date, frequency, dividend_type, raw_payload) "
        "VALUES %s "
        "ON CONFLICT (id) DO UPDATE SET "
        "ex_dividend_date = EXCLUDED.ex_dividend_date, "
        "pay_date = EXCLUDED.pay_date, "
        "cash_amount = EXCLUDED.cash_amount, "
        "raw_payload = EXCLUDED.raw_payload, "
        "updated_at = now();"
    )

    tuples: List[Tuple] = [_dividend_row_tuple_from_dict(r) for r in rows]

    pool = get_global_pool()
    with pool.connection() as conn:
        cur = conn.cursor()
        try:
            execute_values(cur, insert_sql, tuples, page_size=page_size)
            try:
                rowcount = cur.rowcount
            except Exception:
                rowcount = None
            conn.commit()
            return rowcount or len(tuples)
        except Exception:
            conn.rollback()
            logger.exception("_upsert_dividends_batch failed")
            raise
        finally:
            try:
                cur.close()
            except Exception:
                pass

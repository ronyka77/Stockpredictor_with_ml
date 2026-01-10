"""Database utility helpers.

Contains bulk upsert helpers used by feature engineering and other modules.
"""

from dataclasses import dataclass
from typing import Dict, Any, Iterable, Tuple, List
import json
from src.database.connection import execute_values

from src.utils.core.logger import get_logger

logger = get_logger(__name__, utility="database")


@dataclass
class DBBatch:
    """A bulk DB write payload detached from any dataframe library."""

    columns: tuple[str, ...]
    rows: list[tuple]

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError("DBBatch.columns must be non-empty")

        expected = len(self.columns)
        bad = next((i for i, r in enumerate(self.rows) if len(r) != expected), None)
        if bad is not None:
            raise ValueError(
                f"DBBatch row length mismatch at index {bad}: expected {expected}, got {len(self.rows[bad])}"
            )


def _row_tuple_from_dict(d: Dict[str, Any]) -> Tuple:
    """Build a DB parameter tuple for the `technical_features` JSON schema.

    Expected dict keys:
        ticker: str
        date: date or ISO string (DB driver will adapt)
        data: JSON-serializable dict (will be json.dumps'd)
    """

    # Store as a JSON string; execute_values flattens tuple params, so we pass
    # a single scalar for the jsonb column.
    payload = json.dumps(d["data"], default=str)
    return (d["ticker"], d["date"], payload)


def bulk_upsert_technical_features(
    rows: Iterable[Dict[str, Any]], page_size: int = 1000, overwrite: bool = True
) -> int:
    """Bulk upsert technical features using execute_values for performance."""
    rows = list(rows)
    if not rows:
        return 0

    if overwrite:
        conflict_clause = "ON CONFLICT (ticker, date) DO UPDATE SET data = EXCLUDED.data"
    else:
        conflict_clause = "ON CONFLICT (ticker, date) DO NOTHING"

    insert_sql = "INSERT INTO technical_features (ticker, date, data) VALUES %s " + conflict_clause

    tuples = [_row_tuple_from_dict(r) for r in rows]

    try:
        execute_values(insert_sql, tuples, page_size=page_size)
        return len(tuples)
    except Exception:
        logger.exception("bulk_upsert_technical_features failed")
        raise


def _dividend_row_tuple_from_dict(d: Dict[str, Any]) -> Tuple:
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
        (
            json.dumps(d.get("raw_payload", None), default=str)
            if d.get("raw_payload", None) is not None
            else None
        ),
    )


def _upsert_dividends_batch(rows: Iterable[Dict[str, Any]], page_size: int = 500) -> int:
    """Bulk upsert dividend rows into the `dividends` table.

    Args:
        rows: Iterable of dicts with keys matching transformed dividend records
        page_size: page size passed to execute_values

    Returns:
        Number of rows attempted to insert (best-effort). If DB returns
        rowcount it is returned; otherwise returns len(rows).
    """
    rows = list(rows)
    if not rows:
        return 0
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

    try:
        execute_values(insert_sql, tuples, page_size=page_size)
        return len(tuples)
    except Exception:
        logger.exception("_upsert_dividends_batch failed")
        raise

"""Database utility helpers.

Contains bulk upsert helpers used by feature engineering and other modules.
"""

from typing import Dict, Any, Iterable, Tuple
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
    """Bulk upsert technical features using execute_values for performance.

    Args:
        rows:   Iterable of dicts with keys: ticker, date, feature_category,
                feature_name, feature_value, quality_score
        page_size: page size passed to execute_values
        overwrite: If True, ON CONFLICT DO UPDATE; if False, ON CONFLICT DO NOTHING

    Returns:
        Number of rows attempted to insert (best-effort). If DB returns
        rowcount it is returned; otherwise returns len(rows).
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

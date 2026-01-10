"""
Dividend pipeline - compact refactor (safe shrink).

Preserves behavior: bad records are appended to BAD_STAGING_PATH, ValueError is raised
for invalid records, public function signatures unchanged.
"""

from decimal import Decimal, InvalidOperation
from typing import Dict, Any, Optional, List
from datetime import date
import json
import os
from concurrent.futures import ThreadPoolExecutor
import threading

from dateutil.parser import isoparse

from src.data_collector.polygon_data.client import PolygonDataClient
from src.data_collector.polygon_data.data_storage import DataStorage
from src.database.db_utils import _upsert_dividends_batch
from src.utils.core.logger import get_logger

logger = get_logger(__name__)

DIVIDEND_INGESTION_CONFIG = {
    "max_workers": int(os.getenv("DIVIDENDS_MAX_WORKERS", 32)),
    "requests_per_minute": int(os.getenv("DIVIDENDS_RPM", 100)),
    "batch_size": int(os.getenv("DIVIDENDS_BATCH_SIZE", 100)),
}
BAD_STAGING_PATH = os.getenv("DIVIDENDS_BAD_STAGING", "./dividends_staging_bad.jsonl")


def _write_bad_record(raw: Dict[str, Any], reason: str) -> None:
    try:
        with open(BAD_STAGING_PATH, "a", encoding="utf-8") as fh:
            json.dump({"reason": reason, "payload": raw}, fh, default=str)
            fh.write("\n")
    except Exception as e:
        logger.error(f"Failed to write bad record to staging: {e}")


def _get_first(raw: Dict[str, Any], *keys: str):
    return next((v for k in keys if (v := raw.get(k)) is not None), None)


def _parse_date_safe(val: Any) -> Optional[date]:
    if not val:
        return None
    return isoparse(val).date()


def transform_dividend_record(raw: Dict[str, Any], ticker_id: int) -> Optional[Dict[str, Any]]:
    """
    Returns dict on success, None if record should be skipped.
    Raises ValueError on invalid data (staged).
    """
    polygon_id = _get_first(raw, "id", "dividend_id", "polygon_id")
    if not polygon_id:
        _write_bad_record(raw, "missing_id")
        return None

    raw_cash = _get_first(raw, "cash_amount", "amount", "cash")
    if raw_cash is None:
        _write_bad_record(raw, "missing_cash_amount")
        raise ValueError("Missing cash_amount")
    try:
        cash_amount = Decimal(str(raw_cash))
    except (InvalidOperation, TypeError):
        _write_bad_record(raw, "invalid_cash_amount")
        raise ValueError("Invalid cash_amount")

    try:
        declaration_date = _parse_date_safe(_get_first(raw, "declaration_date"))
        ex_dividend_date = _parse_date_safe(_get_first(raw, "ex_dividend_date"))
        pay_date = _parse_date_safe(_get_first(raw, "pay_date"))
        record_date = _parse_date_safe(_get_first(raw, "record_date"))
    except Exception:
        _write_bad_record(raw, "invalid_date")
        raise ValueError("Invalid date field")

    currency = raw.get("currency")
    if currency and (not isinstance(currency, str) or len(currency) != 3):
        _write_bad_record(raw, "invalid_currency")
        raise ValueError("Invalid currency")

    return {
        "id": str(polygon_id),
        "ticker_id": int(ticker_id),
        "cash_amount": cash_amount,
        "currency": currency,
        "declaration_date": declaration_date,
        "ex_dividend_date": ex_dividend_date,
        "pay_date": pay_date,
        "record_date": record_date,
        "frequency": raw.get("frequency"),
        "dividend_type": raw.get("type") or raw.get("dividend_type"),
        "raw_payload": raw,
    }


def _dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate rows by (ticker_id, pay_date, cash_amount, ex_dividend_date).
    When duplicates exist, prefer rows with ex_dividend_date then pay_date."""
    groups = {}
    for r in rows:
        key = (
            r.get("ticker_id"),
            r.get("pay_date"),
            r.get("cash_amount"),
            r.get("ex_dividend_date"),
        )
        existing = groups.get(key)
        if not existing:
            groups[key] = r
            continue
        # choose better row: prefer ex_dividend_date then pay_date
        better = max(
            (existing, r), key=lambda x: (bool(x.get("ex_dividend_date")), bool(x.get("pay_date")))
        )
        groups[key] = better
    dup_count = len(rows) - len(groups)
    if dup_count:
        logger.info(f"Deduplicated {dup_count} rows")
    return list(groups.values())


def ingest_dividends_for_ticker(
    client: PolygonDataClient,
    ticker: Dict[str, Any],
    batch_size: int = DIVIDEND_INGESTION_CONFIG["batch_size"],
) -> Dict[str, int]:
    stats = {"fetched": 0, "upserted": 0, "invalid": 0, "skipped": 0}
    rows: List[Dict[str, Any]] = []
    ticker_id = ticker["id"]
    ticker_name = ticker["ticker"]

    dividends = client.get_dividends(ticker=ticker_name, limit=1000)
    stats["fetched"] = len(dividends)

    for raw in dividends:
        try:
            transformed = transform_dividend_record(raw, ticker_id)
            if transformed is None:
                stats["skipped"] += 1
                continue
            rows.append(transformed)
        except ValueError:
            stats["invalid"] += 1
            continue

        if len(rows) >= batch_size:
            deduped = _dedupe_rows(rows)
            stats["upserted"] += _upsert_dividends_batch(deduped, page_size=batch_size)
            rows.clear()

    if rows:
        deduped = _dedupe_rows(rows)
        stats["upserted"] += _upsert_dividends_batch(deduped, page_size=batch_size)

    logger.info(f"{ticker_name}: fetched={stats['fetched']} upserted={stats['upserted']}")
    return stats


def ingest_dividends_for_all_tickers(
    batch_size: int = DIVIDEND_INGESTION_CONFIG["batch_size"],
    max_workers: int = DIVIDEND_INGESTION_CONFIG["max_workers"],
    requests_per_minute: int = DIVIDEND_INGESTION_CONFIG["requests_per_minute"],
) -> Dict[str, int]:
    storage = DataStorage()
    tickers = storage.get_tickers()
    results_lock = threading.Lock()
    overall_stats = {"tickers_processed": 0, "total_fetched": 0, "total_upserted": 0, "errors": 0}

    def process(ticker: Dict[str, Any]) -> None:
        thread_id = threading.get_ident()
        try:
            client = PolygonDataClient(requests_per_minute=requests_per_minute)
            stats = ingest_dividends_for_ticker(client, ticker, batch_size=batch_size)
            with results_lock:
                overall_stats["tickers_processed"] += 1
                overall_stats["total_fetched"] += stats.get("fetched", 0)
                overall_stats["total_upserted"] += stats.get("upserted", 0)
        except Exception as e:
            logger.error(f"Thread {thread_id} failed {ticker['ticker']}: {e}")
            with results_lock:
                overall_stats["errors"] += 1

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        ex.map(process, tickers)

    # If any worker reported errors, surface as a warning at the end
    if overall_stats.get("errors", 0):
        logger.warning(f"Concurrent dividend ingestion completed with errors: {overall_stats}")
    else:
        logger.info(f"Concurrent dividend ingestion complete: {overall_stats}")
    return overall_stats


if __name__ == "__main__":
    cfg = DIVIDEND_INGESTION_CONFIG
    logger.info(
        f"Starting dividend ingestion: workers={cfg['max_workers']} rpm={cfg['requests_per_minute']} batch_size={cfg['batch_size']}"
    )
    try:
        result = ingest_dividends_for_all_tickers()
        logger.info(f"Dividend ingestion completed successfully: {result}")
    except Exception as e:
        logger.error(f"Dividend ingestion failed: {e}")
        import sys

        sys.exit(1)

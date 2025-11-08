"""
Dividend pipeline: transformation, validation and per-ticker ingestion helpers.

This module implements `transform_dividend_record` according to the
`dividends_ingestion_plan.md` validation requirements and provides lightweight
staging for bad records. Supports both sequential and concurrent processing.
"""

from contextlib import suppress
from decimal import Decimal, InvalidOperation
from typing import Dict, Any, Optional, List
from datetime import date
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from dateutil.parser import isoparse

from src.data_collector.polygon_data.client import PolygonDataClient
from src.data_collector.polygon_data.data_storage import DataStorage
from src.database.db_utils import _upsert_dividends_batch
from src.utils.logger import get_logger

logger = get_logger(__name__, utility="data_collector")

# Configuration variables for dividend ingestion
DIVIDEND_INGESTION_CONFIG = {
    "concurrent": True,  # Enable concurrent processing
    "max_workers": 32,  # Number of worker threads (only used if concurrent=True)
    "requests_per_minute": 100,  # API requests per minute per worker
    "batch_size": 100,  # Database batch size for upserts
}


class TransformError(Exception):
    pass


class SkipRecord(Exception):
    pass


BAD_STAGING_PATH = os.getenv("DIVIDENDS_BAD_STAGING", "./dividends_staging_bad.jsonl")


def _write_bad_record(raw: Dict[str, Any], reason: str) -> None:
    try:
        with open(BAD_STAGING_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"reason": reason, "payload": raw}, default=str) + "\n")
    except Exception as e:
        logger.error(f"Failed to write bad record to staging: {e}")


def transform_dividend_record(raw: Dict[str, Any], ticker_id: int) -> Dict[str, Any]:
    """
    Transform and validate a raw Polygon dividend record into the DB row dict.

    Rules (per plan):
        - Require `id` (Polygon id) — if absent, raise SkipRecord
        - Parse date fields to date objects
        - Parse `cash_amount` to Decimal
        - Validate `currency` length == 3 if present
        - Attach `raw_payload` and `ticker_id`

    Returns a dict suitable for `_upsert_dividends_batch`.
    Raises TransformError on validation failure, SkipRecord to intentionally skip.
    """
    try:
        polygon_id = raw.get("id") or raw.get("dividend_id") or raw.get("polygon_id")
        if not polygon_id:
            _write_bad_record(raw, "missing_id")
            raise SkipRecord("Missing polygon id")

        # Normalize id to string
        record_id = str(polygon_id)

        # Cash amount
        raw_cash = raw.get("cash_amount") or raw.get("amount") or raw.get("cash")
        if raw_cash is None:
            _write_bad_record(raw, "missing_cash_amount")
            raise TransformError("Missing cash_amount")

        try:
            cash_amount = Decimal(str(raw_cash))
        except (InvalidOperation, TypeError):
            _write_bad_record(raw, "invalid_cash_amount")
            raise TransformError("Invalid cash_amount")

        # Dates: try common key names
        def _parse_date_field(key: str) -> Optional[date]:
            val = raw.get(key)
            if not val:
                return None
            with suppress(Exception):
                return isoparse(val).date()

        try:
            declaration_date = _parse_date_field("declaration_date")
        except Exception:
            _write_bad_record(raw, "invalid_declaration_date")
            raise TransformError("Invalid declaration_date")

        try:
            ex_dividend_date = _parse_date_field("ex_dividend_date")
        except Exception:
            _write_bad_record(raw, "invalid_ex_dividend_date")
            raise TransformError("Invalid ex_dividend_date")

        try:
            pay_date = _parse_date_field("pay_date")
        except Exception:
            _write_bad_record(raw, "invalid_pay_date")
            raise TransformError("Invalid pay_date")

        try:
            record_date = _parse_date_field("record_date")
        except Exception:
            _write_bad_record(raw, "invalid_record_date")
            raise TransformError("Invalid record_date")

        # Currency
        currency = raw.get("currency")
        if currency and (not isinstance(currency, str) or len(currency) != 3):
            _write_bad_record(raw, "invalid_currency")
            raise TransformError("Invalid currency")

        # Frequency and type
        frequency = raw.get("frequency")
        dividend_type = raw.get("type") or raw.get("dividend_type")

        transformed = {
            "id": record_id,
            "ticker_id": int(ticker_id),
            "cash_amount": cash_amount,
            "currency": currency,
            "declaration_date": declaration_date,
            "ex_dividend_date": ex_dividend_date,
            "pay_date": pay_date,
            "record_date": record_date,
            "frequency": frequency,
            "dividend_type": dividend_type,
            "raw_payload": raw,
        }

        return transformed

    except SkipRecord:
        logger.info("Skipping dividend record without polygon id")
        raise
    except TransformError:
        logger.warning("Dividend transform failed, record staged for review")
        raise
    except Exception as e:
        _write_bad_record(raw, f"unexpected_transform_error: {e}")
        logger.exception(f"Unexpected error transforming dividend record: {e}")
        raise TransformError(str(e))


def ingest_dividends_for_ticker(
    client, ticker: Dict[str, Any], batch_size: int = 100
) -> Dict[str, int]:
    """
    Fetch, transform and upsert dividends for a single ticker.

    Args:
        client: PolygonDataClient instance
        ticker: ticker dictionary
        batch_size: batch size for DB upsert

    Returns:
        Dict with counts: fetched, inserted/updated, invalid, skipped
    """
    stats = {"fetched": 0, "upserted": 0, "invalid": 0, "skipped": 0}
    rows = []
    ticker_id = ticker["id"]
    ticker_name = ticker["ticker"]
    try:
        dividends = client.get_dividends(ticker=ticker_name, limit=1000)
        stats["fetched"] = len(dividends)

        for raw in dividends:
            try:
                transformed = transform_dividend_record(raw, ticker_id)
                rows.append(transformed)
            except SkipRecord:
                stats["skipped"] += 1
                continue
            except TransformError:
                stats["invalid"] += 1
                continue

            # Flush batch
            if len(rows) >= batch_size:
                upserted = _upsert_dividends_batch(rows, page_size=batch_size)
                stats["upserted"] += upserted
                rows = []

        # Final flush
        if rows:
            upserted = _upsert_dividends_batch(rows, page_size=batch_size)
            stats["upserted"] += upserted

        logger.info(f"Completed ingest for {ticker_name}: {stats}")
        return stats

    except Exception as e:
        logger.exception(f"Error ingesting dividends for {ticker_name}: {e}")
        raise


def ingest_dividends_for_all_tickers(
    batch_size: int = 100,
    concurrent: bool = False,
    max_workers: int = 4,
    requests_per_minute: int = 5,
) -> Dict[str, int]:
    """
    Orchestrator over all tickers in storage.get_available_tickers().

    Args:
        batch_size: Database batch size for upserts
        concurrent: Whether to process tickers concurrently
        max_workers: Maximum concurrent workers (only used if concurrent=True)
        requests_per_minute: API requests per minute per worker

    Returns:
        Dict with overall statistics
    """
    storage = DataStorage()
    tickers = storage.get_tickers()

    if concurrent:
        return _ingest_dividends_concurrent(
            tickers=tickers,
            batch_size=batch_size,
            max_workers=max_workers,
            requests_per_minute=requests_per_minute,
        )
    else:
        return _ingest_dividends_sequential(tickers=tickers, batch_size=batch_size)


def _ingest_dividends_sequential(tickers: List[Dict[str, Any]], batch_size: int) -> Dict[str, int]:
    """Sequential processing of dividend ingestion."""
    client = PolygonDataClient()
    DataStorage()
    overall = {"tickers_processed": 0, "total_fetched": 0, "total_upserted": 0}

    for t in tickers:
        try:
            stats = ingest_dividends_for_ticker(client, t, batch_size=batch_size)
            overall["tickers_processed"] += 1
            overall["total_fetched"] += stats.get("fetched", 0)
            overall["total_upserted"] += stats.get("upserted", 0)
        except Exception as e:
            logger.error(f"Failed ingest for ticker {t['ticker']}: {e}")
            continue

    logger.info(f"Dividend ingest for all tickers complete: {overall}")
    return overall


def _ingest_dividends_concurrent(
    tickers: List[Dict[str, Any]], batch_size: int, max_workers: int, requests_per_minute: int
) -> Dict[str, int]:
    """Concurrent processing of dividend ingestion using ThreadPoolExecutor."""
    logger.info(f"Starting concurrent dividend ingestion with {max_workers} workers")

    # Thread-safe result collection
    results_lock = threading.Lock()
    overall_stats = {"tickers_processed": 0, "total_fetched": 0, "total_upserted": 0, "errors": 0}

    def process_ticker_with_lock(ticker: Dict[str, Any]) -> Dict[str, int]:
        """Process a single ticker and return its stats."""
        thread_id = threading.current_thread().ident
        logger.debug(f"Thread {thread_id}: Processing ticker {ticker['ticker']}")

        try:
            # Each thread gets its own client instance for rate limiting
            client = PolygonDataClient(requests_per_minute=requests_per_minute)
            DataStorage()

            stats = ingest_dividends_for_ticker(client, ticker, batch_size=batch_size)

            with results_lock:
                overall_stats["tickers_processed"] += 1
                overall_stats["total_fetched"] += stats.get("fetched", 0)
                overall_stats["total_upserted"] += stats.get("upserted", 0)

            logger.debug(f"Thread {thread_id}: Completed ticker {ticker['ticker']}: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Thread {thread_id}: Failed ticker {ticker['ticker']}: {e}")
            with results_lock:
                overall_stats["errors"] += 1
            return {"fetched": 0, "upserted": 0, "invalid": 0, "skipped": 0}

    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all ticker processing tasks
        future_to_ticker = {
            executor.submit(process_ticker_with_lock, ticker): ticker for ticker in tickers
        }

        # Wait for all tasks to complete and handle any remaining exceptions
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                future.result()  # This will raise any exception that occurred
            except Exception as e:
                logger.error(f"Ticker {ticker['ticker']} failed in concurrent processing: {e}")

    logger.info(f"Concurrent dividend ingestion complete: {overall_stats}")
    return overall_stats


if __name__ == "__main__":
    # Use configuration variables for dividend ingestion
    config = DIVIDEND_INGESTION_CONFIG

    logger.info(
        f"Starting dividend ingestion with config: "
        f"concurrent={config['concurrent']}, "
        f"workers={config['max_workers']}, "
        f"rpm={config['requests_per_minute']}, "
        f"batch_size={config['batch_size']}"
    )

    try:
        result = ingest_dividends_for_all_tickers(
            batch_size=config["batch_size"],
            concurrent=config["concurrent"],
            max_workers=config["max_workers"],
            requests_per_minute=config["requests_per_minute"],
        )
        logger.info(f"Dividend ingestion completed successfully: {result}")
    except Exception as e:
        logger.error(f"Dividend ingestion failed: {e}")
        import sys

        sys.exit(1)

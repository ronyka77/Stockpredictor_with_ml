"""
Dividend pipeline: transformation, validation and per-ticker ingestion helpers.

This module implements `transform_dividend_record` according to the
`dividends_ingestion_plan.md` validation requirements and provides lightweight
staging for bad records.
"""
from decimal import Decimal, InvalidOperation
from typing import Dict, Any, Optional
from datetime import date
import json
import os

from dateutil.parser import isoparse

from src.data_collector.polygon_data.client import PolygonDataClient
from src.data_collector.polygon_data.data_storage import DataStorage
from src.database.db_utils import _upsert_dividends_batch
from src.utils.logger import get_logger

logger = get_logger(__name__, utility="data_collector")


class TransformError(Exception):
    pass


class SkipRecord(Exception):
    pass


BAD_STAGING_PATH = os.getenv("DIVIDENDS_BAD_STAGING", "./dividends_staging_bad.jsonl")


def _write_bad_record(raw: Dict[str, Any], reason: str) -> None:
    """
    Append a failed/divergent raw record and a short reason to the bad-record staging file (JSONL).
    
    Writes a single JSON line to the file configured by BAD_STAGING_PATH containing {"reason": <reason>, "payload": <raw>}. The payload is serialized using JSON with a string fallback for non-serializable objects. Any IO/serialization failure is swallowed after logging an error; callers should not rely on this function to raise on write failure.
    
    Parameters:
        raw (Dict[str, Any]): The original raw record that failed validation or transformation.
        reason (str): Short human-readable explanation of why the record was staged (e.g., validation error).
    """
    try:
        with open(BAD_STAGING_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"reason": reason, "payload": raw}, default=str) + "\n")
    except Exception as e:
        logger.error(f"Failed to write bad record to staging: {e}")


def transform_dividend_record(raw: Dict[str, Any], ticker_id: int) -> Dict[str, Any]:
    """
    Transform and validate a raw Polygon dividend record into a database-ready row dict.
    
    Parses and normalizes fields from a raw Polygon dividend payload:
    - Ensures a polygon identifier is present (checks "id", "dividend_id", "polygon_id"); missing id stages the record and raises SkipRecord.
    - Normalizes the identifier to a string and attaches ticker_id (DB PK for the ticker).
    - Parses monetary amount from "cash_amount", "amount", or "cash" into a Decimal; missing or invalid amounts stage the record and raise TransformError.
    - Parses date fields ("declaration_date", "ex_dividend_date", "pay_date", "record_date") into datetime.date objects; invalid dates stage the record and raise TransformError.
    - Validates optional "currency" is a 3-character string; invalid currency stages the record and raises TransformError.
    - Preserves optional "frequency" and "type"/"dividend_type" and includes the original raw payload under "raw_payload".
    
    Returns:
        dict: A transformed row with keys suitable for _upsert_dividends_batch:
        {
            "id": str,
            "ticker_id": int,
            "cash_amount": Decimal,
            "currency": Optional[str],
            "declaration_date": Optional[date],
            "ex_dividend_date": Optional[date],
            "pay_date": Optional[date],
            "record_date": Optional[date],
            "frequency": Optional[Any],
            "dividend_type": Optional[Any],
            "raw_payload": Dict[str, Any],
        }
    
    Raises:
        SkipRecord: when the record must be skipped because it lacks a polygon identifier.
        TransformError: for validation or parsing failures (amount, dates, currency, etc.).
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
            """
            Parse an ISO-8601 datetime string stored under `key` in the surrounding `raw` mapping and return a datetime.date.
            
            Parameters:
                key (str): Key to look up in the surrounding `raw` mapping (closure). If the key is missing or the value is falsy, returns None.
            
            Returns:
                Optional[date]: The parsed date component when a valid ISO datetime string is present, otherwise None.
            
            Notes:
                - This function reads from a `raw` variable that must exist in the enclosing scope.
                - Any exception raised by `isoparse` is propagated to the caller.
            """
            val = raw.get(key)
            if not val:
                return None
            try:
                return isoparse(val).date()
            except Exception:
                # Keep failure localized
                raise

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
        if currency:
            if not isinstance(currency, str) or len(currency) != 3:
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
    client, storage, ticker: Dict[str, Any], batch_size: int = 500
) -> Dict[str, int]:
    """
    Fetch, transform, and upsert dividends for a single ticker in batched transactions.
    
    Fetches raw dividend records from the provided client for the given ticker, transforms and validates each record via transform_dividend_record, stages bad records, and upserts validated rows to the database in batches. Tracks and returns simple ingestion statistics.
    
    Parameters:
        ticker (Dict[str, Any]): Ticker record containing at least `id` (int) and `ticker` (str).
        batch_size (int): Number of rows per upsert batch (default 500).
    
    Returns:
        Dict[str, int]: Counts for the ingestion run with keys:
            - fetched: number of raw records retrieved from the client
            - upserted: number of rows successfully upserted into the DB
            - invalid: number of records that failed transformation/validation
            - skipped: number of records intentionally skipped (e.g., missing polygon id)
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


def ingest_dividends_for_all_tickers(batch_size: int = 500) -> Dict[str, int]:
    """
    Run dividend ingestion for every ticker in storage and return aggregated statistics.
    
    Creates a PolygonDataClient and DataStorage, retrieves all tickers via storage.get_tickers(), and invokes ingest_dividends_for_ticker for each ticker. Accumulates and returns totals:
    - tickers_processed: number of tickers successfully processed
    - total_fetched: sum of fetched dividend records across tickers
    - total_upserted: sum of upserted rows across tickers
    
    Per-ticker failures are logged and skipped so processing continues for remaining tickers.
    
    Parameters:
        batch_size (int): Maximum number of transformed rows to buffer before flushing an upsert for each ticker.
    
    Returns:
        Dict[str, int]: Aggregated statistics with keys "tickers_processed", "total_fetched", and "total_upserted".
    """
    client = PolygonDataClient()
    storage = DataStorage()
    tickers = storage.get_tickers()
    overall = {"tickers_processed": 0, "total_fetched": 0, "total_upserted": 0}
    for t in tickers:
        try:
            stats = ingest_dividends_for_ticker(client, storage, t, batch_size=batch_size)
            overall["tickers_processed"] += 1
            overall["total_fetched"] += stats.get("fetched", 0)
            overall["total_upserted"] += stats.get("upserted", 0)
        except Exception:
            logger.error(f"Failed ingest for ticker {t['ticker']}")
            continue

    logger.info(f"Dividend ingest for all tickers complete: {overall}")
    return overall

if __name__ == "__main__":
    ingest_dividends_for_all_tickers(batch_size=500)

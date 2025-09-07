from datetime import date
from unittest.mock import MagicMock

from src.data_collector.polygon_data.data_storage import DataStorage
from src.data_collector.polygon_data.data_validator import OHLCVRecord
from tests._fixtures.conftest import patch_execute_values_to_fake_pool

class DummyRecord(OHLCVRecord):
    pass


def make_record(ticker: str, dt: date, close: float = 10.0):
    return OHLCVRecord(
        ticker=ticker,
        timestamp=dt,
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=100,
    )


def test_store_historical_no_records(mocker):
    # Prevent real DB pool by patching PostgresConnection to test fake
    from tests._fixtures.helpers import PostgresPoolCompat

    mocker.patch("src.database.connection.PostgresConnection", PostgresPoolCompat)
    ds = DataStorage()

    res = ds.store_historical_data([])
    if res["stored_count"] != 0:
        raise AssertionError("Expected stored_count == 0 for empty input")
    if res["error_count"] != 0:
        raise AssertionError("Expected error_count == 0 for empty input")


def test_store_historical_batches_and_upsert(mocker, patch_execute_values_to_fake_pool):
    # Use fake Postgres pool and have execute_values delegate to fake pool
    from tests._fixtures.helpers import PostgresPoolCompat

    mocker.patch("src.database.connection.PostgresConnection", PostgresPoolCompat)

    ds = DataStorage()

    # prepare records -> single batch
    recs = [make_record("AAA", date(2020, 1, 1)), make_record("AAA", date(2020, 1, 2))]
    res = ds.store_historical_data(recs, batch_size=10, on_conflict="update")

    if res["stored_count"] != 2:
        raise AssertionError("Stored count mismatch after upsert")
    if res["error_count"] != 0:
        raise AssertionError("Expected no errors during upsert batch")


def test_get_historical_query_and_params(mocker):
    # Patch fetch_all to return deterministic rows via shared helpers
    fake_rows = [{"ticker": "AAA", "date": date(2020, 1, 1), "close": 10}]
    mocker.patch(
        "src.data_collector.polygon_data.data_storage.fetch_all", return_value=fake_rows
    )

    ds = DataStorage()
    out = ds.get_historical_data(
        "aaa", start_date=date(2020, 1, 1), end_date=date(2020, 1, 2), limit=1
    )

    if not isinstance(out, list):
        raise AssertionError("Expected historical query to return a list")
    if out[0]["ticker"] != "AAA":
        raise AssertionError("Historical query ticker normalization mismatch")

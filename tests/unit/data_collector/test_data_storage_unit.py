import pandas as pd
import pytest

from src.data_collector.polygon_data.data_storage import DataStorage
from tests._fixtures.conftest import patch_execute_values_to_fake_pool

def test_store_historical_data_returns_zero_on_empty_list():
    ds = DataStorage()
    res = ds.store_historical_data([])
    assert res["stored_count"] == 0 and res["error_count"] == 0, f"Expected zero counts for empty input, got: {res}"


def test_upsert_batch_calls_execute_values_and_returns_counts(patch_execute_values_to_fake_pool):
    ds = DataStorage()

    df = pd.DataFrame([
        {
            "ticker": "T1",
            "date": "2025-03-01",
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 100,
            "adjusted_close": 1.5,
            "vwap": 1.6,
        }
    ])

    # patch_execute_values_to_fake_pool fixture replaces execute_values with a wrapper
    inserted, updated = ds._upsert_batch(df)
    assert inserted == 1 and updated == 0, f"Unexpected upsert counts: inserted={inserted}, updated={updated}"


def test_get_historical_data_builds_query_and_returns_rows(mocker):
    ds = DataStorage()
    fake_rows = [{"ticker": "T1", "date": "2025-03-01"}]

    mocker.patch("src.data_collector.polygon_data.data_storage.fetch_all", return_value=fake_rows)

    rows = ds.get_historical_data("T1", start_date="2025-03-01", end_date="2025-03-02", limit=10)
    assert rows == fake_rows, f"Expected returned rows to match mocked fetch_all result; got {rows}"



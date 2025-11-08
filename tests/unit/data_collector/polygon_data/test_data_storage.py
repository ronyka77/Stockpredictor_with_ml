from datetime import date

from src.data_collector.polygon_data.data_storage import DataStorage


def test_store_historical_no_records():
    """Return zero counts when no historical records provided"""
    ds = DataStorage()

    res = ds.store_historical_data([])
    assert res["stored_count"] == 0
    assert res["error_count"] == 0


def test_store_historical_batches_and_upsert(patch_execute_values_to_fake_pool, poly_ohlcv_factory):
    """Store batches and upsert into DB, returning counts"""
    ds = DataStorage()

    # prepare records -> single batch using shared polyfactory model
    recs = [
        poly_ohlcv_factory.build(ticker="AAA", timestamp=date(2020, 1, 1)),
        poly_ohlcv_factory.build(ticker="AAA", timestamp=date(2020, 1, 2)),
    ]
    res = ds.store_historical_data(recs, batch_size=10)
    assert res["stored_count"] == 2
    assert res["error_count"] == 0


def test_get_historical_query_and_params(mocker):
    """Patch fetch_all to return deterministic rows and validate query results"""
    fake_rows = [{"ticker": "AAA", "date": date(2020, 1, 1), "close": 10}]
    mocker.patch("src.data_collector.polygon_data.data_storage.fetch_all", return_value=fake_rows)

    ds = DataStorage()
    out = ds.get_historical_data(
        "aaa", start_date=date(2020, 1, 1), end_date=date(2020, 1, 2), limit=1
    )

    assert isinstance(out, list)
    assert out[0]["ticker"] == "AAA"


def test_insert_ignore_and_upsert_batch_calls_execute_values(mocker, ohlcv_df):
    """Ensure execute_values is invoked by insert and upsert batch helpers"""
    ds = DataStorage()

    # Patch execute_values used by insert/ upsert helpers
    ev = mocker.patch("src.data_collector.polygon_data.data_storage.execute_values", autospec=True)

    # _insert_ignore_batch should return count equal to rows
    count = ds._insert_ignore_batch(ohlcv_df)
    assert count == len(ohlcv_df)

    # _upsert_batch should return (inserted, updated)
    ev.reset_mock()
    inserted, updated = ds._upsert_batch(ohlcv_df)
    assert inserted == len(ohlcv_df)
    assert updated == 0


def test_store_historical_data_returns_zero_on_empty_list():
    """store_historical_data returns zero counts for empty input"""
    ds = DataStorage()
    res = ds.store_historical_data([])
    assert res["stored_count"] == 0 and res["error_count"] == 0, (
        f"Expected zero counts for empty input, got: {res}"
    )


def test_upsert_batch_calls_execute_values_and_returns_counts(
    patch_execute_values_to_fake_pool, ohlcv_df
):
    """_upsert_batch invokes execute_values and returns inserted/updated counts"""
    ds = DataStorage()
    df = ohlcv_df.copy()

    # patch_execute_values_to_fake_pool fixture replaces execute_values with a wrapper
    inserted, updated = ds._upsert_batch(df)
    assert inserted == 1 and updated == 0, (
        f"Unexpected upsert counts: inserted={inserted}, updated={updated}"
    )


def test_get_historical_data_builds_query_and_returns_rows(mocker):
    """get_historical_data builds SQL and returns rows from fetch_all"""
    ds = DataStorage()
    fake_rows = [{"ticker": "T1", "date": "2025-03-01"}]

    mocker.patch("src.data_collector.polygon_data.data_storage.fetch_all", return_value=fake_rows)

    rows = ds.get_historical_data("T1", start_date="2025-03-01", end_date="2025-03-02", limit=10)
    assert rows == fake_rows, f"Expected returned rows to match mocked fetch_all result; got {rows}"

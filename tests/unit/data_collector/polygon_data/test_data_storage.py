from datetime import date

from src.data_collector.polygon_data.data_storage import DataStorage


def test_store_historical_no_records():
    ds = DataStorage()

    res = ds.store_historical_data([])
    assert res["stored_count"] == 0
    assert res["error_count"] == 0


def test_store_historical_batches_and_upsert(patch_execute_values_to_fake_pool, poly_ohlcv_factory):
    ds = DataStorage()

    # prepare records -> single batch using shared polyfactory model
    recs = [
        poly_ohlcv_factory.build(ticker="AAA", timestamp=date(2020, 1, 1)),
        poly_ohlcv_factory.build(ticker="AAA", timestamp=date(2020, 1, 2)),
    ]
    res = ds.store_historical_data(recs, batch_size=10, on_conflict="update")
    assert res["stored_count"] == 2
    assert res["error_count"] == 0


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

    assert isinstance(out, list)
    assert out[0]["ticker"] == "AAA"


def test_insert_ignore_and_upsert_batch_calls_execute_values(mocker, ohlcv_df):
    ds = DataStorage()

    # Patch execute_values used by insert/ upsert helpers
    ev = mocker.patch(
        "src.data_collector.polygon_data.data_storage.execute_values",
        autospec=True,
    )

    # _insert_ignore_batch should return count equal to rows
    count = ds._insert_ignore_batch(ohlcv_df)
    assert count == len(ohlcv_df)

    # _upsert_batch should return (inserted, updated)
    ev.reset_mock()
    inserted, updated = ds._upsert_batch(ohlcv_df)
    assert inserted == len(ohlcv_df)
    assert updated == 0

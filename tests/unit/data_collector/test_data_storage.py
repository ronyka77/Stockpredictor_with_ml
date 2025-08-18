from datetime import date
from unittest.mock import MagicMock, patch

from src.data_collector.polygon_data.data_storage import DataStorage
from src.data_collector.polygon_data.data_validator import OHLCVRecord


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


@patch("src.data_collector.polygon_data.data_storage.create_engine")
def test_store_historical_no_records(mock_create_engine):
    # create storage with mocked engine to avoid real DB connection
    mock_engine = MagicMock()
    mock_create_engine.return_value = mock_engine
    ds = DataStorage(connection_string="sqlite:///:memory:")

    res = ds.store_historical_data([])
    assert res["stored_count"] == 0
    assert res["error_count"] == 0


@patch("src.data_collector.polygon_data.data_storage.create_engine")
def test_store_historical_batches_and_upsert(mock_create_engine):
    mock_engine = MagicMock()
    # ensure .connect() context manager works
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_create_engine.return_value = mock_engine

    ds = DataStorage(connection_string="sqlite:///:memory:")

    # prepare 3 records -> single batch
    recs = [make_record("AAA", date(2020, 1, 1)), make_record("AAA", date(2020, 1, 2))]
    res = ds.store_historical_data(recs, batch_size=10, on_conflict="update")

    assert res["stored_count"] == 2
    assert res["error_count"] == 0


@patch("src.data_collector.polygon_data.data_storage.create_engine")
def test_get_historical_query_and_params(mock_create_engine):
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    # simulate returned rows
    row_mock = MagicMock()
    row_mock._mapping = {"ticker": "AAA", "date": date(2020, 1, 1), "close": 10}
    mock_conn.execute.return_value = [row_mock]
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_create_engine.return_value = mock_engine

    ds = DataStorage(connection_string="sqlite:///:memory:")
    out = ds.get_historical_data("aaa", start_date=date(2020, 1, 1), end_date=date(2020, 1, 2), limit=1)

    assert isinstance(out, list)
    assert out[0]["ticker"] == "AAA"



import pandas as pd
import pytest

from src.data_collector.polygon_data.data_storage import DataStorage
from src.feature_engineering.data_loader import StockDataLoader


def test_data_storage_store_historical_data_empty_records(mocker):
    # Patch create_engine to prevent real DB connection during DataStorage init
    class DummyConn:
        def execute(self, *a, **k):
            class R:
                def fetchone(self):
                    return (1,)

            return R()

        def commit(self):
            pass

    class DummyEngine:
        def connect(self):
            from contextlib import contextmanager

            @contextmanager
            def _cm():
                yield DummyConn()

            return _cm()

    mocker.patch(
        "src.data_collector.polygon_data.data_storage.create_engine",
        return_value=DummyEngine(),
    )
    ds = DataStorage(connection_string="sqlite:///:memory:")
    res = ds.store_historical_data([], batch_size=10)
    if not (res["stored_count"] == 0 and res["error_count"] == 0):
        raise AssertionError(f"Expected no records stored, got {res}")


def test_stock_data_loader_load_returns_empty_dataframe_when_no_data(mocker):
    cfg = {"host": "h", "port": 5432, "database": "db", "user": "u", "password": "p"}

    # Patch create_engine to avoid real DB
    class DummyEngine:
        pass

    mocker.patch(
        "src.feature_engineering.data_loader.create_engine", return_value=DummyEngine()
    )
    # Patch pandas.read_sql_query to return empty df
    mocker.patch("pandas.read_sql_query", return_value=pd.DataFrame())
    loader = StockDataLoader(db_config=cfg)
    df = loader.load_stock_data("TEST", "2020-01-01", "2020-01-03")
    if not df.empty:
        raise AssertionError("Expected empty DataFrame when no data returned from database")


def test_validate_and_clean_data_missing_columns_raises(mocker):
    cfg = {"host": "h", "port": 5432, "database": "db", "user": "u", "password": "p"}
    mocker.patch(
        "src.feature_engineering.data_loader.create_engine", return_value=object()
    )
    loader = StockDataLoader(db_config=cfg)
    df = pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], "open": [1]})
    with pytest.raises(ValueError, match="Missing required columns"):
        loader._validate_and_clean_data(df, "TEST")

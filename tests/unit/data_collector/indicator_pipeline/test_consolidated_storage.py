from types import SimpleNamespace
import pandas as pd
from pathlib import Path
from unittest.mock import patch
import pytest
from datetime import date

from src.data_collector.indicator_pipeline.consolidated_storage import (
    ConsolidatedFeatureStorage,
)


def make_sample_ticker_df():
    idx = pd.date_range("2025-01-01", periods=4)
    # Ensure index has name 'date' so reset_index creates a 'date' column
    idx.name = "date"
    df = pd.DataFrame({"a": [1, 2, 3, 4]}, index=idx)
    return df


def test_combine_ticker_data_and_year_partitioning(tmp_path):
    """Combine ticker data and write year-partitioned consolidated files"""
    storage = ConsolidatedFeatureStorage()
    # Override paths to tmp
    storage.base_path = tmp_path
    storage.version_path = tmp_path / storage.config.version
    storage.consolidated_path = storage.version_path / "consolidated"
    storage.consolidated_path.mkdir(parents=True, exist_ok=True)

    # Ensure dataframes have a date column after reset_index in combine
    a = make_sample_ticker_df()
    b = make_sample_ticker_df()
    tdata = {"AAA": a, "BBB": b}

    # Patch pyarrow write to avoid real IO and Path.stat to avoid file existence checks
    with (
        patch("pyarrow.parquet.write_table", return_value=None),
        patch.object(
            Path,
            "stat",
            return_value=SimpleNamespace(
                st_size=int(0.5 * 1024 * 1024), st_ctime=0, st_mtime=0
            ),
        ),
    ):
        result = storage.save_multiple_tickers(tdata, metadata={})

    assert "files_created" in result
    assert result["total_tickers"] == 2


def test_save_multiple_tickers_creates_files_and_returns_summary(
    tmp_path, mock_pipeline_logger
):
    storage = ConsolidatedFeatureStorage()
    storage.base_path = tmp_path
    storage.version_path = tmp_path / storage.config.version
    storage.consolidated_path = storage.version_path / "consolidated"
    storage.consolidated_path.mkdir(parents=True, exist_ok=True)

    a = make_sample_ticker_df()
    b = make_sample_ticker_df()
    tdata = {"AAA": a, "BBB": b}

    # Patch pyarrow write and Path.stat
    with (
        patch("pyarrow.parquet.write_table", return_value=None) as write_table,
        patch(  # pyright: ignore[reportUndefinedVariable]
            "pathlib.Path.stat",
            return_value=SimpleNamespace(
                st_size=int(0.5 * 1024 * 1024), st_ctime=0, st_mtime=0
            ),
        ),
    ):
        result = storage.save_multiple_tickers(tdata, metadata={})

    assert "files_created" in result
    assert result["total_tickers"] == 2
    assert write_table.called


def test_save_multiple_tickers_handles_small_files(tmp_path, mock_pipeline_logger):
    storage = ConsolidatedFeatureStorage()
    storage.base_path = tmp_path
    storage.version_path = tmp_path / storage.config.version
    storage.consolidated_path = storage.version_path / "consolidated"
    storage.consolidated_path.mkdir(parents=True, exist_ok=True)

    a = make_sample_ticker_df()
    b = make_sample_ticker_df()
    tdata = {"AAA": a, "BBB": b}

    with (
        patch("pyarrow.parquet.write_table", return_value=None) as write_table,
        patch(
            "pathlib.Path.stat",
            return_value=SimpleNamespace(st_size=10, st_ctime=0, st_mtime=0),
        ),
    ):
        result = storage.save_multiple_tickers(tdata, metadata={})

    assert result["total_tickers"] == 2
    assert write_table.called


def test_save_multiple_tickers_unsupported_strategy_raises(tmp_path):
    storage = ConsolidatedFeatureStorage()
    storage.base_path = tmp_path
    storage.version_path = tmp_path / storage.config.version
    storage.consolidated_path = storage.version_path / "consolidated"
    storage.consolidated_path.mkdir(parents=True, exist_ok=True)

    # force unsupported partitioning strategy
    storage.config.partitioning_strategy = "unsupported"

    a = make_sample_ticker_df()
    tdata = {"AAA": a}

    with pytest.raises(ValueError):
        storage.save_multiple_tickers(tdata, metadata={})


def test_build_parquet_filters_combinations():
    storage = ConsolidatedFeatureStorage()

    # ticker only
    f = storage._build_parquet_filters("AAA", None, None)
    assert f == [("ticker", "==", "AAA")]

    # start and end dates
    f = storage._build_parquet_filters(None, date(2025, 1, 1), date(2025, 1, 31))
    assert ("date", ">=", pd.Timestamp("2025-01-01")) in f
    assert ("date", "<=", pd.Timestamp("2025-01-31")) in f


def test_combine_ticker_data_empty_and_non_datetime_index(tmp_path):
    storage = ConsolidatedFeatureStorage()

    # empty input returns empty DataFrame
    res = storage._combine_ticker_data({})
    assert isinstance(res, pd.DataFrame) and res.empty

    # non-datetime index but with name should still reset_index
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.index.name = "date"
    res = storage._combine_ticker_data({"T": df})
    assert "ticker" in res.columns and "date" in res.columns


def test_calculate_compression_ratio_zero_and_positive():
    storage = ConsolidatedFeatureStorage()
    df = pd.DataFrame({"a": [1, 2], "date": pd.date_range("2025-01-01", periods=2)})

    # zero compressed size -> return 1.0
    assert storage._calculate_compression_ratio(df, 0) == 1.0

    # positive compressed size should return float > 0
    ratio = storage._calculate_compression_ratio(df, 0.01)
    assert isinstance(ratio, float) and ratio > 0


def test_load_consolidated_features_no_files_raises(tmp_path):
    storage = ConsolidatedFeatureStorage()
    storage.base_path = tmp_path
    storage.version_path = tmp_path / storage.config.version
    storage.consolidated_path = storage.version_path / "consolidated"
    storage.consolidated_path.mkdir(parents=True, exist_ok=True)

    # ensure no parquet files exist
    for p in storage.consolidated_path.glob("*.parquet"):
        p.unlink()

    with pytest.raises(FileNotFoundError):
        storage.load_consolidated_features()


def test_load_consolidated_features_reads_files_and_applies_filters(tmp_path):
    # Create storage and a dummy parquet file path
    storage = ConsolidatedFeatureStorage()
    storage.base_path = tmp_path
    storage.version_path = tmp_path / storage.config.version
    storage.consolidated_path = storage.version_path / "consolidated"
    storage.consolidated_path.mkdir(parents=True, exist_ok=True)

    file_path = storage.consolidated_path / "features_2025.parquet"
    file_path.write_bytes(b"")

    # Build a dataframe that will be returned by read_parquet
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "feat1": [1, 2],
        }
    )

    # Patch pd.read_parquet to return our df and filter_columns_by_categories to return feat1
    with (
        patch.object(pd, "read_parquet", lambda *a, **k: df),
        patch(
            "src.data_collector.indicator_pipeline.consolidated_storage.filter_columns_by_categories",
            lambda cols, cats: [c for c in cols if c.startswith("feat")],
        ),
    ):
        # No filters -> should return combined data for all tickers
        res = storage.load_consolidated_features()
    assert isinstance(res, pd.DataFrame) and len(res) == 2

    # With ticker filter that does not match -> warning and empty DF
    res = storage.load_consolidated_features(ticker="ZZZ")
    assert isinstance(res, pd.DataFrame) and res.empty


def test_consolidate_existing_features_handles_no_available_and_success(
    tmp_path, mock_pipeline_logger
):
    # Patch FeatureStorage to simulate no available tickers
    class FakeStorageEmpty:
        def get_available_tickers(self):
            return []

    with patch(
        "src.data_collector.indicator_pipeline.consolidated_storage.FeatureStorage",
        FakeStorageEmpty,
    ):
        from src.data_collector.indicator_pipeline.consolidated_storage import (
            consolidate_existing_features,
        )

        with pytest.raises(ValueError):
            consolidate_existing_features()

    # Now simulate success path
    class FakeMeta:
        def __init__(self):
            self.feature_version = "v1"
            self.calculation_date = "2025-01-01"
            self.quality_score = 0.9
            self.total_features = 5
            self.file_size_mb = 0.1
            self.record_count = 2
            self.feature_categories = None
            self.warnings = None

    class FakeStorage:
        def get_available_tickers(self):
            return ["AAA"]

        def load_features(self, ticker):
            df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"]), "a": [1]})
            return df, FakeMeta()

    # Create a fake consolidated storage that avoids filesystem mkdir/write
    class FakeConsolidated:
        def __init__(self):
            pass

        def save_multiple_tickers(self, ticker_data, metadata):
            return {
                "files_created": 1,
                "total_size_mb": 0.1,
                "compression_ratio": 1.0,
                "files": [],
            }

    with (
        patch(
            "src.data_collector.indicator_pipeline.consolidated_storage.FeatureStorage",
            FakeStorage,
        ),
        patch(
            "src.data_collector.indicator_pipeline.consolidated_storage.ConsolidatedFeatureStorage",
            FakeConsolidated,
        ),
        patch("pyarrow.parquet.write_table", lambda *a, **k: None),
    ):
        from src.data_collector.indicator_pipeline.consolidated_storage import (
            consolidate_existing_features,
        )

        res = consolidate_existing_features()
        assert "files_created" in res or "files" in res


def test_main_handles_failure_and_success():
    # Failure path
    with patch(
        "src.data_collector.indicator_pipeline.consolidated_storage.consolidate_existing_features",
        lambda strategy="by_date": (_ for _ in ()).throw(ValueError("no files")),
    ):
        from src.data_collector.indicator_pipeline.consolidated_storage import main

        assert main() == 1

    # Success path: patch consolidate_existing_features to return a minimal result
    with patch(
        "src.data_collector.indicator_pipeline.consolidated_storage.consolidate_existing_features",
        lambda strategy="by_date": {
            "files_created": 1,
            "total_size_mb": 0.1,
            "compression_ratio": 1.0,
            "files": [],
        },
    ):
        from src.data_collector.indicator_pipeline.consolidated_storage import main

        assert main() == 0

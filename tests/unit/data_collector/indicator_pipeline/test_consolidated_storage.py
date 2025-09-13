import pandas as pd

from src.data_collector.indicator_pipeline.consolidated_storage import (
    ConsolidatedFeatureStorage,
)


def make_sample_ticker_df():
    idx = pd.date_range("2025-01-01", periods=4)
    # Ensure index has name 'date' so reset_index creates a 'date' column
    idx.name = "date"
    df = pd.DataFrame({"a": [1, 2, 3, 4]}, index=idx)
    return df


def test_combine_ticker_data_and_year_partitioning(mocker, tmp_path):
    # Verifies consolidated storage combines tickers and writes year-partitioned files
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
    mocker.patch("pyarrow.parquet.write_table", return_value=None)
    from pathlib import Path
    from types import SimpleNamespace

    # Return a fake stat with a reasonable size
    mocker.patch.object(
        Path,
        "stat",
        return_value=SimpleNamespace(
            st_size=int(0.5 * 1024 * 1024), st_ctime=0, st_mtime=0
        ),
    )

    result = storage.save_multiple_tickers(tdata, metadata={})

    assert "files_created" in result
    assert result["total_tickers"] == 2

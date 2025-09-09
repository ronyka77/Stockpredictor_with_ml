import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from src.data_collector.indicator_pipeline.feature_storage import (
    FeatureStorage,
    StorageConfig,
)


def test_parquet_roundtrip(tmp_path):
    # Prepare sample DataFrame with datetime index
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame({"sma": [1.0, 2.0, 3.0], "rsi": [30.0, 40.0, 50.0]}, index=idx)
    # The storage system resets index to a 'date' column and restores it as index on load,
    # so name the index to match round-trip behavior.
    df.index.name = "date"

    # Prepare storage config to use temporary directory
    config = StorageConfig(
        base_path=str(tmp_path),
        version="vtest",
        compression=None,
        engine="pyarrow",
        row_group_size=64,
        cleanup_old_versions=False,
        max_versions=3,
    )

    fs = FeatureStorage(config=config)

    metadata = {"categories": ["trend", "momentum"], "quality_score": 0.95}

    # Save features
    feature_meta = fs.save_features("TICK", df, metadata)

    # Load features back
    loaded_df, loaded_meta = fs.load_features("TICK")

    # Dataframe round-trip: same shape, index values and columns
    if not np.array_equal(df.index.values, loaded_df.index.values):
        raise AssertionError(f"Index values differ: {df.index} vs {loaded_df.index}")
    if list(loaded_df.columns) != list(df.columns):
        raise AssertionError("Loaded columns differ from saved columns")
    # Numeric equality check (ignore index/column name metadata and frequency)
    assert_frame_equal(df, loaded_df, check_names=False, check_freq=False)

    # Metadata sanity checks
    if feature_meta.ticker != "TICK":
        raise AssertionError("Feature meta ticker mismatch")
    if feature_meta.record_count != len(df):
        raise AssertionError("Feature meta record_count mismatch")
    if feature_meta.total_features != len(df.columns):
        raise AssertionError("Feature meta total_features mismatch")

    # Available tickers and storage stats should reflect saved file
    tickers = fs.get_available_tickers()
    if "TICK" not in tickers:
        raise AssertionError("Available tickers missing expected 'TICK'")

    stats = fs.get_storage_stats()
    if not isinstance(stats, dict):
        raise AssertionError("Storage stats should be a dict")
    if stats["version"] != config.version:
        raise AssertionError("Storage stats version mismatch")

import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from datetime import datetime

import pytest
from types import SimpleNamespace
from unittest.mock import patch

from src.data_collector.indicator_pipeline.feature_storage import (
    FeatureStorage,
    StorageConfig,
    FeatureMetadata,
)


def make_df():
    idx = pd.date_range("2025-01-01", periods=3)
    return pd.DataFrame({"f1": [1, 2, 3]}, index=idx)


def test_save_features_returns_metadata_with_expected_fields(mocker, tmp_path):
    """Check FeatureStorage.save_features returns FeatureMetadata with expected fields"""
    df = make_df()

    # Create a custom config to place files in tmp_path
    cfg = StorageConfig(
        base_path=str(tmp_path),
        version="vtest",
        compression="NONE",
        engine="pyarrow",
        row_group_size=64,
        cleanup_old_versions=False,
        max_versions=1,
    )

    storage = FeatureStorage(config=cfg)

    # Patch internals that would perform IO
    mocker.patch.object(storage, "_save_parquet", return_value=None)
    mocker.patch.object(storage, "_save_metadata_to_parquet", return_value=None)
    mocker.patch.object(
        storage,
        "_get_file_stats",
        return_value={
            "size_mb": 0.1,
            "created": datetime.now(),
            "modified": datetime.now(),
        },
    )

    metadata = {"categories": ["trend"], "quality_score": 95.0, "warnings": []}

    feat_meta = storage.save_features("TICK", df, metadata)

    assert isinstance(feat_meta, FeatureMetadata)
    assert feat_meta.ticker == "TICK"
    assert feat_meta.quality_score == pytest.approx(95.0)
    assert feat_meta.total_features == 1
    # file_path should be relative to base_path
    assert isinstance(feat_meta.file_path, str)


def test_save_features_success_calls_parquet_and_returns_metadata(
    tmp_path, mock_pipeline_logger
):
    cfg = StorageConfig(
        base_path=str(tmp_path),
        version="vtest",
        compression="NONE",
        engine="pyarrow",
        row_group_size=64,
        cleanup_old_versions=False,
        max_versions=1,
    )
    storage = FeatureStorage(config=cfg)

    df = make_df()
    metadata = {"categories": ["trend"], "quality_score": 95.0, "warnings": []}

    with (
        patch("pyarrow.parquet.write_table", return_value=None) as write_table,
        patch(
            "pathlib.Path.stat",
            return_value=SimpleNamespace(
                st_size=int(0.5 * 1024 * 1024), st_ctime=0, st_mtime=0
            ),
        ),
    ):
        feat_meta = storage.save_features("TICK", df, metadata)

    assert isinstance(feat_meta, FeatureMetadata)
    assert feat_meta.ticker == "TICK"
    assert feat_meta.quality_score == 95.0
    assert write_table.called


def test_save_features_small_file_triggers_metadata_handling(
    tmp_path, mock_pipeline_logger
):
    cfg = StorageConfig(
        base_path=str(tmp_path),
        version="vtest",
        compression="NONE",
        engine="pyarrow",
        row_group_size=64,
        cleanup_old_versions=False,
        max_versions=1,
    )
    storage = FeatureStorage(config=cfg)

    df = make_df()
    metadata = {"categories": ["trend"], "quality_score": 60.0, "warnings": []}

    # Small file size
    with (
        patch("pyarrow.parquet.write_table", return_value=None) as write_table,
        patch(
            "pathlib.Path.stat",
            return_value=SimpleNamespace(st_size=10, st_ctime=0, st_mtime=0),
        ),
    ):
        feat_meta = storage.save_features("TICK", df, metadata)

    assert feat_meta.total_features == 1
    assert isinstance(feat_meta.file_path, str)
    assert write_table.called


def test_save_features_pyarrow_raises_and_logged(tmp_path, mock_pipeline_logger):
    cfg = StorageConfig(
        base_path=str(tmp_path),
        version="vtest",
        compression="NONE",
        engine="pyarrow",
        row_group_size=64,
        cleanup_old_versions=False,
        max_versions=1,
    )
    storage = FeatureStorage(config=cfg)

    df = make_df()
    metadata = {"categories": ["trend"], "quality_score": 50.0, "warnings": []}

    # Patch the module-level logger used by feature_storage and the pyarrow writer
    with (
        patch(
            "pyarrow.parquet.write_table", side_effect=RuntimeError("io fail")
        ) as write_table,
        patch(
            "src.data_collector.indicator_pipeline.feature_storage.logger"
        ) as fs_logger,
    ):
        with pytest.raises(RuntimeError):
            _ = storage.save_features("TICK", df, metadata)

    # logger should have an error call recorded on the feature_storage logger
    fs_logger.error.assert_called()
    assert write_table.called

def test_parquet_roundtrip(tmp_path):
    """Save and load features via parquet round-trip preserving data and metadata."""
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

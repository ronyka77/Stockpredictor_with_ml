import pandas as pd
from datetime import datetime

import pytest

from src.data_collector.indicator_pipeline.feature_storage import (
    FeatureStorage,
    StorageConfig,
    FeatureMetadata,
)


def test_save_features_returns_metadata_with_expected_fields(mocker, tmp_path):
    # Checks FeatureStorage.save_features returns FeatureMetadata with correct fields
    idx = pd.date_range("2025-01-01", periods=3)
    df = pd.DataFrame({"f1": [1, 2, 3]}, index=idx)

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

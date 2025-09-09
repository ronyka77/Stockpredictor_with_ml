import pandas as pd
import numpy as np
import pytest

from src.data_collector.indicator_pipeline.feature_storage import (
    FeatureStorage,
    StorageConfig,
)


@pytest.mark.integration
def test_feature_storage_roundtrip(tmp_path):
    cfg = StorageConfig()
    cfg.base_path = str(tmp_path)
    cfg.version = "v1"
    fs = FeatureStorage(config=cfg)

    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "SMA_5": np.linspace(1, 5, 5),
            "RSI_14": np.linspace(30, 70, 5),
        },
        index=idx,
    )

    meta = {"categories": ["trend", "momentum"], "quality_score": 95.0, "warnings": []}
    m = fs.save_features("TEST", df, meta)
    if not (m.ticker == "TEST" and m.total_features == 2):
        raise AssertionError("Saved feature metadata does not match expected values")

    loaded_df, loaded_meta = fs.load_features("TEST")
    if loaded_df.shape != df.shape:
        raise AssertionError("Loaded DataFrame shape does not match original")
    if set(loaded_df.columns) != set(df.columns):
        raise AssertionError("Loaded DataFrame columns do not match original")

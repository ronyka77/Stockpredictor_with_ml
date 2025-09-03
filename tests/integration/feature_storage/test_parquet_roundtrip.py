import pandas as pd
import numpy as np
import pytest

from src.data_collector.indicator_pipeline.feature_storage import FeatureStorage, StorageConfig


@pytest.mark.integration
def test_feature_storage_roundtrip(tmp_path):
    cfg = StorageConfig()
    cfg.base_path = str(tmp_path)
    cfg.version = "v1"
    fs = FeatureStorage(config=cfg)

    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "SMA_5": np.linspace(1, 5, 5),
        "RSI_14": np.linspace(30, 70, 5),
    }, index=idx)

    meta = {"categories": ["trend", "momentum"], "quality_score": 95.0, "warnings": []}
    m = fs.save_features("TEST", df, meta)
    assert m.ticker == "TEST" and m.total_features == 2

    loaded_df, loaded_meta = fs.load_features("TEST")
    assert loaded_df.shape == df.shape
    assert set(loaded_df.columns) == set(df.columns)



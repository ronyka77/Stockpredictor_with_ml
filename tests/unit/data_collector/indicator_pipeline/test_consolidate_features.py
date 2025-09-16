from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.mark.unit
def test_main_returns_none_when_no_available_tickers():
    """Return None when there are no tickers to consolidate"""
    from src.data_collector.indicator_pipeline import consolidate_features as mod

    fake_storage = MagicMock()
    fake_storage.get_available_tickers.return_value = []
    fake_storage.get_storage_stats.return_value = {
        "total_size_mb": 0.0,
        "base_path": "features",
    }

    with patch.object(mod, "FeatureStorage", return_value=fake_storage):
        result = mod.main()

    assert result is None


@pytest.mark.unit
def test_main_success_with_sampling_and_consolidation():
    """Ensure consolidation returns expected file summary when sampling succeeds"""
    from src.data_collector.indicator_pipeline import consolidate_features as mod

    # Prepare fake storage with 3 tickers; one load raises to hit warning path
    fake_storage = MagicMock()
    fake_storage.get_available_tickers.return_value = ["AAPL", "MSFT", "GOOG"]
    fake_storage.get_storage_stats.return_value = {
        "total_size_mb": 123.45,
        "base_path": "features",
    }

    idx = pd.date_range("2021-01-01", periods=10, freq="D")
    df = pd.DataFrame({"a": range(10)}, index=idx)
    fake_storage.load_features.side_effect = [
        (df, {"ticker": "AAPL"}),
        Exception("load error"),
        (df, {"ticker": "GOOG"}),
    ]

    fake_result = {
        "files_created": 5,
        "total_size_mb": 50.0,
        "total_rows": 10000,
        "compression_ratio": 2.0,
        "files": [
            {
                "file": "features/2021.parquet",
                "rows": 6000,
                "size_mb": 30.0,
                "year": 2021,
            },
            {
                "file": "features/2022.parquet",
                "rows": 4000,
                "size_mb": 20.0,
                "year": 2022,
            },
        ],
    }

    with (
        patch.object(mod, "FeatureStorage", return_value=fake_storage),
        patch.object(mod, "consolidate_existing_features", return_value=fake_result),
    ):
        result = mod.main()

    assert result == fake_result


@pytest.mark.unit
def test_main_handles_consolidation_exception_and_returns_none():
    """Swallow consolidation exceptions and return None to indicate failure"""
    from src.data_collector.indicator_pipeline import consolidate_features as mod

    fake_storage = MagicMock()
    fake_storage.get_available_tickers.return_value = ["AAPL"]
    fake_storage.get_storage_stats.return_value = {
        "total_size_mb": 10.0,
        "base_path": "features",
    }
    # For sampling path
    idx = pd.date_range("2021-01-01", periods=3, freq="D")
    df = pd.DataFrame({"a": range(3)}, index=idx)
    fake_storage.load_features.return_value = (df, {"ticker": "AAPL"})

    with (
        patch.object(mod, "FeatureStorage", return_value=fake_storage),
        patch.object(
            mod, "consolidate_existing_features", side_effect=RuntimeError("boom")
        ),
    ):
        # Should swallow exception and return None
        result = mod.main()

    assert result is None

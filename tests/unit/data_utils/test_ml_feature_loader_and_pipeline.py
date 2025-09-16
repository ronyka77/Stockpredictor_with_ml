import pandas as pd
import pytest

from src.data_utils import ml_feature_loader as mfl
from src.data_utils import ml_data_pipeline as mp


def test__load_from_consolidated_uses_target_and_drops_future_cols(
    mocker, features_with_future
):
    """
    Verify that loading consolidated features:
    - returns targets as a pandas Series,
    - removes any future-labelled feature columns (e.g., `Future_Close_10D`) from the feature frame,
    - and preserves one-to-one alignment between features and targets with a non-empty result.
    
    This test patches the loader's consolidated storage to return a DataFrame that includes future horizon columns, calls `_load_from_consolidated(prediction_horizon=10)`, and asserts the outcomes above.
    """
    loader = mfl.MLFeatureLoader()
    fake_df = features_with_future
    # patch consolidated storage loader
    mocker.patch.object(
        loader.consolidated_storage, "load_consolidated_features", return_value=fake_df
    )
    features, targets = loader._load_from_consolidated(prediction_horizon=10)
    # targets should be Series with no NaNs (rows with NaN removed)
    if not isinstance(targets, pd.Series):
        raise AssertionError("Targets not returned as Series")
    if "Future_Close_10D" in features.columns:
        raise AssertionError("Future_* columns were not removed from features")
    if not (len(features) == len(targets) and len(features) > 0):
        raise AssertionError("Feature/target mismatch after cleaning")


def test_load_all_data_combines_years_and_maps_ticker_id(mocker):
    """Combine yearly feature frames and map ticker metadata to `ticker_id`."""
    # prepare per-year frames: only one year has data
    df2024 = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": ["2024-01-01", "2024-01-02"],
            "close": [1, 2],
        }
    )
    # patch load_yearly_data to return df for 2024 and empty for others
    mocker.patch(
        "src.data_utils.ml_feature_loader.load_yearly_data",
        side_effect=[pd.DataFrame(), df2024, pd.DataFrame()],
    )
    # patch StockDataLoader.get_ticker_metadata to return mapping
    fake_meta = pd.DataFrame({"ticker": ["AAA", "BBB"], "id": [10, 20]})
    mocker.patch(
        "src.data_utils.ml_feature_loader.StockDataLoader.get_ticker_metadata",
        return_value=fake_meta,
    )
    out = mfl.load_all_data(ticker=None)
    # Expect combined DataFrame with ticker_id mapped
    if out.empty:
        raise AssertionError(
            "load_all_data returned empty when it should have combined data"
        )
    if "ticker_id" not in out.columns:
        raise AssertionError("ticker_id column missing after metadata mapping")
    # date converted to datetime in loader
    if not pd.api.types.is_datetime64_any_dtype(out["date"]):
        raise AssertionError("Date column not converted to datetime")


def test_prepare_ml_data_for_training_raises_on_missing_date_column(mocker):
    """Raise ValueError when loaded data is missing the 'date' column."""
    # load_all_data returns DataFrame without 'date' -> expect ValueError when splitting
    mocker.patch(
        "src.data_utils.ml_data_pipeline.load_all_data",
        return_value=pd.DataFrame({"close": [1, 2, 3]}),
    )
    with pytest.raises(ValueError, match="'date' column not found"):
        mp.prepare_ml_data_for_training()


def test_prepare_ml_data_for_training_with_cleaning_uses_cache_when_available(mocker):
    """Return cached cleaned training payload when cache exists and is fresh."""
    # patch cache to simulate existing valid cache
    fake_cached = {
        "X_train": pd.DataFrame([[1.0]]),
        "X_test": pd.DataFrame([[2.0]]),
        "y_train": pd.Series([0.1]),
        "y_test": pd.Series([0.2]),
        "feature_count": 1,
    }
    mocker.patch.object(mp, "_cleaned_data_cache", create=True)
    mp._cleaned_data_cache.cache_exists = lambda *a, **k: True
    mp._cleaned_data_cache.get_cache_age_hours = lambda *a, **k: 1.0
    mp._cleaned_data_cache.load_cleaned_data = lambda *a, **k: fake_cached
    res = mp.prepare_ml_data_for_training_with_cleaning()
    if res != fake_cached:
        raise AssertionError("Cached result not returned when cache present")

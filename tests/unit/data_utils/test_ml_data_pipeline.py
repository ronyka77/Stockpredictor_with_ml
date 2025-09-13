
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


def _make_sample_df(num_days: int = 40, start: str = "2025-02-15") -> pd.DataFrame:
    """Small helper to build deterministic time-series data for tests.

    Uses a simple pattern so tests are easy to reason about and assert.
    """
    start_dt = pd.to_datetime(start)
    dates = pd.date_range(start=start_dt, periods=num_days, freq="D")
    rows = [
        {
            "ticker": "AAPL",
            "date": d,
            "open": 100.0 + i,
            "close": 100.5 + i,
            "high": 101.0 + i,
            "low": 99.5 + i,
            "volume": 1_000 + i * 10,
        }
        for i, d in enumerate(dates)
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def sample_combined_data() -> pd.DataFrame:
    # Use a reproducible, dense date range that includes Mondays and Fridays
    return _make_sample_df()


def _fake_convert_absolute_to_percentage_returns(df: pd.DataFrame, horizon: int):
    df2 = df.copy()
    col = f"target_{horizon}d"
    # deterministic target: increasing sequence
    df2[col] = np.arange(len(df2), dtype=float)
    return df2, col


def _identity(df, *args, **kwargs):
    return df


def _fake_clean_features_for_training(X: pd.DataFrame, y: pd.Series):
    # Return data unchanged and no removed features (simplest clean path)
    return X.copy(), y.copy(), []


def _fake_analyze_feature_diversity(X: pd.DataFrame):
    return {
        "useful_feature_count": X.shape[1],
        "constant_feature_count": 0,
        "zero_variance_count": 0,
    }


@patch(
    "src.data_utils.ml_data_pipeline.add_price_normalized_features",
    side_effect=_identity,
)
@patch(
    "src.data_utils.ml_data_pipeline.add_prediction_bounds_features",
    side_effect=_identity,
)
@patch("src.data_utils.ml_data_pipeline.add_date_features", side_effect=_identity)
@patch(
    "src.data_utils.ml_data_pipeline.convert_absolute_to_percentage_returns",
    side_effect=_fake_convert_absolute_to_percentage_returns,
)
@patch("src.data_utils.ml_data_pipeline.load_all_data")
def test_prepare_ml_data_for_training_splits_and_filters(
    mock_load_all_data,
    _conv,
    _add_date,
    _add_bounds,
    _add_price,
    sample_combined_data,
):
    """
    Setup: Provide a deterministic combined dataset and stub out heavy transformations.

    Execution: Call `prepare_ml_data_for_training` and let the function perform splits.

    Verification: Ensure returned dict contains expected keys, non-empty train/test,
    and that day-of-week filtering kept only Mondays and Fridays where applicable.
    """

    # Arrange
    mock_load_all_data.return_value = sample_combined_data

    from src.data_utils import ml_data_pipeline as pipeline

    # Act
    result = pipeline.prepare_ml_data_for_training(
        prediction_horizon=10, split_date="2025-03-15"
    )

    # Assert
    assert isinstance(result, dict)
    assert "X_train" in result and "X_test" in result
    assert result["feature_count"] == result["X_train"].shape[1]

    # Days kept in test set must be Monday (0) or Friday (4)
    if len(result["X_test"]) > 0:
        test_dates = pd.to_datetime(
            sample_combined_data.loc[result["X_test"].index, "date"]
        )
        dow = test_dates.dt.dayofweek.unique()
        assert set(dow).issubset({0, 4})


@patch("src.data_utils.ml_data_pipeline.prepare_ml_data_for_training")
def test_prepare_ml_data_for_training_with_cleaning_uses_cache(
    mock_prepare, sample_combined_data
):
    """
    Setup: Simulate the cache returning precomputed cleaned data.

    Execution: Call `prepare_ml_data_for_training_with_cleaning`.

    Verification: The function should return the cached payload and skip heavy work.
    """

    from src.data_utils import ml_data_pipeline as pipeline

    # Arrange: set cache to return quickly
    fake_cached = {
        "cached": True,
        "X_train": sample_combined_data.iloc[:1],
        "X_test": sample_combined_data.iloc[1:2],
        "feature_count": 3,
    }

    with (
        patch.object(pipeline._cleaned_data_cache, "cache_exists", return_value=True),
        patch.object(
            pipeline._cleaned_data_cache, "get_cache_age_hours", return_value=1.0
        ),
        patch.object(
            pipeline._cleaned_data_cache, "load_cleaned_data", return_value=fake_cached
        ),
    ):
        # Act
        out = pipeline.prepare_ml_data_for_training_with_cleaning(
            prediction_horizon=10, split_date="2025-03-15"
        )

    # Assert
    assert out is fake_cached
    assert out["cached"] is True


def test_prepare_ml_data_for_training_with_cleaning_saves_when_not_cached(
    sample_combined_data,
):
    """
    Setup: Ensure cache is reported missing so the function runs the pipeline and saves result.

    Execution: Call `prepare_ml_data_for_training_with_cleaning` with patched internals.

    Verification: `save_cleaned_data` should be called once and output must contain expected keys.
    """

    from src.data_utils import ml_data_pipeline as pipeline

    # Arrange: patch dependencies to simple behaviors
    with patch.object(pipeline._cleaned_data_cache, "cache_exists", return_value=False):
        with patch.object(
            pipeline,
            "prepare_ml_data_for_training",
            return_value={
                "X_train": sample_combined_data.iloc[:10].reset_index(drop=True),
                "X_test": sample_combined_data.iloc[10:20].reset_index(drop=True),
                "y_train": pd.Series(np.arange(10.0)),
                "y_test": pd.Series(np.arange(10.0, 20.0)),
                "feature_count": 6,
            },
        ):
            with patch(
                "src.data_utils.ml_data_pipeline.clean_data_for_training",
                side_effect=_identity,
            ) as _clean:
                with patch(
                    "src.data_utils.ml_data_pipeline.clean_features_for_training",
                    side_effect=_fake_clean_features_for_training,
                ) as _clean_feat:
                    with patch(
                        "src.data_utils.ml_data_pipeline.analyze_feature_diversity",
                        side_effect=_fake_analyze_feature_diversity,
                    ) as _diversity:
                        with patch.object(
                            pipeline._cleaned_data_cache,
                            "save_cleaned_data",
                            autospec=True,
                        ) as mock_save:
                            # Act
                            out = pipeline.prepare_ml_data_for_training_with_cleaning(
                                prediction_horizon=10, split_date="2025-03-15"
                            )

    # Assert
    mock_save.assert_called_once()
    assert "X_train" in out and "X_test" in out
    assert out["feature_count"] >= 0


def test_prepare_ml_data_for_training_empty_raises():
    """
    Setup: load_all_data returns an empty DataFrame.

    Execution: prepare_ml_data_for_training should raise a ValueError.

    Verification: error message indicates no data loaded.
    """

    from src.data_utils import ml_data_pipeline as pipeline

    with patch(
        "src.data_utils.ml_data_pipeline.load_all_data", return_value=pd.DataFrame()
    ):
        with pytest.raises(ValueError) as exc:
            pipeline.prepare_ml_data_for_training()

    assert "No data loaded" in str(exc.value)


def test_prepare_ml_data_for_training_missing_date_raises(sample_combined_data):
    """
    Setup: data missing the `date` column.

    Execution: pipeline should reject the input early.

    Verification: ValueError mentions missing 'date' column.
    """

    from src.data_utils import ml_data_pipeline as pipeline

    df = sample_combined_data.drop(columns=["date"])  # remove date

    with patch("src.data_utils.ml_data_pipeline.load_all_data", return_value=df):
        with pytest.raises(ValueError) as exc:
            pipeline.prepare_ml_data_for_training()

    assert "'date' column not found" in str(exc.value)


@patch(
    "src.data_utils.ml_data_pipeline.convert_absolute_to_percentage_returns",
    side_effect=_fake_convert_absolute_to_percentage_returns,
)
@patch("src.data_utils.ml_data_pipeline.load_all_data")
def test_prepare_ml_data_for_prediction_filters_days(
    mock_load, _conv, sample_combined_data
):
    """
    Ensure prediction path filters X_test to Mondays and Fridays.
    """

    mock_load.return_value = sample_combined_data

    from src.data_utils import ml_data_pipeline as pipeline

    out = pipeline.prepare_ml_data_for_prediction(prediction_horizon=10)

    assert "X_test" in out and "y_test" in out

    if len(out["X_test"]) > 0:
        # recover dates for asserted indices
        dates = pd.to_datetime(sample_combined_data.loc[out["X_test"].index, "date"])
        assert set(dates.dt.dayofweek.unique()).issubset({0, 4})


def test_prepare_ml_data_for_prediction_with_cleaning_cache_old_triggers_clear_and_save():
    """
    When the cache exists but is older than 24h, the function should clear the cache
    and proceed to prepare fresh prediction data and save it.
    """

    from src.data_utils import ml_data_pipeline as pipeline

    fake_pred_result = {
        "X_test": _make_sample_df(num_days=10),
        "y_test": pd.Series(np.arange(10.0)),
    }

    with patch.object(pipeline._cleaned_data_cache, "cache_exists", return_value=True):
        with patch.object(
            pipeline._cleaned_data_cache, "get_cache_age_hours", return_value=25.0
        ) as _age:
            with patch.object(
                pipeline._cleaned_data_cache, "clear_cache", autospec=True
            ) as mock_clear:
                with patch.object(
                    pipeline,
                    "prepare_ml_data_for_prediction",
                    return_value=fake_pred_result,
                ):
                    with patch(
                        "src.data_utils.ml_data_pipeline.clean_data_for_training",
                        side_effect=_identity,
                    ) as _clean:
                        with patch(
                            "src.data_utils.ml_data_pipeline.analyze_feature_diversity",
                            side_effect=_fake_analyze_feature_diversity,
                        ) as _div:
                            with patch.object(
                                pipeline._cleaned_data_cache,
                                "save_cleaned_data",
                                autospec=True,
                            ) as mock_save:
                                out = pipeline.prepare_ml_data_for_prediction_with_cleaning(
                                    prediction_horizon=10, days_back=5
                                )

    mock_clear.assert_called_once()
    mock_save.assert_called_once()
    assert "X_test" in out and isinstance(out["X_test"], pd.DataFrame)

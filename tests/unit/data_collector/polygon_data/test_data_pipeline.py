import datetime
from unittest.mock import patch
import pytest

from src.data_collector.polygon_data.data_pipeline import DataPipeline


def test_run_grouped_daily_pipeline_skips_weekends_and_counts_success():
    """Ensure pipeline skips weekends and correctly counts processed tickers"""
    dp = DataPipeline(api_key="x", requests_per_minute=100)

    # health checks isolated
    with patch.object(dp, "_perform_health_checks", return_value=None):
        # deterministic fetcher: weekday produce one ticker
        def fake_get_grouped_daily_data(target_date, validate_data=True):
            """
            Fake data fetcher used in tests: returns a non-empty grouped-daily payload for weekdays and an empty payload for weekends.
            
            Parameters:
                target_date (datetime.date | datetime.datetime): Date for which to fetch grouped daily data.
                validate_data (bool, optional): Accepted for API compatibility; ignored by this fake implementation.
            
            Returns:
                dict: For weekday dates (Mon–Fri) returns {"T": [1]}; for weekend dates (Sat–Sun) returns an empty dict.
            """
            if target_date.weekday() < 5:
                return {"T": [1]}
            return {}

        with patch.object(
            dp.data_fetcher,
            "get_grouped_daily_data",
            side_effect=fake_get_grouped_daily_data,
        ):
            with patch.object(
                dp.storage, "store_historical_data", return_value={"stored_count": 1}
            ):
                start = datetime.date(2025, 3, 10)  # Monday
                end = datetime.date(2025, 3, 16)  # Sunday

                stats = dp.run_grouped_daily_pipeline(
                    start, end, validate_data=True, save_stats=False
                )
                assert stats.tickers_processed == 5
                assert stats.tickers_successful == 5


def test_run_grouped_daily_pipeline_raises_when_health_check_fails():
    """Pipeline should raise if pre-flight health checks fail"""
    dp = DataPipeline(api_key="x", requests_per_minute=100)
    with patch.object(
        dp, "_perform_health_checks", side_effect=RuntimeError("api down")
    ):
        with pytest.raises(RuntimeError):
            dp.run_grouped_daily_pipeline("2025-03-01", "2025-03-02", save_stats=False)

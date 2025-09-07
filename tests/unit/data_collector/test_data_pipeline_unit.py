import datetime
import pytest

from src.data_collector.polygon_data.data_pipeline import DataPipeline
from tests._fixtures.conftest import fake_http_response


def test_run_grouped_daily_pipeline_skips_weekends_and_counts_success(mocker, fake_http_response):
    dp = DataPipeline(api_key="x", requests_per_minute=100)

    # health checks isolated
    mocker.patch.object(dp, "_perform_health_checks", return_value=None)

    # deterministic fetcher: weekday produce one ticker
    def fake_get_grouped_daily_data(target_date, validate_data=True):
        if target_date.weekday() < 5:
            return {"T": [1]}
        return {}

    mocker.patch.object(dp.data_fetcher, "get_grouped_daily_data", side_effect=fake_get_grouped_daily_data)
    mocker.patch.object(dp.storage, "store_historical_data", return_value={"stored_count": 1})

    start = datetime.date(2025, 3, 10)  # Monday
    end = datetime.date(2025, 3, 16)    # Sunday

    stats = dp.run_grouped_daily_pipeline(start, end, validate_data=True, save_stats=False)
    assert stats.tickers_processed == 5, f"Expected 5 trading days processed, got {stats.tickers_processed}"
    assert stats.tickers_successful == 5, f"Expected 5 successes, got {stats.tickers_successful}"


def test_run_grouped_daily_pipeline_raises_when_health_check_fails(mocker):
    dp = DataPipeline(api_key="x", requests_per_minute=100)
    mocker.patch.object(dp, "_perform_health_checks", side_effect=RuntimeError("api down"))

    with pytest.raises(RuntimeError):
        dp.run_grouped_daily_pipeline("2025-03-01", "2025-03-02", save_stats=False)



from unittest.mock import patch
import pytest


from src.data_collector.polygon_data.data_pipeline import DataPipeline
from tests._fixtures import canned_api_factory
from src.data_collector.polygon_data.client import PolygonDataClient


@pytest.mark.parametrize(
    "api_results,expected_count",
    [
        ({"A": [{"T": "A", "c": 1.0, "o": 1.0, "h": 1.0, "l": 1.0, "v": 10}]}, 1),
        ({}, 0),
    ],
)
def test_run_grouped_daily_pipeline_mocks(api_results, expected_count, tmp_path):
    """
    Unit test for run_grouped_daily_pipeline that patches external API client and storage.
    Uses unittest.mock.patch (preferred) and parametrization per ETL guide.
    """

    # Arrange - patch health check and data fetching
    with patch(
        "src.data_collector.polygon_data.client.PolygonDataClient.health_check",
        return_value=True,
    ):
        with patch(
            "src.data_collector.polygon_data.data_fetcher.HistoricalDataFetcher.get_grouped_daily_data",
            return_value=api_results,
        ):
            with patch(
                "src.data_collector.polygon_data.data_storage.DataStorage.store_historical_data",
                return_value={"stored_count": expected_count},
            ):
                pipeline = DataPipeline(api_key="test", requests_per_minute=1000)
                # Run pipeline for a single date range (1 day)
                stats = pipeline.run_grouped_daily_pipeline(
                    start_date="2025-01-01",
                    end_date="2025-01-01",
                    validate_data=False,
                    save_stats=False,
                )

                # Assert - the pipeline tracked stored records correctly
                assert stats.total_records_stored == expected_count


def test_ticker_manager_get_all_active_tickers_returns_from_db(polygon_client):
    from src.data_collector.ticker_manager import TickerManager

    tm = TickerManager(polygon_client)
    with patch.object(tm.storage, "get_ticker_symbols", return_value=["A", "B"]):
        out = tm.get_all_active_tickers()
    assert out == ["A", "B"]

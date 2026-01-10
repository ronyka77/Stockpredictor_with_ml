"""
Main data pipeline orchestrator for Polygon.io data acquisition
"""

from typing import Union
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field

from src.utils.core.logger import get_logger
from src.data_collector.polygon_data.client import PolygonDataClient
from src.data_collector.ticker_manager import TickerManager
from src.data_collector.polygon_data.data_fetcher import HistoricalDataFetcher
from src.data_collector.polygon_data.data_storage import DataStorage
from src.data_collector.polygon_data.data_validator import DataValidator

logger = get_logger(__name__)


@dataclass
class PipelineStats:
    """Minimal pipeline execution statistics"""

    successful: int = 0
    failed: int = 0
    records_stored: int = 0
    errors: list = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.successful + self.failed

    @property
    def success_rate(self) -> float:
        return (self.successful / self.total * 100) if self.total else 0.0


class DataPipeline:
    """
    Main data acquisition pipeline orchestrating Polygon.io data fetch, validation, and storage
    """

    def __init__(self, requests_per_minute: int = 5):
        """Initialize pipeline with API client and data components"""
        self.client = PolygonDataClient(requests_per_minute)
        self.ticker_manager = TickerManager(self.client)
        self.data_fetcher = HistoricalDataFetcher(self.client)
        self.storage = DataStorage()
        self.validator = DataValidator()
        self.stats = PipelineStats()
        logger.info("Data pipeline initialized")

    def run_grouped_daily_pipeline(
        self, start_date: Union[str, date], end_date: Union[str, date], validate_data: bool = True
    ) -> PipelineStats:
        """
        Fetch grouped daily data for date range, storing validated records
        """
        # Parse dates
        start = (
            datetime.strptime(start_date, "%Y-%m-%d").date()
            if isinstance(start_date, str)
            else start_date
        )
        end = (
            datetime.strptime(end_date, "%Y-%m-%d").date()
            if isinstance(end_date, str)
            else end_date
        )

        logger.info(f"Starting pipeline: {start} to {end}")

        if not self.client.health_check():
            raise RuntimeError("API health check failed")

        # Generate weekday dates
        dates = [
            d
            for d in (start + timedelta(days=i) for i in range((end - start).days + 1))
            if d.weekday() < 5
        ]

        logger.info(f"Processing {len(dates)} trading days")

        # Process dates
        for i, dt in enumerate(dates, 1):
            logger.info(f"Processing {i}/{len(dates)}: {dt}")
            try:
                grouped = self.data_fetcher.get_grouped_daily_data(dt, validate_data)

                if not grouped:
                    self.stats.failed += 1
                    self.stats.errors.append(f"{dt}: No data")
                    continue

                records = list(grouped.values())
                if not records:
                    self.stats.failed += 1
                    continue

                result = self.storage.store_historical_data(records)
                stored = result.get("stored_count", 0)

                self.stats.successful += 1
                self.stats.records_stored += stored

                logger.info(f"Stored {stored} records ({len(grouped)} tickers)")

            except Exception as e:
                self.stats.failed += 1
                self.stats.errors.append(f"{dt}: {e}")
                logger.error(f"Error on {dt}: {e}")

        logger.info(
            f"Completed: {self.stats.successful}/{self.stats.total} dates ({self.stats.success_rate:.1f}%), "
            f"{self.stats.records_stored} records"
        )

        return self.stats

    def cleanup(self) -> None:
        """Release pipeline resources"""
        try:
            if hasattr(self.client, "session"):
                self.client.session.close()
            if hasattr(self.storage, "engine"):
                self.storage.engine.dispose()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def run_data_pipeline(
    days_back: int = 90, validate_data: bool = True, requests_per_minute: int = 5
) -> PipelineStats:
    """Run grouped daily pipeline for past N days"""
    end = datetime.now().date()
    start = end - timedelta(days=days_back)

    with DataPipeline(requests_per_minute) as pipeline:
        return pipeline.run_grouped_daily_pipeline(
            start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), validate_data
        )


if __name__ == "__main__":
    run_data_pipeline(days_back=375)

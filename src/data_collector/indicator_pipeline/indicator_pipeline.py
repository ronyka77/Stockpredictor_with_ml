"""
Batch Feature Engineering Processor

This module provides batch processing capabilities to calculate technical indicators
for all tickers in the database and store results in the technical_features table.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date
from src.feature_engineering.config import config
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from dataclasses import dataclass, asdict

from src.feature_engineering.data_loader import StockDataLoader
from src.data_collector.indicator_pipeline.feature_calculator import FeatureCalculator
from src.data_collector.indicator_pipeline.feature_storage import FeatureStorage
from src.data_collector.indicator_pipeline.consolidated_storage import (
    ConsolidatedFeatureStorage,
)
from src.utils.logger import get_logger
from src.utils.feature_categories import classify_feature_name

logger = get_logger(__name__, utility="feature_engineering")


def _process_ticker_worker(
    ticker: str,
    config_dict: Dict[str, Any],
    job_id: str,
):
    """Module-level worker used by ProcessPoolExecutor (must be top-level for Windows).

    Recreates required components inside the worker process to avoid sharing non-picklable
    objects (DB connections, class instances) across processes.
    """
    start_time = time.time()
    result = {
        "ticker": ticker,
        "success": False,
        "features_calculated": 0,
        "warnings": 0,
        "error": None,
        "processing_time": 0.0,
        "quality_score": 0.0,
    }

    try:
        # Local imports to keep worker lightweight and avoid top-level state
        from src.feature_engineering.data_loader import StockDataLoader
        from src.data_collector.indicator_pipeline.feature_calculator import (
            FeatureCalculator,
        )
        from src.data_collector.indicator_pipeline.feature_storage import (
            FeatureStorage,
        )

        loader = StockDataLoader()
        calculator = FeatureCalculator()
        storage = FeatureStorage()

        # Normalize date strings
        start_date = config_dict.get("start_date") or "2022-01-01"
        end_date = config_dict.get("end_date") or datetime.now().strftime("%Y-%m-%d")

        # If dates are date objects, convert to ISO strings
        if isinstance(start_date, (datetime, date)):
            start_date = start_date.isoformat()
        if isinstance(end_date, (datetime, date)):
            end_date = end_date.isoformat()

        stock_data = loader.load_stock_data(ticker, start_date, end_date)

        if stock_data.empty:
            result["error"] = "No data available"
            return result

        if len(stock_data) < config_dict.get("min_data_points", 0):
            result["error"] = (
                f"Insufficient data: {len(stock_data)} < {config_dict.get('min_data_points')}"
            )
            return result

        feature_result = calculator.calculate_all_features(
            stock_data, include_categories=config_dict.get("feature_categories")
        )

        result["features_calculated"] = len(feature_result.data.columns)
        result["warnings"] = len(feature_result.warnings)
        result["quality_score"] = feature_result.quality_score
        result["processing_time"] = time.time() - start_time

        if config_dict.get("save_to_parquet"):
            storage_metadata = {
                "categories": config_dict.get("feature_categories"),
                "quality_score": feature_result.quality_score,
                "warnings": feature_result.warnings,
                "job_id": job_id,
            }
            storage.save_features(ticker, feature_result.data, storage_metadata)

        if config_dict.get("save_to_database"):
            # Prepare rows for bulk upsert to avoid per-row DB inserts
            try:
                from src.database.db_utils import bulk_upsert_technical_features

                rows = []
                quality_score_val = (
                    float(feature_result.quality_score.item())
                    if hasattr(feature_result.quality_score, "item")
                    else float(feature_result.quality_score)
                )

                for date_idx, row in feature_result.data.iterrows():
                    for feature_name, feature_value in row.items():
                        if pd.isna(feature_value) or np.isinf(feature_value):
                            continue

                        feature_value = float(feature_value)
                        if abs(feature_value) >= 1e9:
                            continue

                        rows.append(
                            {
                                "ticker": ticker,
                                "date": date_idx.date(),
                                "feature_category": classify_feature_name(feature_name),
                                "feature_name": feature_name,
                                "feature_value": feature_value,
                                "quality_score": quality_score_val,
                            }
                        )

                bulk_upsert_technical_features(
                    rows,
                    page_size=1000,
                    overwrite=config_dict.get("overwrite_existing", True),
                )
            except Exception as e:
                logger.error(f"Bulk upsert failed for {ticker}: {e}")

        result["success"] = True
        return result

    except Exception as e:
        result["error"] = str(e)
        return result


@dataclass
class BatchJobConfig:
    """
    Configuration for batch processing jobs
    start_date and end_date accept either an ISO date string (`YYYY-MM-DD`) or a
    `datetime.date` object. They are normalized to `date` objects in
    `__post_init__` for consistent internal usage.
    """

    batch_size: int = config.batch_processing.DEFAULT_BATCH_SIZE
    max_workers: int = config.batch_processing.MAX_WORKERS
    use_processes: bool = True
    start_date: Optional[Union[str, date]] = None
    end_date: Optional[Union[str, date]] = None
    feature_categories: List[str] = None
    min_data_points: int = config.data_quality.MIN_DATA_POINTS
    save_to_database: bool = config.storage.SAVE_TO_DATABASE
    save_to_parquet: bool = config.storage.SAVE_TO_PARQUET
    use_consolidated_storage: bool = config.storage.USE_CONSOLIDATED_STORAGE
    partitioning_strategy: str = config.storage.PARTITIONING_STRATEGY
    overwrite_existing: bool = config.storage.OVERWRITE_EXISTING

    def __post_init__(self):
        if self.feature_categories is None:
            self.feature_categories = config.feature_categories.DEFAULT_CATEGORIES

        # Default start_date to HISTORICAL_YEARS start and end_date to today
        if self.start_date is None:
            self.start_date = date(
                datetime.now().year - config.fundamental.HISTORICAL_YEARS, 1, 1
            )
        else:
            # Accept ISO strings or datetime/date objects
            if isinstance(self.start_date, str):
                try:
                    self.start_date = date.fromisoformat(self.start_date)
                except Exception:
                    self.start_date = datetime.strptime(
                        self.start_date, "%Y-%m-%d"
                    ).date()
            elif isinstance(self.start_date, datetime):
                self.start_date = self.start_date.date()

        if self.end_date is None:
            self.end_date = date.today()
        else:
            if isinstance(self.end_date, str):
                try:
                    self.end_date = date.fromisoformat(self.end_date)
                except Exception:
                    self.end_date = datetime.strptime(self.end_date, "%Y-%m-%d").date()
            elif isinstance(self.end_date, datetime):
                self.end_date = self.end_date.date()


@dataclass
class ProcessingStats:
    """Statistics for batch processing"""

    total_tickers: int = 0
    processed_tickers: int = 0
    failed_tickers: int = 0
    total_features: int = 0
    total_warnings: int = 0
    start_time: datetime = None
    end_time: datetime = None

    @property
    def success_rate(self) -> float:
        if self.total_tickers == 0:
            return 0.0
        return (self.processed_tickers / self.total_tickers) * 100

    @property
    def processing_time(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


class BatchFeatureProcessor:
    """
    Batch processor for calculating technical indicators across multiple tickers
    """

    def __init__(self):
        """
        Initialize the batch processor
        """
        self.data_loader = StockDataLoader()
        self.feature_calculator = FeatureCalculator()
        self.feature_storage = FeatureStorage()
        self.consolidated_storage = ConsolidatedFeatureStorage()
        self.stats = ProcessingStats()
        self._lock = threading.Lock()

        logger.info("Initialized BatchFeatureProcessor")

    def get_available_tickers(
        self, min_data_points: int = None, market: str = None
    ) -> List[str]:
        """
        Get list of tickers with sufficient data for processing

        Args:
            min_data_points: Minimum number of data points required
            market: Market type filter ('stocks', 'crypto', 'forex', 'all')

        Returns:
            List of ticker symbols
        """
        # Apply config defaults
        min_data_points = min_data_points or config.data_quality.MIN_DATA_POINTS
        market = market or config.feature_categories.DEFAULT_MARKET

        logger.info(f"Getting tickers with at least {min_data_points} data points")
        logger.info(f"Filters: market={market}")

        try:
            tickers = self.data_loader.get_available_tickers(
                min_data_points=min_data_points,
                market=market,
            )

            logger.info(f"Found {len(tickers)} tickers ready for processing")
            return tickers
        except Exception as e:
            logger.error(f"Error getting available tickers: {str(e)}")
            raise

    def process_single_ticker(
        self, ticker: str, config: BatchJobConfig, job_id: str
    ) -> Dict[str, Any]:
        """
        Process a single ticker and calculate all features

        Args:
            ticker: Stock ticker symbol
            config: Batch job configuration
            job_id: Unique job identifier

        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        result = {
            "ticker": ticker,
            "success": False,
            "features_calculated": 0,
            "warnings": 0,
            "error": None,
            "processing_time": 0.0,
            "quality_score": 0.0,
        }

        try:
            # Load stock data
            stock_data = self.data_loader.load_stock_data(
                ticker,
                config.start_date or "2022-01-01",
                config.end_date or datetime.now().strftime("%Y-%m-%d"),
            )

            if stock_data.empty:
                result["error"] = "No data available"
                return result

            if len(stock_data) < config.min_data_points:
                result["error"] = (
                    f"Insufficient data: {len(stock_data)} < {config.min_data_points}"
                )
                return result

            # Calculate features
            feature_result = self.feature_calculator.calculate_all_features(
                stock_data, include_categories=config.feature_categories
            )

            result["features_calculated"] = len(feature_result.data.columns)
            result["warnings"] = len(feature_result.warnings)
            result["quality_score"] = feature_result.quality_score
            result["processing_time"] = time.time() - start_time

            # Save to storage systems
            if config.save_to_parquet:
                # Save to Parquet (primary storage)
                storage_metadata = {
                    "categories": config.feature_categories,
                    "quality_score": feature_result.quality_score,
                    "warnings": feature_result.warnings,
                    "job_id": job_id,
                }
                self.feature_storage.save_features(
                    ticker, feature_result.data, storage_metadata
                )
                # logger.info(f"Saved features to Parquet: {parquet_metadata.file_path}")

            if config.save_to_database:
                try:
                    from src.database.db_utils import bulk_upsert_technical_features

                    rows = []
                    quality_score_val = (
                        float(feature_result.quality_score.item())
                        if hasattr(feature_result.quality_score, "item")
                        else float(feature_result.quality_score)
                    )

                    for date_idx, row in feature_result.data.iterrows():
                        for feature_name, feature_value in row.items():
                            if pd.isna(feature_value) or np.isinf(feature_value):
                                continue
                            feature_value = float(feature_value)
                            if abs(feature_value) >= 1e9:
                                continue

                            rows.append(
                                {
                                    "ticker": ticker,
                                    "date": date_idx.date(),
                                    "feature_category": classify_feature_name(
                                        feature_name
                                    ),
                                    "feature_name": feature_name,
                                    "feature_value": feature_value,
                                    "quality_score": quality_score_val,
                                }
                            )

                    saved_count = bulk_upsert_technical_features(
                        rows, page_size=1000, overwrite=config.overwrite_existing
                    )
                    logger.info(
                        f"Saved {saved_count} feature records to database for {ticker}"
                    )
                except Exception as e:
                    logger.error(f"Error saving features for {ticker}: {e}")

            result["success"] = True
            logger.info(
                f"Successfully processed {ticker}: {result['features_calculated']} features"
            )

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error processing {ticker}: {str(e)}")

        return result

    def process_batch(
        self, tickers: List[str], config: BatchJobConfig
    ) -> Dict[str, Any]:
        """
        Process a batch of tickers with parallel processing

        Args:
            tickers: List of ticker symbols to process
            config: Batch job configuration

        Returns:
            Dictionary with batch processing results
        """
        job_id = str(uuid.uuid4())
        logger.info(
            f"Starting batch processing job {job_id} for {len(tickers)} tickers"
        )

        # Initialize stats
        self.stats = ProcessingStats(
            total_tickers=len(tickers), start_time=datetime.now()
        )

        results = []
        failed_tickers = []

        try:
            # Choose executor based on config flag. For process-based parallelism on
            # Windows, ProcessPoolExecutor requires module-level callables and
            if config.use_processes:
                executor_cls = ProcessPoolExecutor
            else:
                executor_cls = ThreadPoolExecutor

            with executor_cls(max_workers=config.max_workers) as executor:
                if config.use_processes:
                    # Convert config to serializable dict
                    config_dict = asdict(config)
                    # Submit module-level worker that recreates resources per process
                    future_to_ticker = {
                        executor.submit(
                            _process_ticker_worker,
                            ticker,
                            config_dict,
                            job_id,
                        ): ticker
                        for ticker in tickers
                    }
                else:
                    future_to_ticker = {
                        executor.submit(
                            self.process_single_ticker, ticker, config, job_id
                        ): ticker
                        for ticker in tickers
                    }

                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        result = future.result()
                        results.append(result)

                        # Update stats
                        with self._lock:
                            if result["success"]:
                                self.stats.processed_tickers += 1
                                self.stats.total_features += result[
                                    "features_calculated"
                                ]
                                self.stats.total_warnings += result["warnings"]
                            else:
                                self.stats.failed_tickers += 1

                        progress = (
                            (self.stats.processed_tickers + self.stats.failed_tickers)
                            / self.stats.total_tickers
                            * 100
                        )
                        logger.info(
                            f"Progress: {progress:.1f}% - Processed {ticker} ({'✓' if result['success'] else '✗'})"
                        )

                    except Exception as e:
                        logger.error(f"Error processing {ticker}: {str(e)}")
                        failed_tickers.append(ticker)
                        with self._lock:
                            self.stats.failed_tickers += 1

            self.stats.end_time = datetime.now()
            # Prepare summary
            summary = {
                "job_id": job_id,
                "total_tickers": self.stats.total_tickers,
                "successful": self.stats.processed_tickers,
                "failed": self.stats.failed_tickers,
                "success_rate": self.stats.success_rate,
                "total_features": self.stats.total_features,
                "total_warnings": self.stats.total_warnings,
                "processing_time": self.stats.processing_time,
                "failed_tickers": failed_tickers,
                "results": results,
            }

            logger.info(
                f"Batch processing completed: {self.stats.processed_tickers}/{self.stats.total_tickers} successful"
            )
            logger.info(f"Total features calculated: {self.stats.total_features}")
            logger.info(f"Processing time: {self.stats.processing_time:.2f} seconds")

            return summary

        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise

    def process_all_tickers(self, config: BatchJobConfig) -> Dict[str, Any]:
        """
        Process all available tickers in the database

        Args:
            config: Batch job configuration

        Returns:
            Dictionary with processing results
        """
        logger.info("Starting full database processing")

        # Get all available tickers
        all_tickers = self.get_available_tickers(config.min_data_points)

        if not all_tickers:
            logger.warning("No tickers found for processing")
            return {"error": "No tickers available for processing"}

        logger.info(
            f"Processing {len(all_tickers)} tickers in batches of {config.batch_size}"
        )

        # Process in batches
        all_results = []
        total_successful = 0
        total_failed = 0
        total_features = 0

        for i in range(0, len(all_tickers), config.batch_size):
            batch_tickers = all_tickers[i : i + config.batch_size]
            batch_num = (i // config.batch_size) + 1
            total_batches = (
                len(all_tickers) + config.batch_size - 1
            ) // config.batch_size

            logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch_tickers)} tickers)"
            )

            try:
                batch_result = self.process_batch(batch_tickers, config)
                all_results.append(batch_result)

                total_successful += batch_result["successful"]
                total_failed += batch_result["failed"]
                total_features += batch_result["total_features"]

                logger.info(
                    f"Batch {batch_num} completed: {batch_result['successful']}/{len(batch_tickers)} successful"
                )

            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {str(e)}")
                total_failed += len(batch_tickers)

        # Final summary
        summary = {
            "total_tickers": len(all_tickers),
            "successful": total_successful,
            "failed": total_failed,
            "success_rate": (total_successful / len(all_tickers)) * 100
            if all_tickers
            else 0,
            "total_features": total_features,
            "batch_results": all_results,
        }

        logger.info(
            f"Full processing completed: {total_successful}/{len(all_tickers)} tickers successful"
        )
        logger.info(f"Total features calculated: {total_features}")

        return summary

    def _save_features_to_database(
        self, ticker: str, feature_result, job_id: str, overwrite: bool = False
    ) -> int:
        """
        Save calculated features to the database

        Args:
            ticker: Stock ticker symbol
            feature_result: FeatureResult object
            job_id: Job identifier
            overwrite: Whether to overwrite existing features

        Returns:
            Number of records saved
        """
        # New implementation uses bulk upsert helper to improve throughput
        try:
            from src.database.db_utils import bulk_upsert_technical_features

            rows = []
            quality_score_val = (
                float(feature_result.quality_score.item())
                if hasattr(feature_result.quality_score, "item")
                else float(feature_result.quality_score)
            )

            for date_idx, row in feature_result.data.iterrows():
                for feature_name, feature_value in row.items():
                    if pd.isna(feature_value) or np.isinf(feature_value):
                        continue

                    feature_value = float(feature_value)
                    if abs(feature_value) >= 1e9:
                        logger.info(
                            f"Skipping {feature_name} value {feature_value} - too large for database precision"
                        )
                        continue

                    rows.append(
                        {
                            "ticker": ticker,
                            "date": date_idx.date(),
                            "feature_category": classify_feature_name(feature_name),
                            "feature_name": feature_name,
                            "feature_value": feature_value,
                            "quality_score": quality_score_val,
                        }
                    )

            saved_count = bulk_upsert_technical_features(
                rows, page_size=1000, overwrite=overwrite
            )
            return saved_count
        except Exception as e:
            logger.error(f"Error saving features for {ticker}: {e}")
            raise

    def close(self):
        """Close database connections"""
        if hasattr(self.data_loader, "close"):
            self.data_loader.close()


def run_production_batch():
    """Run production batch processing for all available tickers"""
    logger.info("🚀 Starting Production Feature Engineering Batch...")

    # Production configuration
    job_config = BatchJobConfig(
        batch_size=config.batch_processing.DEFAULT_BATCH_SIZE,
        max_workers=config.batch_processing.MAX_WORKERS,
        start_date="2023-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
        min_data_points=config.data_quality.MIN_DATA_POINTS // 2,
        save_to_parquet=config.storage.SAVE_TO_PARQUET,
        save_to_database=False,
        overwrite_existing=config.storage.OVERWRITE_EXISTING,
    )

    processor = BatchFeatureProcessor()

    try:
        logger.info("📊 Getting all available tickers...")
        all_tickers = processor.get_available_tickers(
            min_data_points=job_config.min_data_points, market="stocks"
        )
        logger.info(f"📈 Processing {len(all_tickers)} tickers:")
        storage = FeatureStorage()
        storage.remove_all_versions_for_all_tickers()

        # Run batch processing
        start_time = time.time()
        results = processor.process_batch(all_tickers, job_config)
        processing_time = time.time() - start_time

        # log results
        logger.info("🎉 Batch Processing Completed!")
        logger.info(f"   Total tickers: {results['total_tickers']}")
        logger.info(f"   Successful: {results['successful']}")
        logger.info(f"   Failed: {results['failed']}")
        logger.info(f"   Success rate: {results['success_rate']:.1f}%")
        logger.info(f"   Total features: {results['total_features']:,}")
        logger.info(f"   Processing time: {processing_time:.1f} seconds")

        # Check storage stats

        stats = storage.get_storage_stats()
        logger.info("📁 Storage Statistics:")
        logger.info(f"   Total tickers stored: {stats['total_tickers']}")
        logger.info(f"   Total storage size: {stats['total_size_mb']:.2f} MB")
        logger.info(f"   Storage path: {stats['base_path']}")

        # Show failed tickers if any
        if results["failed"] > 0:
            failed_tickers = [
                r["ticker"] for r in results["results"] if not r["success"]
            ]
            logger.warning(f"Failed tickers: {', '.join(failed_tickers[:10])}")

        # Consolidate into date-based partitions
        if results["successful"] > 0:
            logger.info("🗓️ Consolidating features into date-based partitions...")
            try:
                from src.data_collector.indicator_pipeline.consolidated_storage import (
                    consolidate_existing_features,
                )

                consolidation_start = time.time()
                consolidation_result = consolidate_existing_features(strategy="by_date")
                consolidation_time = time.time() - consolidation_start

                logger.info(
                    f"✅ Date-based consolidation completed in {consolidation_time:.2f} seconds"
                )
                logger.info(
                    f"   Date-partitioned files: {consolidation_result['files_created']}"
                )
                logger.info(
                    f"   Consolidated size: {consolidation_result['total_size_mb']:.2f} MB"
                )

                # Show date breakdown
                logger.info("📁 Date-based Files:")
                for file_info in consolidation_result["files"]:
                    date_label = file_info.get("date", file_info.get("year"))
                    logger.info(
                        f"   {file_info['file']}: {file_info['rows']:,} rows, Date: {date_label}"
                    )

                results["consolidation"] = consolidation_result

            except Exception as e:
                logger.error(f"⚠️  Consolidation failed: {str(e)}")
                results["consolidation_error"] = str(e)

        return results

    except Exception as e:
        logger.error(f"❌ Error in production batch: {e}", exc_info=True)
        return None
    finally:
        processor.close()


def main():
    """Main function for production batch processing"""
    run_production_batch()

    from src.utils.cleaned_data_cache import CleanedDataCache

    cache = CleanedDataCache()
    cache.clear_cache()


if __name__ == "__main__":
    main()

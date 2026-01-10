"""
Batch Feature Engineering Processor

This module provides batch processing capabilities to calculate technical indicators
for all tickers in the database and store results in the technical_features table.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Generator
from datetime import datetime, date
from src.data_collector.polygon_data.data_storage import DataStorage
from src.feature_engineering.config import config
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from dataclasses import dataclass, asdict

from src.feature_engineering.data_loader import StockDataLoader
from src.data_collector.indicator_pipeline.feature_calculator import FeatureCalculator
from src.data_collector.indicator_pipeline.feature_storage import FeatureStorage
from src.utils.core.logger import get_logger
from src.data_collector.indicator_pipeline.dividend_features import compute_dividend_features
from src.database.db_utils import bulk_upsert_technical_features
from src.data_collector.indicator_pipeline.consolidated_storage import consolidate_existing_features
from src.utils.data.cleaned_data_cache import CleanedDataCache

logger = get_logger(__name__)


def _normalize_date_value(value: Any, fallback_iso: str) -> str:
    """Normalize a date-like value to an ISO `YYYY-MM-DD` string."""
    if value is None or value == "":
        return fallback_iso
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _sanitize_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize feature data for persistence.

    - Replace infinite values with NaN.
    - Convert pandas nullable integer dtypes (Int64/UInt64) to float.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        dtype_name = str(df[col].dtype)
        if dtype_name.startswith("Int64") or dtype_name.startswith("UInt64"):
            df[col] = df[col].astype(float)
    return df


def _coerce_feature_value_for_json(value: Any) -> Optional[object]:
    """Convert a feature value into a JSON-serializable scalar.

    Returns None for values that should be skipped.
    Filtering rules mirror prior DB persistence behavior:
    - skip NaN and +/-inf
    - skip extreme magnitudes (abs(value) >= 1e9) for numeric values
    """

    if value is None:
        return None
    if pd.isna(value):
        return None
    # Handle numpy/pandas scalars cleanly.
    if isinstance(value, (np.generic,)):
        value = value.item()

    # Booleans should remain booleans (bool is also an int, so check first).
    if isinstance(value, bool):
        return value

    # Numeric values: filter inf and extreme magnitudes, cast to float for consistency.
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isinf(value):
            return None
        val = float(value)
        if abs(val) >= 1e9:
            return None
        return val

    # Fallback: stringify unknown types.
    return str(value)


def _process_ticker_worker(ticker: str, config_dict: Dict[str, Any], job_id: str):
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
        loader = StockDataLoader()
        calculator = FeatureCalculator()
        storage = FeatureStorage()

        start_date = _normalize_date_value(config_dict.get("start_date"), "2022-01-01")
        end_date = _normalize_date_value(
            config_dict.get("end_date"), datetime.now().strftime("%Y-%m-%d")
        )

        stock_data = loader.load_stock_data(ticker, start_date, end_date)

        if stock_data.empty:
            result["error"] = "No data available"
            return result

        if len(stock_data) < config_dict.get("min_data_points", 0):
            result["error"] = (
                f"Insufficient data: {len(stock_data)} < {config_dict.get('min_data_points')} for ticker {ticker}"
            )
            return result

        feature_result = calculator.calculate_all_features(
            stock_data, include_categories=config_dict.get("feature_categories")
        )

        data_length = len(feature_result.data)
        logger.info(
            f"Calculated {len(feature_result.data.columns)} features for {ticker} ({data_length} rows)"
        )

        # Load and compute dividend features
        try:
            dividends_df = DataStorage().load_dividends_for_ticker(ticker, start_date, end_date)

            if not dividends_df.empty:
                dividend_features = compute_dividend_features(stock_data, dividends_df)
                feature_result.data = feature_result.data.combine_first(dividend_features)

                # Add dividend feature metadata
                if hasattr(feature_result, "metadata") and feature_result.metadata:
                    feature_result.metadata["dividend_features_included"] = True
                    feature_result.metadata["dividend_records_count"] = len(dividends_df)
                else:
                    feature_result.metadata = {
                        "dividend_features_included": True,
                        "dividend_records_count": len(dividends_df),
                    }

        except Exception as e:
            warning_msg = f"Failed to compute dividend features for {ticker}: {str(e)}"
            feature_result.warnings.append(warning_msg)

        result["features_calculated"] = len(feature_result.data.columns)
        result["warnings"] = len(feature_result.warnings)
        result["quality_score"] = feature_result.quality_score
        result["processing_time"] = time.time() - start_time

        try:
            feature_result.data = _sanitize_feature_frame(feature_result.data)
        except Exception:
            logger.warning("Failed to sanitize feature_result.data")

        if config_dict.get("save_to_parquet"):
            storage_metadata = {
                "categories": config_dict.get("feature_categories"),
                "quality_score": feature_result.quality_score,
                "warnings": feature_result.warnings,
                "job_id": job_id,
                "dividend_features": feature_result.metadata.get(
                    "dividend_features_included", False
                )
                if hasattr(feature_result, "metadata") and feature_result.metadata
                else False,
                "dividend_source": "polygon"
                if (
                    hasattr(feature_result, "metadata")
                    and feature_result.metadata
                    and feature_result.metadata.get("dividend_features_included")
                )
                else None,
            }
            storage.save_features(ticker, feature_result.data, storage_metadata)

        if config_dict.get("save_to_database"):
            try:
                rows = []
                for date_idx, row in feature_result.data.iterrows():
                    payload: Dict[str, object] = {}
                    for feature_name, feature_value in row.items():
                        coerced = _coerce_feature_value_for_json(feature_value)
                        if coerced is None:
                            continue
                        payload[str(feature_name)] = coerced

                    if not payload:
                        continue

                    rows.append({"ticker": ticker, "date": date_idx.date(), "data": payload})

                bulk_upsert_technical_features(
                    rows, page_size=1000, overwrite=config_dict.get("overwrite_existing", True)
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
    start_date: Optional[Union[str, date]] = None
    end_date: Optional[Union[str, date]] = None
    feature_categories: Optional[List[str]] = None
    min_data_points: int = config.data_quality.MIN_DATA_POINTS
    save_to_database: bool = config.storage.SAVE_TO_DATABASE
    save_to_parquet: bool = config.storage.SAVE_TO_PARQUET
    overwrite_existing: bool = config.storage.OVERWRITE_EXISTING

    def __post_init__(self):
        if self.feature_categories is None:
            self.feature_categories = config.feature_categories.DEFAULT_CATEGORIES

        # Default start_date to HISTORICAL_YEARS start and end_date to today
        if self.start_date is None:
            self.start_date = date(datetime.now().year - config.fundamental.HISTORICAL_YEARS, 1, 1)
        else:
            # Accept ISO strings or datetime/date objects
            if isinstance(self.start_date, str):
                try:
                    self.start_date = date.fromisoformat(self.start_date)
                except Exception:
                    self.start_date = datetime.strptime(self.start_date, "%Y-%m-%d").date()
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
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a percentage of processed tickers."""
        if self.total_tickers == 0:
            return 0.0
        return (self.processed_tickers / self.total_tickers) * 100

    @property
    def processing_time(self) -> float:
        """Calculate total processing time in seconds."""
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
        self.stats = ProcessingStats()
        self._lock = threading.Lock()

        logger.info("Initialized BatchFeatureProcessor")

    def get_available_tickers(self, min_data_points: int = None, market: str = None) -> List[str]:
        """
        Get list of tickers with sufficient data for processing
        """
        min_data_points = min_data_points or config.data_quality.MIN_DATA_POINTS
        market = market or config.feature_categories.DEFAULT_MARKET

        logger.info(f"Getting tickers with at least {min_data_points} data points")
        logger.info(f"Filters: market={market}")

        try:
            tickers = self.data_loader.get_available_tickers(
                min_data_points=min_data_points, market=market
            )

            logger.info(f"Found {len(tickers)} tickers ready for processing")
            return tickers
        except Exception as e:
            logger.error(f"Error getting available tickers: {str(e)}")
            raise

    def process_batch(self, tickers: List[str], config: BatchJobConfig) -> Dict[str, Any]:
        """
        Process a batch of tickers with parallel processing.

        Note: This pipeline always uses process-based parallelism.
        """
        job_id = str(uuid.uuid4())
        logger.info(f"Starting batch processing job {job_id} for {len(tickers)} tickers")

        self.stats = ProcessingStats(total_tickers=len(tickers), start_time=datetime.now())

        results = []
        failed_tickers = []

        try:
            config_dict = asdict(config)

            with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
                future_to_ticker = {
                    executor.submit(_process_ticker_worker, ticker, config_dict, job_id): ticker
                    for ticker in tickers
                }

                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        result = future.result()
                        results.append(result)

                        if not result.get("success", False):
                            failed_tickers.append(ticker)

                        # Update stats
                        with self._lock:
                            if result["success"]:
                                self.stats.processed_tickers += 1
                                self.stats.total_features += result["features_calculated"]
                                self.stats.total_warnings += result["warnings"]
                            else:
                                self.stats.failed_tickers += 1

                        progress = (
                            (self.stats.processed_tickers + self.stats.failed_tickers)
                            / self.stats.total_tickers
                            * 100
                        )
                        if progress == int(progress):
                            logger.info(
                                f"Progress: {progress:.1f}% - Processed {ticker} ({'OK' if result['success'] else 'FAIL'})"
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
            logger.info(f"Processing time: {self.stats.processing_time:.2f} seconds")

            return summary

        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise

    def chunk_tickers(
        self, tickers: List[str], chunk_size: int
    ) -> Generator[List[str], None, None]:
        """
        Split a list of tickers into chunks for memory-efficient processing

        Args:
            tickers: List of ticker symbols
            chunk_size: Number of tickers per chunk

        Yields:
            Chunks of ticker symbols
        """
        for i in range(0, len(tickers), chunk_size):
            yield tickers[i : i + chunk_size]

    def close(self):
        """Close database connections"""
        if hasattr(self.data_loader, "close"):
            self.data_loader.close()


def run_indicator_pipeline():
    """Run production batch processing for all available tickers"""
    logger.info("Starting Production Feature Engineering Batch...")

    # Production configuration
    job_config = BatchJobConfig(
        batch_size=config.batch_processing.DEFAULT_BATCH_SIZE,
        max_workers=config.batch_processing.MAX_WORKERS,
        start_date="2023-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
        min_data_points=config.data_quality.MIN_DATA_POINTS,
        save_to_parquet=config.storage.SAVE_TO_PARQUET,
        save_to_database=True,
        overwrite_existing=config.storage.OVERWRITE_EXISTING,
    )

    processor = BatchFeatureProcessor()

    try:
        logger.info("Getting all available tickers...")
        all_tickers = processor.get_available_tickers(
            min_data_points=job_config.min_data_points, market="stocks"
        )
        logger.info(f"Processing {len(all_tickers)} tickers:")
        storage = FeatureStorage()
        storage.remove_all_versions_for_all_tickers()

        # Run batch processing
        start_time = time.time()
        results = processor.process_batch(all_tickers, job_config)
        processing_time = time.time() - start_time

        # log results
        logger.info("Batch Processing Completed!")
        logger.info(f"Total tickers: {results['total_tickers']}")
        logger.info(f"Successful: {results['successful']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Success rate: {results['success_rate']:.1f}%")
        logger.info(f"Total features: {results['total_features']:,}")
        logger.info(f"Processing time: {processing_time:.1f} seconds")

        # Check storage stats
        stats = storage.get_storage_stats()
        logger.info("Storage Statistics:")
        logger.info(f"Total tickers stored: {stats['total_tickers']}")
        logger.info(f"Total storage size: {stats['total_size_mb']:.2f} MB")
        logger.info(f"Storage path: {stats['base_path']}")

        # Show failed tickers if any
        if results["failed"] > 0:
            logger.warning(f"Failed tickers: {', '.join(results.get('failed_tickers', [])[:10])}")

        # Consolidate into year-partitioned files
        if results["successful"] > 0:
            logger.info("Consolidating features into year-partitioned files...")
            try:
                consolidation_result = consolidate_existing_features()
                logger.info(f"Year-partitioned files: {consolidation_result['files_created']}")
                logger.info(f"Consolidated size: {consolidation_result['total_size_mb']:.2f} MB")

                # Show year breakdown
                logger.info("Year-based files:")
                for file_info in consolidation_result["files"]:
                    logger.info(
                        f"{file_info['file']}: {file_info['rows']:,} rows, Year: {file_info['year']}"
                    )

                results["consolidation"] = consolidation_result

            except Exception as e:
                logger.error(f" Consolidation failed: {str(e)}")
                results["consolidation_error"] = str(e)

        return results

    except Exception as e:
        logger.error(f"Error in production batch: {e}", exc_info=True)
        return None
    finally:
        processor.close()


def main():
    """Main function for production batch processing"""
    run_indicator_pipeline()

    cache = CleanedDataCache()
    cache.clear_cache()


if __name__ == "__main__":
    main()

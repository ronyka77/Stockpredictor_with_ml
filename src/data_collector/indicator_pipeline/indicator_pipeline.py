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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from dataclasses import dataclass, asdict

from src.feature_engineering.data_loader import StockDataLoader
from src.data_collector.indicator_pipeline.feature_calculator import FeatureCalculator
from src.data_collector.indicator_pipeline.feature_storage import FeatureStorage
from src.data_collector.indicator_pipeline.consolidated_storage import ConsolidatedFeatureStorage
from src.utils.logger import get_logger
from src.utils.feature_categories import classify_feature_name

logger = get_logger(__name__, utility="feature_engineering")


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

        # Load and compute dividend features
        try:
            dividends_df = DataStorage().load_dividends_for_ticker(ticker, start_date, end_date)

            if not dividends_df.empty:
                from src.data_collector.indicator_pipeline.dividend_features import (
                    compute_dividend_features,
                )

                dividend_features = compute_dividend_features(stock_data, dividends_df)

                # Merge dividend features into main feature DataFrame
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
    use_processes: bool = True
    start_date: Optional[Union[str, date]] = None
    end_date: Optional[Union[str, date]] = None
    feature_categories: Optional[List[str]] = None
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
        self.feature_calculator = FeatureCalculator()
        self.feature_storage = FeatureStorage()
        self.consolidated_storage = ConsolidatedFeatureStorage()
        self.data_storage = DataStorage()
        self.stats = ProcessingStats()
        self._lock = threading.Lock()

        logger.info("Initialized BatchFeatureProcessor")

    def get_available_tickers(self, min_data_points: int = None, market: str = None) -> List[str]:
        """
        Get list of tickers with sufficient data for processing
        """
        # Apply config defaults
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

    def process_single_ticker(
        self, ticker: str, config: BatchJobConfig, job_id: str
    ) -> Dict[str, Any]:
        """
        Process a single ticker and calculate all features
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
                result["error"] = f"Insufficient data: {len(stock_data)} < {config.min_data_points}"
                return result

            # Calculate features
            feature_result = self.feature_calculator.calculate_all_features(
                stock_data, include_categories=config.feature_categories
            )

            # Load and compute dividend features
            try:
                dividends_df = self.data_storage.load_dividends_for_ticker(
                    ticker,
                    config.start_date or "2022-01-01",
                    config.end_date or datetime.now().strftime("%Y-%m-%d"),
                )

                if not dividends_df.empty:
                    from src.data_collector.indicator_pipeline.dividend_features import (
                        compute_dividend_features,
                    )

                    dividend_features = compute_dividend_features(stock_data, dividends_df)

                    # Merge dividend features into main feature DataFrame
                    # Use combine_first to preserve existing values and add new ones
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

                    logger.info(
                        f"Added {len(dividend_features.columns)} dividend features for {ticker}"
                    )
                else:
                    logger.info(f"No dividend data available for {ticker}")

            except Exception as e:
                warning_msg = f"Failed to compute dividend features for {ticker}: {str(e)}"
                feature_result.warnings.append(warning_msg)
                logger.warning(warning_msg)

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
                self.feature_storage.save_features(ticker, feature_result.data, storage_metadata)
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
                                    "feature_category": classify_feature_name(feature_name),
                                    "feature_name": feature_name,
                                    "feature_value": feature_value,
                                    "quality_score": quality_score_val,
                                }
                            )

                    saved_count = bulk_upsert_technical_features(
                        rows, page_size=1000, overwrite=config.overwrite_existing
                    )
                    logger.info(f"Saved {saved_count} feature records to database for {ticker}")
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

    def process_batch(self, tickers: List[str], config: BatchJobConfig) -> Dict[str, Any]:
        """
        Process a batch of tickers with parallel processing
        """
        job_id = str(uuid.uuid4())
        logger.info(f"Starting batch processing job {job_id} for {len(tickers)} tickers")

        # Initialize stats
        self.stats = ProcessingStats(total_tickers=len(tickers), start_time=datetime.now())

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
                        executor.submit(_process_ticker_worker, ticker, config_dict, job_id): ticker
                        for ticker in tickers
                    }
                else:
                    future_to_ticker = {
                        executor.submit(self.process_single_ticker, ticker, config, job_id): ticker
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
                                self.stats.total_features += result["features_calculated"]
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

    def chunk_tickers(self, tickers: List[str], chunk_size: int) -> Generator[List[str], None, None]:
        """
        Split a list of tickers into chunks for memory-efficient processing

        Args:
            tickers: List of ticker symbols
            chunk_size: Number of tickers per chunk

        Yields:
            Chunks of ticker symbols
        """
        for i in range(0, len(tickers), chunk_size):
            yield tickers[i:i + chunk_size]

    def process_in_chunks(
        self,
        tickers: List[str],
        config: BatchJobConfig,
        chunk_size: int = 50
    ) -> Dict[str, Any]:
        """
        Process tickers in memory-efficient chunks instead of all at once

        This method processes tickers in batches to reduce memory usage and provide
        better progress tracking for large ticker lists.

        Args:
            tickers: List of ticker symbols to process
            config: Batch processing configuration
            chunk_size: Number of tickers to process per chunk

        Returns:
            Dictionary with processing results and statistics
        """
        total_chunks = (len(tickers) + chunk_size - 1) // chunk_size
        all_results = []
        all_failed_tickers = []

        logger.info(f"Processing {len(tickers)} tickers in {total_chunks} chunks (size: {chunk_size})")

        chunk_count = 0
        for ticker_chunk in self.chunk_tickers(tickers, chunk_size):
            chunk_count += 1
            logger.info(f"Processing chunk {chunk_count}/{total_chunks} ({len(ticker_chunk)} tickers)")

            # Process chunk
            chunk_result = self.process_batch(ticker_chunk, config)

            # Aggregate results
            all_results.extend(chunk_result["results"])
            all_failed_tickers.extend(chunk_result.get("failed_tickers", []))

            # Log chunk summary
            logger.info(f"Chunk {chunk_count} complete: {chunk_result['successful']}/{len(ticker_chunk)} successful")

        # Create final aggregated summary
        total_successful = sum(1 for result in all_results if result["success"])
        total_failed = len(all_results) - total_successful
        total_features = sum(result.get("features_calculated", 0) for result in all_results if result["success"])

        final_summary = {
            "total_tickers": len(tickers),
            "successful": total_successful,
            "failed": total_failed,
            "success_rate": (total_successful / len(tickers) * 100) if tickers else 0,
            "total_features": total_features,
            "failed_tickers": all_failed_tickers,
            "results": all_results,
            "chunks_processed": total_chunks,
            "chunk_size": chunk_size,
        }

        logger.info(f"All chunks processed. Final results: {total_successful}/{len(tickers)} successful")
        return final_summary

    def process_with_callback(
        self,
        tickers: List[str],
        config: BatchJobConfig,
        callback: Any,
        chunk_size: int = 100,
        **callback_kwargs: Any
    ) -> int:
        """
        Process tickers in chunks and call a callback function for each chunk

        This method allows processing large ticker lists with custom callback logic,
        reducing memory usage by processing in batches.

        Args:
            tickers: List of ticker symbols to process
            config: Batch processing configuration
            callback: Function to call for each processed chunk
            chunk_size: Number of tickers to process per chunk before calling callback
            **callback_kwargs: Additional keyword arguments to pass to callback

        Returns:
            Total number of tickers processed
        """
        total_processed = 0
        total_chunks = (len(tickers) + chunk_size - 1) // chunk_size

        logger.info(f"Processing {len(tickers)} tickers with callback in {total_chunks} chunks")

        chunk_count = 0
        for ticker_chunk in self.chunk_tickers(tickers, chunk_size):
            chunk_count += 1
            logger.debug(f"Processing callback chunk {chunk_count}/{total_chunks}")

            # Process the chunk
            chunk_result = self.process_batch(ticker_chunk, config)

            # Call the callback with the results
            callback(chunk_result, chunk_number=chunk_count, **callback_kwargs)

            total_processed += len(ticker_chunk)
            logger.debug(f"Callback chunk {chunk_count} complete. Processed: {total_processed}")

        logger.info(f"Callback processing complete. Total tickers processed: {total_processed}")
        return total_processed

    def _save_features_to_database(
        self, ticker: str, feature_result, overwrite: bool = False
    ) -> int:
        """
        Save calculated features to the database
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

            saved_count = bulk_upsert_technical_features(rows, page_size=1000, overwrite=overwrite)
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
            failed_tickers = [r["ticker"] for r in results["results"] if not r["success"]]
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
                logger.info(f"   Date-partitioned files: {consolidation_result['files_created']}")
                logger.info(f"   Consolidated size: {consolidation_result['total_size_mb']:.2f} MB")

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

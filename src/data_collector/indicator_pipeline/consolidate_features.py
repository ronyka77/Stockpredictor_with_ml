#!/usr/bin/env python3
"""
Feature Consolidation Script - Year-Based Partitioning Only

Consolidate existing individual Parquet files into year-partitioned format
for optimal ML performance and time-series analysis.
"""

from src.data_collector.indicator_pipeline.consolidated_storage import (
    consolidate_existing_features,
)
from src.data_collector.indicator_pipeline.feature_storage import FeatureStorage
import time
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Consolidate existing feature files into year-based partitions"""
    logger.info("🗓️ Starting Year-Based Feature Consolidation...")

    # Check current storage
    storage = FeatureStorage()
    available_tickers = storage.get_available_tickers()
    current_stats = storage.get_storage_stats()

    logger.info("📊 Current Storage Status:")
    logger.info(f"   Individual files: {len(available_tickers)}")
    logger.info(f"   Total size: {current_stats['total_size_mb']:.2f} MB")
    logger.info(f"   Storage path: {current_stats['base_path']}")

    if not available_tickers:
        logger.info("❌ No individual feature files found to consolidate")
        return

    # Sample a few tickers to show date ranges
    logger.info("\n📅 Sample Date Ranges:")
    sample_tickers = available_tickers[:3]
    for ticker in sample_tickers:
        try:
            features, metadata = storage.load_features(ticker)
            logger.info(
                f"   {ticker}: {features.index.min().strftime('%Y-%m-%d')} to {features.index.max().strftime('%Y-%m-%d')}"
            )
        except Exception as e:
            logger.warning(f"   {ticker}: Error loading - {str(e)}")

    logger.info("\n🚀 Consolidating into year-based partitions...")

    try:
        start_time = time.time()

        # Consolidate features using year-based partitioning
        result = consolidate_existing_features(strategy="by_date")

        consolidation_time = time.time() - start_time

        # log results
        logger.info(
            f"✅ Year-based consolidation completed in {consolidation_time:.2f} seconds"
        )
        logger.info(f"   Files created: {result['files_created']}")
        logger.info(f"   Total size: {result['total_size_mb']:.2f} MB")
        logger.info(f"   Total rows: {result['total_rows']:,}")
        logger.info(f"   Compression ratio: {result['compression_ratio']:.1f}x")
        logger.info(
            f"   Size reduction: {((current_stats['total_size_mb'] - result['total_size_mb']) / current_stats['total_size_mb'] * 100):.1f}%"
        )

        # Show year-based file breakdown
        logger.info("\n📁 Year-based Files Created:")
        for file_info in result["files"]:
            logger.info(
                f"   {file_info['file']}: {file_info['rows']:,} rows, {file_info['size_mb']:.2f} MB, Year: {file_info['year']}"
            )

        return result

    except Exception as e:
        logger.warning(f"❌ Error with year-based consolidation: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    result = main()
    if result:
        logger.info("\n🎉 Year-based consolidation completed successfully!")
        logger.info("\n💡 Benefits of Year-Based Partitioning:")
        logger.info("   ✅ Fast year-specific loading")
        logger.info("   ✅ Easy train/test splits by year")
        logger.info("   ✅ Incremental data updates")
        logger.info("   ✅ Perfect for time-series ML")
        logger.info("   ✅ Memory efficient")
    else:
        logger.info("\n⚠️  Consolidation completed with issues.")

#!/usr/bin/env python3
"""
Clear problematic prediction cache files

This script clears the prediction cache files that are causing
Parquet serialization errors due to empty struct types.
"""

from src.utils.cleaned_data_cache import CleanedDataCache
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clear_prediction_cache():
    """
    Clear all prediction cache files to resolve Parquet serialization issues
    """
    logger.info("🧹 Clearing prediction cache files...")
    
    # Initialize cache manager
    cache = CleanedDataCache()
    
    # Clear all prediction cache files
    cache.clear_cache(data_type="prediction")
    
    logger.info("✅ Prediction cache cleared successfully!")
    logger.info("   The next prediction run will create new cache files with the fixed format.")

if __name__ == "__main__":
    clear_prediction_cache() 
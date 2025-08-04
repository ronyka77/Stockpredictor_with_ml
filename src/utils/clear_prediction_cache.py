#!/usr/bin/env python3
"""
Clear problematic prediction cache files

This script clears the prediction cache files that are causing
Parquet serialization errors due to empty struct types.
"""

import os
import shutil
from pathlib import Path
from src.utils.cleaned_data_cache import CleanedDataCache

def clear_prediction_cache():
    """
    Clear all prediction cache files to resolve Parquet serialization issues
    """
    print("🧹 Clearing prediction cache files...")
    
    # Initialize cache manager
    cache = CleanedDataCache()
    
    # Clear all prediction cache files
    cache.clear_cache(data_type="prediction")
    
    print("✅ Prediction cache cleared successfully!")
    print("   The next prediction run will create new cache files with the fixed format.")

if __name__ == "__main__":
    clear_prediction_cache() 
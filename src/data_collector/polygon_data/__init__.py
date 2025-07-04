"""
Polygon.io Data Acquisition Module

This module provides a comprehensive framework for acquiring stock market data
from Polygon.io API with rate limiting, error handling, and data validation.
"""

from .client import PolygonDataClient
from .data_fetcher import HistoricalDataFetcher
from ..ticker_manager import TickerManager
from .data_storage import DataStorage
from .data_validator import DataValidator, OHLCVRecord
from .data_pipeline import DataPipeline
from .rate_limiter import RateLimiter

__version__ = "1.0.0"
__author__ = "StockPredictor_V1"

__all__ = [
    "PolygonDataClient",
    "HistoricalDataFetcher", 
    "TickerManager",
    "DataStorage",
    "DataValidator",
    "OHLCVRecord",
    "DataPipeline",
    "RateLimiter"
] 
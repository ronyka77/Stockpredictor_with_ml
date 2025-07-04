"""
Polygon News Data Collection Package

This package provides comprehensive news data collection from Polygon.io API
with intelligent ticker prioritization, incremental updates, and quality validation.

Main Components:
- PolygonNewsCollector: Main orchestrator for news collection
- PolygonNewsClient: Specialized Polygon API client for news
- PolygonNewsStorage: Database operations and storage management
- NewsTickerIntegration: Ticker prioritization and management
- NewsProcessor: Content processing and normalization
- NewsValidator: Quality validation and filtering

Usage:
    from polygon_news import PolygonNewsCollector
    
    collector = PolygonNewsCollector(db_session, api_key)
    stats = collector.collect_incremental_news()
"""

from .news_pipeline import PolygonNewsCollector
from .news_client import PolygonNewsClient
from .storage import PolygonNewsStorage
from .models import PolygonNewsArticle, PolygonNewsTicker, PolygonNewsInsight, create_tables
from .ticker_integration import NewsTickerIntegration
from .processor import NewsProcessor
from .validator import NewsValidator

__version__ = "1.0.0"
__author__ = "StockPredictor Team"

__all__ = [
    'PolygonNewsCollector',
    'PolygonNewsClient', 
    'PolygonNewsStorage',
    'PolygonNewsArticle',
    'PolygonNewsTicker',
    'PolygonNewsInsight',
    'NewsTickerIntegration',
    'NewsProcessor',
    'NewsValidator',
    'create_tables'
] 
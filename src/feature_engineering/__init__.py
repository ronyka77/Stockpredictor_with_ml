"""
Feature Engineering Pipeline for Stock Prediction System

This package provides comprehensive technical indicator calculation,
feature storage, and pipeline orchestration for financial time series data.

Main Components:
- Technical Indicators: Trend, momentum, volatility, and volume indicators
- Data Loading: Integration with existing stock data storage
- Feature Storage: Hybrid database/Parquet storage system
- Pipeline: Complete feature engineering workflow
- Validation: Data quality and feature validation

Usage:
    from src.feature_engineering import FeaturePipeline
    
    pipeline = FeaturePipeline()
    features = pipeline.calculate_features('AAPL', '2024-01-01', '2024-12-31')
"""

from src.feature_engineering.technical_indicators.feature_calculator import FeatureCalculator
from src.feature_engineering.data_loader import StockDataLoader

# Import technical indicators for convenience
from src.feature_engineering.technical_indicators.base import BaseIndicator, IndicatorResult
from src.feature_engineering.technical_indicators.trend_indicators import TrendIndicatorCalculator
from src.feature_engineering.technical_indicators.momentum_indicators import MomentumIndicatorCalculator
from src.feature_engineering.technical_indicators.volatility_indicators import VolatilityIndicatorCalculator
from src.feature_engineering.technical_indicators.volume_indicators import VolumeIndicatorCalculator

__version__ = "1.0.0"
__author__ = "StockPredictor Team"

__all__ = [
    'FeatureCalculator', 
    'StockDataLoader',
    'BaseIndicator',
    'IndicatorResult',
    'TrendIndicatorCalculator',
    'MomentumIndicatorCalculator',
    'VolatilityIndicatorCalculator',
    'VolumeIndicatorCalculator'
] 
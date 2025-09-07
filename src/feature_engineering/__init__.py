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
    from src.data_collector.indicator_pipeline import FeatureCalculator

    calculator = FeatureCalculator()
    features = calculator.calculate_features('AAPL', '2024-01-01', '2024-12-31')
"""

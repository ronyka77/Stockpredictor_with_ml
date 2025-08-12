"""
Data Utils Package

This package provides comprehensive data loading, transformation, and preparation
utilities for ML models, organized into focused modules:

- target_engineering: Target transformation functions (absolute to percentage returns)
- feature_engineering: Feature transformation and cleaning functions
- ml_data_pipeline: Main pipeline functions for ML data preparation

Key improvements implemented:
- Phase 1: Target engineering (absolute prices → percentage returns)
- Phase 2: Price-normalized features and prediction bounds
- Comprehensive data cleaning and caching system
"""

# Core ML data pipeline functions
from .ml_data_pipeline import (
    prepare_ml_data_for_training,
    prepare_ml_data_for_prediction,
    prepare_ml_data_for_training_with_cleaning,
    prepare_ml_data_for_prediction_with_cleaning,
)

# Target engineering functions
from .target_engineering import (
    convert_absolute_to_percentage_returns,
    convert_percentage_predictions_to_prices,
    create_target_features
)

# Feature engineering functions
from .feature_engineering import (
    add_price_normalized_features,
    add_prediction_bounds_features,
    clean_data_for_xgboost,
    analyze_feature_diversity,
    clean_features_for_training,
    add_temporal_features,
)

# Backward compatibility - import main functions at package level
__all__ = [
    # Main pipeline functions (most commonly used)
    'prepare_ml_data_for_training',
    'prepare_ml_data_for_prediction',
    'prepare_ml_data_for_training_with_cleaning',
    'prepare_ml_data_for_prediction_with_cleaning',
    
    # Target engineering (Phase 1 fixes)
    'convert_absolute_to_percentage_returns',
    'convert_percentage_predictions_to_prices',
    
    # Feature engineering (Phase 2 fixes)
    'add_price_normalized_features',
    'add_prediction_bounds_features',
    
    # Data cleaning
    'clean_data_for_xgboost',
    'clean_features_for_training',
    'analyze_feature_diversity',
    
    # Utility functions
    'add_temporal_features',
    'create_target_features',
]
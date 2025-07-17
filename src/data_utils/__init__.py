"""
Data Utils Package

This package provides comprehensive data loading, transformation, and preparation
utilities for ML models, organized into focused modules:

- ml_feature_loader: MLFeatureLoader class and data loading functions
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

# ML feature loading
from .ml_feature_loader import (
    MLFeatureLoader,
    load_ml_ready_data,
    load_yearly_data,
    load_date_range_data,
    load_all_data
)

# Target engineering functions
from .target_engineering import (
    convert_absolute_to_percentage_returns,
    convert_percentage_predictions_to_prices,
    validate_target_quality,
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
    validate_feature_quality
)

# Backward compatibility - import main functions at package level
__all__ = [
    # Main pipeline functions (most commonly used)
    'prepare_ml_data_for_training',
    'prepare_ml_data_for_prediction',
    'prepare_ml_data_for_training_with_cleaning',
    'prepare_ml_data_for_prediction_with_cleaning',
    
    # Data loading
    'load_all_data',
    'load_ml_ready_data',
    'MLFeatureLoader',
    
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
    
    # Validation functions
    'validate_target_quality',
    'validate_feature_quality',
    
    # Utility functions
    'add_temporal_features',
    'create_target_features',
    'load_yearly_data',
    'load_date_range_data'
]

# Package metadata
__version__ = "2.0.0"
__author__ = "StockPredictor_V1"
__description__ = "Comprehensive data utilities for ML model training and prediction"

# Quick usage examples in docstring
__doc__ += """

Quick Usage Examples:
====================

1. Basic ML data preparation:
```python
from src.data_utils import prepare_ml_data_for_training_with_cleaning

# Prepare data with all enhancements (recommended)
data = prepare_ml_data_for_training_with_cleaning(
    prediction_horizon=10,
    split_date='2025-02-01',
    clean_features=True,
    use_cache=True
)

X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']
```

2. Load data for specific ticker:
```python
from src.data_utils import load_all_data

# Load all data for AAPL
data = load_all_data(ticker='AAPL')
```

3. Target engineering (Phase 1 fix):
```python
from src.data_utils import convert_absolute_to_percentage_returns

# Convert absolute prices to percentage returns
data, target_col = convert_absolute_to_percentage_returns(data, horizon=10)
```

4. Feature engineering (Phase 2 fixes):
```python
from src.data_utils import add_price_normalized_features, add_prediction_bounds_features

# Add price-normalized features
features = add_price_normalized_features(features)

# Add prediction bounds features  
features = add_prediction_bounds_features(features)
```

5. Cache management:
```python
from src.data_utils import clear_cleaned_data_cache, get_cache_info

# Check cache status
cache_info = get_cache_info()

# Clear cache if needed
clear_cleaned_data_cache()
```
""" 
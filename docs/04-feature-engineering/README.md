# Feature Engineering Overview

This section summarizes the project's feature engineering process and points to the detailed documentation and code.

## Goals
- Transform raw market data into stable, informative features
- Normalize price-dependent signals to ratios
- Add temporal/context features
- Clean and validate features for robust model training
- Produce targets as percentage returns for realistic predictions

## High-level Flow
1. Load and assemble base market data
2. Compute technical indicators (trend, momentum, volatility, volume)
3. Apply normalization and context features (ratios, bounds context, temporal)
4. Clean and validate features (NaNs, extremes, variance, correlation)
5. Select/train with model-specific pipelines
6. Prepare targets as percentage returns (and optional target-derived features)

## Key Documents
- Feature and Target Engineering (APIs, usage):
  - `docs/04-feature-engineering/FEATURE_AND_TARGET_ENGINEERING.md`
- Fundamentals (collection and usage for features):
  - `docs/04-feature-engineering/fundamental/FUNDAMENTAL_PIPELINE_README.md`
  - `docs/04-feature-engineering/fundamental/SECTOR_ANALYSIS_IMPLEMENTATION.md`
  - V2 fundamentals plan: `docs/04-feature-engineering/fundamental/FUNDAMENTAL_DATA_COLLECTION_V2_PLAN.md`

## Core Code Modules
- Feature utilities: `src/data_utils/feature_engineering.py`
- Target utilities: `src/data_utils/target_engineering.py`
- Technical indicators (calculators, pipeline, storage):
  - `src/feature_engineering/technical_indicators/`
  - `src/feature_engineering/technical_indicators/indicator_pipeline.py`

## Recommended Practices
- Preserve essentials through cleaning: `close`, `ticker_id`, `date_int`
- Normalize absolute metrics to ratios (e.g., `close/SMA_*`) to stabilize learning
- Add `date_int` and basic calendar features for lightweight temporal context
- Use median imputation and cap extremes before casting to float64
- Remove constant/zero-variance/highly-correlated features (with safeguards)
- Use percentage-return targets (decimal) for training; convert back to prices for reporting

## Minimal Usage Snippet
```python
from src.data_utils.feature_engineering import (
    add_price_normalized_features,
    add_prediction_bounds_features,
    add_temporal_features,
    clean_data_for_xgboost,
    clean_features_for_training,
)
from src.data_utils.target_engineering import (
    convert_absolute_to_percentage_returns,
)

X = add_price_normalized_features(X)
X = add_prediction_bounds_features(X)
X = add_temporal_features(X, date_column='date')
X = clean_data_for_xgboost(X)
X, y, _ = clean_features_for_training(X, y)

data, target_col = convert_absolute_to_percentage_returns(data, prediction_horizon=10)
```

## Where to Go Next
- Full function reference and examples: `FEATURE_AND_TARGET_ENGINEERING.md`
- Technical indicator orchestration: `src/feature_engineering/technical_indicators/indicator_pipeline.py`
- Fundamentals data pipeline (for fundamental-based features): see fundamental docs above

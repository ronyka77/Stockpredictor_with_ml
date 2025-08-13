# Feature & Target Engineering Guide

This guide documents the core engineering utilities under `src/data_utils/` that prepare data for model training and prediction.

- Feature engineering: `src/data_utils/feature_engineering.py`
- Target engineering: `src/data_utils/target_engineering.py`

The functions here are model-agnostic but tuned for gradient-boosting workflows (e.g., XGBoost) used in this project.

## 1) Feature Engineering

File: `src/data_utils/feature_engineering.py`

### add_price_normalized_features(df)
- Purpose: Convert absolute price-based indicators into ratios relative to `close` to stabilize learning.
- Adds:
  - `SMA_*_Ratio`: close / SMA_*
  - `BB_Position`: normalized price position within Bollinger Bands (0=lower, 1=upper)
  - `Price_ATR_Ratio`: close / ATR
  - `Return_Volume_Efficiency`: abs(Return_1D) / volume
  - `Ichimoku_*_Ratio`: close / Ichimoku_*
  - `Close_Open_Ratio`: close / open
- Requires: `close`; optional columns used if present (`SMA_*`, `BB_Lower`, `BB_Upper`/`BB_Width`, `ATR`, `volume`, `Return_1D`, `Ichimoku_*`, `open`).

### add_prediction_bounds_features(df)
- Purpose: Provide context features to discourage extreme predictions.
- Adds if present:
  - `Expected_Daily_Move`, `Expected_10D_Move` from `ATR_Percent`
  - `Recent_Momentum_5D`, `Momentum_Acceleration` from returns
  - `Vol_Regime_Context` from volatility regime flags
  - `RSI_Mean_Reversion_Pressure` from `RSI_14`
  - `BB_Range_Context` from `BB_Percent`

### add_temporal_features(df, date_column='date')
- Purpose: Lightweight time-based features.
- Adds: `date_int`, `year`, `month`, `day_of_year`, `quarter`, `day_of_week`, `is_month_end`, `is_quarter_end`.
- Converts non-datetime date columns as needed.

### clean_data_for_xgboost(df)
- Purpose: Robust cleaning tailored for XGBoost.
- Actions:
  - Replace ±∞ → NaN
  - Cap extreme numeric values to safe float32-like range
  - Fill NaNs with median per column (fallback 0.0 when all-NaN)
  - Cast numerics to float64

### analyze_feature_diversity(df, min_variance_threshold=1e-8)
- Purpose: Diagnose feature variance issues prior to training.
- Returns counts and lists for zero/low-variance and constant features; top-variance map; summary stats.

### clean_features_for_training(X, y, ...)
- Purpose: Remove problematic features with safeguards.
- Removes:
  - Non-numeric columns (while preserving essentials: `close`, `ticker_id`, `date_int`)
  - Constant and zero-variance features
  - Highly correlated features above `correlation_threshold` (default 0.99), excluding essentials
- Returns: `(X_clean, y_clean, removed_features_info)`

### Example (features)
```python
from src.data_utils.feature_engineering import (
    add_price_normalized_features,
    add_prediction_bounds_features,
    add_temporal_features,
    clean_data_for_xgboost,
    clean_features_for_training,
)

X = add_price_normalized_features(X)
X = add_prediction_bounds_features(X)
X = add_temporal_features(X, date_column='date')
X = clean_data_for_xgboost(X)
X_clean, y_clean, removed = clean_features_for_training(X, y)
```

## 2) Target Engineering

File: `src/data_utils/target_engineering.py`

### convert_absolute_to_percentage_returns(df, prediction_horizon=10)
- Purpose: CRITICAL fix — convert absolute future price targets to percentage returns.
- Default expects `Future_High_{h}D` and `close`; falls back to any `Future_High_*` if the exact column is missing.
- Adds: `Future_Return_{h}D` (decimal form, e.g., 0.05 = 5%).
- Returns: `(updated_df, new_target_column_name)`

### convert_percentage_predictions_to_prices(preds, current_prices, apply_bounds=True, max_daily_move=10.0)
- Purpose: Convert model return predictions back to prices with optional bounds.
- Bounds: Scales `max_daily_move` by `sqrt(10)` for 10D horizon (configurable logic).

### create_target_features(df, target_column, lookback_periods=[5, 10, 20])
- Purpose: Create target-derived context features.
- Adds (if sufficient length):
  - Rolling volatility/mean of target, momentum vs mean
  - Regime flags and percentiles
  - Simple trend indicators over 5D/10D

### Example (targets)
```python
from src.data_utils.target_engineering import (
    convert_absolute_to_percentage_returns,
    convert_percentage_predictions_to_prices,
    create_target_features,
)

# Build % return target
data, target_col = convert_absolute_to_percentage_returns(data, prediction_horizon=10)

# Optional: target-derived context features
data = create_target_features(data, target_column=target_col, lookback_periods=[5, 10, 20])

# Convert predictions back to prices
predicted_prices = convert_percentage_predictions_to_prices(y_pred, current_prices=data['close'].to_numpy())
```

## 3) Integration Notes
- Use `clean_data_for_xgboost` before training to standardize numeric stability.
- Preserve `close`, `ticker_id`, `date_int` through cleaning for downstream evaluators and predictors.
- Pair `add_price_normalized_features` and `add_prediction_bounds_features` to stabilize both scale and behavior.
- The target conversion operates in decimal returns; ensure consistency across training, evaluation, and export.
- For pipeline usage, these utilities are commonly orchestrated by `src/data_utils/ml_data_pipeline.py` with caching.

## 4) Data & Logging
- Functions are defensive: they only add features when required columns exist.
- Centralized logging via `src/utils/logger.py` provides progress and diagnostics.

## 5) Gotchas
- If `date` is not timezone-aware or not standard ISO, ensure it parses correctly before temporal feature creation.
- When converting predictions to prices, adjust `max_daily_move` for your horizon if not 10D.
- Highly correlated feature removal is powerful — validate the final feature set for domain relevance.

## 6) Quick checklist
- Targets are percentage returns (decimal), not prices
- Features cleaned and numeric; essentials preserved
- Ratios and context features added where possible
- Temporal columns present and typed correctly
- Consistent handling from training → prediction (same feature/target logic)

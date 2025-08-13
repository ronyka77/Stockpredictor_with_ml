# StockPredictor V1

Advanced stock prediction system combining Polygon.io data collection, feature engineering, gradient-boosted models, and neural networks with production-oriented tooling (caching, logging, MLflow, and batch processing).

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Data Collection](#data-collection)
- [Feature Engineering](#feature-engineering)
- [Data Utilities (ML Prep + Caching)](#data-utilities-ml-prep--caching)
- [Model Training and Prediction](#model-training-and-prediction)
- [Evaluation and Threshold Optimization](#evaluation-and-threshold-optimization)
- [Testing](#testing)
- [Known Issues and Notes](#known-issues-and-notes)
- [Recent Updates](#recent-updates)
- [License](#license)

## Overview
StockPredictor V1 is an end-to-end ML system for stock prediction:
- Collects OHLCV prices, fundamentals, and news from Polygon.io
- Engineers robust technical features at scale (with Parquet storage and consolidation)
- Prepares data for ML with target and feature transformations
- Trains and evaluates models (LightGBM, XGBoost, MLP) with MLflow integration
- Exports predictions with profit and confidence-based filtering

## Key Features
- **Data Collection**
  - Polygon.io OHLCV and News pipelines with rate limiting and validation
  - Fundamentals Pipeline V1 (operational) and V2 (clean layered pipeline with staging table)
- **Feature Engineering**
  - Technical indicators across trend, momentum, volatility, and volume categories
  - Batch processor with parallelism, Parquet storage, and year-based consolidation
- **Data Utilities**
  - Target engineering (absolute → percentage returns), price-normalized features, prediction bounds
  - Robust data cleaning utilities for XGBoost and training pipelines
  - Cleaned data caching to speed up iteration (training/prediction TTL)
- **Models and Prediction**
  - LightGBM/XGBoost pipeline with MLflow logging
  - MLP (PyTorch) with GPU support (CUDA 12.8)
  - Threshold optimization and profit-based evaluation
  - Prediction framework with confidence filtering and Excel export
- **Observability and Ops**
  - Centralized logging (`src/utils/logger.py`)
  - Configurable batch processing with job tracking and statistics
  - SQL schemas for relational storage and staging

## Architecture
```
StockPredictor V1
├── Data Collection (Polygon.io)
│   ├── OHLCV + Tickers (src/data_collector/polygon_data)
│   ├── News (src/data_collector/polygon_news)
│   └── Fundamentals v1 (src/data_collector/polygon_fundamentals)
│       Fundamentals v2 (src/data_collector/polygon_fundamentals_v2)
├── Feature Engineering
│   └── Technical Indicators (src/feature_engineering/technical_indicators)
│       Parquet storage + consolidated year-based storage
├── Data Utilities (src/data_utils)
│   ├── target_engineering
│   ├── feature_engineering
│   └── ml_data_pipeline (+ cleaned data caching)
├── Models
│   ├── Gradient Boosting (LightGBM/XGBoost)
│   └── Time Series / MLP (PyTorch)
├── Evaluation
│   └── Threshold optimization and profit metrics
└── Infrastructure
    ├── SQL schemas (sql/)
    └── Logging, MLflow, and caching utilities
```

## Requirements
- Python 3.12+
- PostgreSQL (for relational storage; required by data pipelines and some batch processes)
- Polygon.io API key
- Windows 11 supported; GPU optional (PyTorch with CUDA 12.8)

## Installation
Use uv exclusively.

```bash
# Create and activate environment
uv venv
# Install project dependencies from pyproject.toml / uv.lock
uv sync
```

Optional (GPU):
- Install CUDA 12.8 and a compatible PyTorch build per official guidance.

## Configuration
- Central logging: `src/utils/logger.py`
- Technical indicators and batch config: `src/feature_engineering/config.py`
- MLflow utilities: `src/utils/mlflow_integration.py`, `src/utils/mlflow_utils.py`
- Database connection (example variables):
  - `DATABASE_URL` (e.g., postgres://user:pass@host:port/dbname)
  - `POLYGON_API_KEY` for API clients

Schemas:
- Core schema: `sql/database_schema.sql`
- Fundamentals V2 staging: `sql/fundamentals_v2_schema.sql`

Apply staging schema (example):
```bash
# Use your PostgreSQL client; example with psql
psql -d your_db -f sql/fundamentals_v2_schema.sql
```

## Data Collection
### Fundamentals Pipeline V2 (staging-first, clean layering)
- Modules: `src/data_collector/polygon_fundamentals_v2/`
- Entry point: `run_pipeline.py` (async)

Run:
```bash
uv run python -m src.data_collector.polygon_fundamentals_v2.run_pipeline
```

What it does:
- Client → Parser → Validator → Repository → Service → Processor → Runner
- Writes raw fundamentals JSON to staging table (`raw_fundamental_json`), idempotent via `response_hash`
- Marks tickers with no fundamentals (`has_financials=false`) when applicable

### Technical Indicators Batch
- Module: `src/feature_engineering/technical_indicators/indicator_pipeline.py`
- Storage: Parquet per ticker; optional consolidation by year

Run production batch:
```bash
uv run python -m src.feature_engineering.technical_indicators.indicator_pipeline
```

Behavior:
- Discovers tickers with sufficient data
- Calculates all categories (trend, momentum, volatility, volume)
- Saves to Parquet (and optionally to DB)
- Optional consolidation into year-partitioned Parquet for fast ML workflows

## Feature Engineering
Technical indicator calculation and storage live in `src/feature_engineering/technical_indicators/` with:
- `feature_calculator.py` (orchestrates category calculations)
- `feature_storage.py` (Parquet I/O)
- `consolidated_storage.py` (year-based partitions + stats)
- `indicator_pipeline.py` (batch execution with stats and job tracking)

## Data Utilities (ML Prep + Caching)
Location: `src/data_utils/`

Main components:
- `target_engineering.py`
  - `convert_absolute_to_percentage_returns`
  - `convert_percentage_predictions_to_prices`
  - `create_target_features`
- `feature_engineering.py`
  - `add_price_normalized_features`
  - `add_prediction_bounds_features`
  - `clean_data_for_xgboost`
  - `analyze_feature_diversity`
  - `clean_features_for_training`
  - `add_temporal_features`
- `ml_data_pipeline.py`
  - `prepare_ml_data_for_training` / `prepare_ml_data_for_prediction`
  - Enhanced cached variants:
    - `prepare_ml_data_for_training_with_cleaning`
    - `prepare_ml_data_for_prediction_with_cleaning`
    - `prepare_ml_data_for_training_with_cleaning_memory_optimized`

Cleaned data cache:
- `src/utils/cleaned_data_cache.py` with Parquet-based cache and 24h freshness checks in callers.

Example (training data prep with caching and cleaning):
```python
from src.data_utils import prepare_ml_data_for_training_with_cleaning

result = prepare_ml_data_for_training_with_cleaning(
    prediction_horizon=10,
    split_date='2025-02-01',
    ticker=None,  # all available
    clean_features=True,
    apply_stationarity_transform=False
)

 X_train = result['X_train']
 y_train = result['y_train']
 X_test  = result['X_test']
 y_test  = result['y_test']
print("Features:", result['feature_count'], "Train:", len(X_train), "Test:", len(X_test))
```

Example (prediction data prep for recent period with caching):
```python
from src.data_utils import prepare_ml_data_for_prediction_with_cleaning

pred = prepare_ml_data_for_prediction_with_cleaning(
    prediction_horizon=10,
    days_back=30
)

X_test = pred['X_test']
'y_test = pred['y_test']
print("Prediction samples:", len(X_test))
```

Notes:
- Pipelines convert absolute targets to percentage returns (Phase 1 fix).
- Feature engineering includes price normalization and prediction-bound context (Phase 2).
- Monday/Friday filtering and a 3-day holdout are applied where relevant before caching.

## Model Training and Prediction
- Gradient Boosting: `src/models/gradient_boosting/` (LightGBM, XGBoost)
- MLP (PyTorch): `src/models/time_series/mlp/`
- Prediction framework: `src/models/predictors/`

Export the best LightGBM run (by profit) across an experiment:
```python
from src.models.predictors.lightgbm_all_run_predictor import run_all_and_export_best

# Evaluates active MLflow runs for the experiment on recent data,
# exports the best run's predictions to Excel in predictions/lightgbm/
run_all_and_export_best(
    experiment_name="LightGBM-Experiment",
    days_back=30
)
```

Prediction outputs:
- Saved under `predictions/lightgbm/` with timestamped filenames
- Include confidence, profit analysis, and metadata

## Evaluation and Threshold Optimization
- Threshold optimizer and profit-based evaluation:
  - `src/models/evaluation/threshold_evaluator.py`
  - Confidence-based filtering (simple, margin, leaf depth, variance, lstm_hidden)
- Custom metrics:
  - Conservative accuracy, profit per investment, success rates
- Integrated into LightGBM and MLP workflows

## Testing
```bash
uv run pytest -q
```
- Pytest is configured in `pyproject.toml`
- Includes ML, data, and logging tests under `src/tests/`

## Known Issues and Notes
- Ensure database schemas are applied before running collection and batch feature jobs.
- Cleaned data cache uses a 24h freshness policy; clear with:
  ```bash
  uv run python -m src.utils.cleaned_data_cache
  ```
  or delete `data/cleaned_cache/`.
- Fundamentals V1 and V2 coexist; V2 introduces a staging table and a layered approach without breaking V1.

## Recent Updates
- **Data Utilities Refactor**
  - Removed unused validation functions from exports:
    - `validate_target_quality` (target_engineering) and
    - `validate_feature_quality` (feature_engineering)
  - Removed `MLFeatureLoader` and related data-loading helpers from `src/data_utils/__init__.py` exports to simplify the API
- **Caching and Performance**
  - Added `CleanedDataCache` integration to training/prediction prep flows
  - Introduced memory-optimized training prep variant with optional memory tracking
- **Technical Indicators**
  - Streamlined console output in production batch routine
- **Prediction**
  - LightGBM all-run predictor: fixed best-run selection guard for `None` profits
- **Documentation**
  - Generalized ticker language (removed S&P 500-specific phrasing) to reflect broader ticker management

## License
See `LICENSE`.

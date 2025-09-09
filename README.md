# StockPredictor V1

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ronyka77/Stockpredictor_with_ml)

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
 - MLflow (project utilities target MLflow 3.x; see `src/utils/mlflow_integration.py`)

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

## Quickstart
- Prepare a PostgreSQL database and set environment variables (see Configuration).
- Collect data (pick one):
  - OHLCV grouped daily, last week:
    ```bash
    uv run python -m src.data_collector.polygon_data.data_pipeline
    ```
  - Technical indicators batch (features → Parquet):
    ```bash
    uv run python -m src.feature_engineering.technical_indicators.indicator_pipeline
    ```
  - Fundamentals v2 staging pipeline (async):
    ```bash
    uv run python -m src.data_collector.polygon_fundamentals_v2.run_pipeline
    ```
- Train a model (example LightGBM; see module docstring for flags/behavior):
  ```bash
  uv run python -m src.models.gradient_boosting.lightgbm_model
  ```
- Evaluate recent predictions for best LightGBM run in an experiment:
  ```python
  from src.models.predictors.lightgbm_all_run_predictor import run_all_and_export_best
  run_all_and_export_best(experiment_name="LightGBM-Experiment", days_back=30)
  ```

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

### Environment variables (.env)
You can configure the system via environment variables (loaded by `python-dotenv` where applicable):

```env
# Polygon API
POLYGON_API_KEY=your_polygon_key

# Database (required for data collection and feature pipelines)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=stock_data
DB_USER=postgres
DB_PASSWORD=your_password

# Feature engineering storage
FEATURES_STORAGE_PATH=data/features
FEATURE_VERSION=v1.0
```

Notes:
- `src/database/connection.py` and feature/data collectors fail fast if `DB_PASSWORD` is missing.
- `src/data_collector/polygon_data/data_storage.py` auto-creates the `historical_prices` and `tickers` tables when needed; SQL files in `sql/` provide broader schemas for fundamentals and staging.

## Data Collection
### Prices (Polygon Aggregates and Grouped Daily)
- Module: `src/data_collector/polygon_data/`
- What it does:
  - Fetches OHLCV via per-ticker aggregates or grouped-daily endpoints
  - Validates and normalizes data (VWAP fallback, gap/outlier checks)
  - Upserts to PostgreSQL tables: `historical_prices`, `tickers`
  - Saves pipeline execution stats under `pipeline_stats/`
- Environment:
  - `POLYGON_API_KEY` (required)
  - `DB_*` variables (required): `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
  - Default rate limit: 5 rpm with adaptive backoff
- Commands:
  - Grouped-daily (last week by default when run as a module):
    ```bash
    uv run python -m src.data_collector.polygon_data.data_pipeline
    ```
  - Programmatic full pipeline (per-ticker aggregates):
    ```python
    from datetime import date, timedelta
    from src.data_collector.polygon_data.data_pipeline import DataPipeline

    end = date.today()
    start = end - timedelta(days=30)
    with DataPipeline() as pipeline:
        stats = pipeline.run_full_pipeline(start_date=start, end_date=end, batch_size=10)
        print(stats.to_dict())
    ```
  - Single-ticker example:
    ```python
    from datetime import date, timedelta
    from src.data_collector.polygon_data.data_pipeline import DataPipeline

    end = date.today(); start = end - timedelta(days=365)
    res = DataPipeline().run_single_ticker("AAPL", start, end)
    print(res)
    ```

Notes:
- Health checks verify API connectivity, DB health, and create tables if missing.
- Storage uses safe upsert semantics and basic indexing defined in code.

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

Environment:
- `POLYGON_API_KEY` (required)
- `DB_*` variables (required)
- Schema: `sql/fundamentals_v2_schema.sql` (repository also ensures staging table exists)

### Technical Indicators Batch
- Module: `src/feature_engineering/technical_indicators/indicator_pipeline.py`
- Storage: Parquet per ticker; optional consolidation by year
### News (Polygon News API)
- Module: `src/data_collector/polygon_news/`
- What it does:
  - Collects Polygon news articles and insights
  - Supports three modes: Incremental update, Historical backfill, Targeted collection
  - Stores to tables: `polygon_news_articles`, `polygon_news_tickers`, `polygon_news_insights`
  - Provides status/health and recent-activity stats
- Environment:
  - `POLYGON_API_KEY` (required)
  - `DATABASE_URL` (optional; falls back to centralized DB config)
  - `NEWS_MAX_TICKERS`, `NEWS_DAYS_LOOKBACK`, `NEWS_RETENTION_YEARS` (optional; see `src/data_collector/config.py`)
- Run incremental/backfill driver:
  ```bash
  uv run python -m src.data_collector.polygon_news.news_pipeline
  ```
- Programmatic usage examples:
  ```python
  from datetime import datetime, timedelta, timezone
  from sqlalchemy import create_engine
  from sqlalchemy.orm import sessionmaker
  from src.data_collector.polygon_news.news_pipeline import PolygonNewsCollector
  from src.data_collector.polygon_news.models import create_tables
  from src.data_collector.config import config

  engine = create_engine(config.database_url)
  create_tables(engine)
  Session = sessionmaker(bind=engine)
  session = Session()

  collector = PolygonNewsCollector(db_session=session, polygon_api_key=config.API_KEY, requests_per_minute=config.REQUESTS_PER_MINUTE)

  # Historical backfill (1 year)
  stats = collector.collect_historical_news(max_tickers=100, years_back=1, batch_size_days=30)
  print(stats)

  # Targeted collection
  end = datetime.now(timezone.utc); start = end - timedelta(days=7)
  stats2 = collector.collect_targeted_news(["AAPL","MSFT"], start, end, limit_per_ticker=200)
  print(stats2)
  ```

Notes:
- The collector uses the same centralized logging. Storage includes health checks and can summarize recent DB stats.

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
    split_date='2025-03-15',
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
y_test = pred['y_test']
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

Training entry-points (examples):
- LightGBM: `uv run python -m src.models.gradient_boosting.lightgbm_model`
- XGBoost: `uv run python -m src.models.gradient_boosting.xgboost_model`
- RandomForest: `uv run python -m src.models.gradient_boosting.random_forest_model`

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
- Saved under `predictions/` with timestamped filenames
- Include confidence, profit analysis, and metadata

## MLflow Integration
- Utilities: `src/utils/mlflow_integration.py`, `src/utils/mlflow_utils.py`
- Common flows:
  - Experiment setup, parameter/metric logging, model artifact logging
  - Registry-friendly logging and run metadata normalization
  - Utility: normalize MLflow `meta.yaml` artifact paths relative to local workspace
    ```bash
    uv run python -m src.utils.mlflow_utils
    ```

## Evaluation and Threshold Optimization
- Threshold optimizer and profit-based evaluation:
  - `src/models/evaluation/threshold_evaluator.py`
  - Confidence-based filtering (simple, margin, leaf depth, variance, lstm_hidden)
- Custom metrics:
  - Conservative accuracy, profit per investment, success rates
- Integrated into LightGBM and MLP workflows

## Observability and Logging
- Central logging via `src/utils/logger.py` (no prints in production code). Logs under `logs/`.
- Data pipelines record execution stats under `pipeline_stats/`.
- News and data collectors perform health checks before execution.

## Testing
```bash
uv run pytest -q
```
- Pytest is configured in `pyproject.toml`
- Includes ML, data, and logging tests under `src/tests/`

## Troubleshooting
- Database connection fails: ensure `DB_PASSWORD` is set; verify connectivity with `psql` and that the user has privileges.
- Missing Polygon data: confirm `POLYGON_API_KEY`; free tier is conservatively rate-limited; backoffs are built-in.
- MLflow path issues when moving runs: run the `mlflow_utils` normalizer.
- Large feature files: enable consolidated storage, compression `snappy`, and adjust row group size in `feature_engineering/config.py`.

## Known Issues and Notes
- Ensure database schemas are applied before running collection and batch feature jobs.
- Cleaned data cache uses a 24h freshness policy; clear with:
  ```bash
  uv run python -m src.utils.cleaned_data_cache
  ```
  or delete `data/cleaned_cache/`.
- Fundamentals V1 and V2 coexist; V2 introduces a staging table and a layered approach without breaking V1.
 - `DB_PASSWORD` is required by `src/database/connection.py`; missing it will raise a ValueError during initialization.

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

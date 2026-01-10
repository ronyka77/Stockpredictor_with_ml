# Feature Engineering Process (Overview)

This document describes the high-level process, responsibilities, and quality gates for feature engineering used by the indicator pipeline.

- **Purpose**: produce a stable, well-typed feature table for ML training and prediction with reproducible calculations and clear provenance.
- **Key inputs**:
  - OHLCV price series (open/high/low/close/volume/vwap/adjusted_close)
  - Dividend records (ex_dividend_date, cash_amount)
  - Fundamental records (quarterly / annual) where available

- **Main components (code locations)**:
  - Feature orchestration: `src/data_collector/indicator_pipeline/feature_calculator.py`
  - Trend indicators (SMA/EMA/MACD/Ichimoku): `src/data_collector/indicator_pipeline/trend_indicators.py`
  - Volume & momentum indicators: `src/data_collector/indicator_pipeline/volume_indicators.py`
  - Dividend features: `src/data_collector/indicator_pipeline/dividend_features.py`
  - Raw ingestion & validation: `src/data_collector/polygon_data/data_validator.py`
  - Consolidation & export: `src/data_collector/indicator_pipeline/consolidated_storage.py`

- **Process flow**:
  1. Raw ingestion and validation (normalize timestamps, validate numeric fields).
  2. Compute core trend/volatility/momentum indicators (SMA/EMA/MACD/Ichimoku/ATR/BB/ROC/RSI/etc).
  3. Compute derived price/volume ratios and returns (Return_XD, Log_Return_XD, Volume_Price_Ratio).
  4. Append dividend-derived features mapped to trading dates.
  5. Add forward targets (Future_Close_XD, Future_High_XD) by aligned shifts.
  6. Run data-quality sanitization (dtype enforcement, plausibility checks) before persisting.
  7. Export to consolidated Parquet (year partitioned). Keep JSON fallback only for metadata, not core numeric columns.

- **Quality gates (must pass before export)**:
  - Numeric dtype enforcement for core numeric columns (coerce errors -> NaN; log failures).
  - Plausibility checks: `low <= high`, `0 <= volume <= 1e12`, `is_ex_dividend` in {0,1}.
  - No object/string values in columns that are expected numeric (fail-fast and save "bad_rows" if present).

- **How to run locally (developer)**:
  - Run the indicator pipeline for a single ticker via the batch processor (see `indicator_pipeline.py`).
  - Validate outputs by loading `ConsolidatedFeatureStorage` preview CSV (the repo exports a preview under `consolidated`).
  - Run unit tests with the repo's test runner (project policy: `uv run pytest -q`).

Keep this README short; for feature-level definitions see `FEATURE_CALCULATIONS.md`.



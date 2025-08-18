---
title: Feature Engineering Module Test Guide — StockPredictor_V1
module: feature_engineering
created_by: Ronyka77
created_on: 2025-08-15
confidence: 94/100
---

Module summary
- The `feature_engineering` package computes technical indicators, consolidates features, and stores them (notably under `technical_indicators/`). Key files: `feature_calculator.py`, `feature_storage.py`, `indicator_pipeline.py`, and multiple indicator modules (momentum, trend, volatility).

Purpose of tests
- Ensure indicator calculations are numerically correct and deterministic, feature storage roundtrip integrity (parquet), and pipeline consolidation logic.

Traceability table (requirement → tests)
| Req ID | Acceptance Criteria | Test(s) | Type | Notes |
|---|---|---:|---|---|
| FE-REQ-001 | SMA/EMA calculations produce consistent columns and deterministic outputs | `tests/unit/feature_engineering/technical_indicators/test_trend_momentum_core.py::test_sma_ema_column_naming_and_determinism` | unit | Deterministic RNG seed used |
| FE-REQ-002 | RSI calculation produces bounded [0,100] and signal columns | `tests/unit/feature_engineering/technical_indicators/test_trend_momentum_core.py::test_rsi_bounds_and_signals` | unit | Edge-case coverage implied |
| FE-REQ-003 | Feature storage parquet roundtrip preserves data and metadata | `tests/integration/feature_storage/test_parquet_roundtrip.py`, `tests/unit/feature_engineering/test_feature_storage_parquet.py` | integration/unit | Parquet engine specifics tested (pyarrow)
| FE-REQ-004 | Indicator functions deterministic given seeded RNG and produce expected column naming | `tests/unit/feature_engineering/technical_indicators/test_trend_momentum_core.py::test_sma_ema_column_naming_and_determinism`, `tests/unit/feature_engineering/technical_indicators/test_trend_momentum_core.py::test_rsi_bounds_and_signals` | unit | Determinism and bounds checks |

Key test cases & rationale
- Indicator correctness and naming conventions: ensure column names match expected patterns (`SMA_5`, `RSI_14`).
- Determinism: same input leads to identical outputs.
- Storage: parquet roundtrip tests ensure index/column preservation and metadata integrity.

Fixtures & golden datasets
- Use synthetic DataFrames generated in tests (small, deterministic) rather than large golden files.
- Parquet tests use `tmp_path` to avoid repo artifacts.

Property-based testing recommendation
- Add Hypothesis tests for indicator functions to fuzz input series shapes and NaN placements.

Priority & estimates (doc-only)
- High: consolidate naming conventions and add traceability references to test functions (0.5 day).
- Medium: add Hypothesis examples for `calculate_rsi` and `calculate_sma` (0.5 day).

How to run
- Unit tests: `uv run pytest tests/unit/feature_engineering -q -m unit`
- Parquet integration: `uv run pytest tests/integration/feature_storage -q -m integration`



---
title: Predictors Module Test Guide — StockPredictor_V1
module: predictors
created_by: Ronyka77
created_on: 2025-08-15
confidence: 94/100
---

Module summary
- Predictors group contains model wrappers and exporters for LightGBM, XGBoost, and MLP predictors. Key files: `src/models/predictors/*`, `src/models/predictors/lightgbm_predictor.py`, `mlp_predictor.py` and `lightgbm_all_run_predictor.py`.

Purpose of tests
- Verify save/export schema (Excel), prediction outputs shape and types, threshold evaluation interactions, and basic model loading stubs.

Traceability table (requirement → tests)
| Req ID | Acceptance Criteria | Test(s) | Type | Notes |
|---|---|---:|---|---|
| PR-REQ-001 | Save predictions to Excel creates expected columns and file | `tests/integration/predictors/test_excel_export_schema.py::test_excel_export_schema` | integration | Uses DummyPredictor to avoid MLflow dependencies |
| PR-REQ-002 | DummyPredictor and BasePredictor interfaces return expected prediction arrays and confidence values | `tests/integration/predictors/test_excel_export_schema.py::test_excel_export_schema` | integration | Ensures exporter reads model outputs correctly |
| PR-REQ-002 | BasePredictor utilities behave as expected for stubs and model wrappers | `tests/integration/predictors/test_excel_export_schema.py` and unit tests under `tests/unit/models` | unit/integration | Predictor save/load interfaces tested |
| PR-REQ-003 | Threshold evaluation helpers return expected metrics | `tests/unit/models/evaluation/test_threshold_evaluator*.py` | unit | ThresholdEvaluator tested thoroughly |

Key test cases & rationale
- Excel export: ensures downstream consumers (analysts) receive expected schema columns.
- Predictor predict path: return shapes and types; ensure exceptions when untrained.
- Threshold evaluator: vectorized profit/optimization behaviors tested via synthetic arrays.

Fixtures & mocks
- Use `DummyPredictor` and `MockModel` patterns to avoid MLflow or heavy model artifacts.
- Use `tmp_path` for file outputs and `monkeypatch` to change working directory.

Priority & estimates (doc-only)
- High: validate exporter contracts and ensure Excel schema remains stable (0.25 day).
- Medium: add mapping in traceability table for all predictor-related unit tests (0.5 day).

How to run
- Run predictor tests: `uv run pytest tests/integration/predictors -q -m integration`



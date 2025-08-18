---
title: MLP Module Test Guide — StockPredictor_V1
module: mlp
created_by: Ronyka77
created_on: 2025-08-15
confidence: 94/100
---

Module summary
- The `mlp` module contains model architecture, data utilities, optimization, and predictor wrappers for MLP time-series models. Key files: `src/models/time_series/mlp/mlp_architecture.py`, `src/models/time_series/mlp/mlp_optimization.py`, `src/models/predictors/mlp_predictor.py` (core predictor: `src/models/time_series/mlp/mlp_predictor.py`).

Purpose of tests
- Ensure correctness of architecture shapes and activations, scaler usage and data cleaning, predictor training loop, prediction outputs, and hyperparameter optimization objective behavior.

Traceability table (requirement → tests)
| Req ID | Acceptance Criteria | Test(s) | Type | Notes |
|---|---|---:|---|---|
| MLP-REQ-001 | MLP forward pass returns correct output shape for given input sizes | `tests/unit/mlp/test_mlp_architecture.py::test_activation_output_shape` | unit | Parametrized activations covered |
| MLP-REQ-002 | Invalid model configurations raise ValueError | `tests/unit/mlp/test_mlp_architecture.py::test_invalid_configurations_raise` | unit | Config validation |
| MLP-REQ-003 | Predictor raises when untrained and returns predictions after training | `tests/unit/mlp/test_mlp_architecture.py::test_predictor_create_model_and_predict_raises_for_untrained`, `tests/unit/mlp/test_mlp_architecture_predictor.py::test_mlppredictor_fit_predict_and_confidence` | integration/unit | Training checkpoint creation also tested |
| MLP-REQ-004 | Scaler used consistently during optimization and prediction | `tests/unit/mlp/test_mlp_optimization_scaler.py::test_prepare_data_for_training_with_scaler`, `tests/unit/mlp/test_mlp_scaler_implementation.py::test_validate_and_clean_data_method` | unit/integration | Policy: single-scaler documented in fixtures doc |
| MLP-REQ-005 | Predictor fit sets `is_trained` and creates checkpoint file; predictions return expected length | `tests/unit/mlp/test_mlp_architecture_predictor.py::test_mlppredictor_fit_predict_and_confidence`, `tests/unit/mlp/test_mlp_predictor_advanced.py::test_basic_training` | integration/unit | Checkpoint file creation validated in tests |
| MLP-REQ-006 | Hyperparameter optimization objective is callable and integrates with Optuna | `tests/unit/mlp/test_mlp_hyperparameter_optimization.py::test_mlp_hyperparameter_objective_callable` | unit | Objective factory behavior |
| MLP-REQ-007 | Hyperparameter optimization integration run produces best-trial info (slow) | `tests/unit/mlp/test_mlp_hyperparameter_optimization.py::test_mlp_hyperparameter_optimization_integration` | slow/integration | Marked `@pytest.mark.slow` |
| MLP-REQ-008 | Predictor refactored flows call validation and scaling utilities correctly | `tests/unit/mlp/test_mlp_predictor_refactored.py::TestMLPPredictorRefactored::test_predict_uses_validate_and_clean_data` | unit | Uses patching to assert internal calls |
| MLP-REQ-009 | Property-based fuzzing example validates cleaning invariants (no NaN/Inf) | `tests/unit/mlp/test_mlp_hypothesis_example.py::test_validate_and_clean_data_property` | unit/property | Hypothesis living example with bounded examples |

Key test cases & rationale
- Architecture
  - Deterministic shape checks for multiple activations and residual configurations.
- Data utilities & scaler
  - Validate `validate_and_clean_data` handles NaN/Inf and `scale_data` returns scaler when `fit_scaler=True`.
- Predictor
  - Fit loop completes for 1 epoch in fast test; predictions return correct shape and length.
- Hyperparameter optimization
  - Objective factory callable and basic integration via Optuna (short run in `@pytest.mark.slow` tests).

Fixtures needed
- `rng_seed` (ensures deterministic RNG for numpy/torch). Present in `tests/unit/mlp/conftest.py`.
- `small_df`, `small_Xy` for tiny dataset tests.

Mocking & golden datasets
- No large golden datasets required for MLP; use synthetic in-memory data in tests.

Priority & estimates (doc-only)
- High: ensure existing unit tests remain green; add traceability IDs to each test docstring or a mapping table (0.5 day).
- Medium: add Hypothesis property tests for `MLPDataUtils.validate_and_clean_data` to fuzz NaN/Inf handling (0.5 day).

Notes & deferred scenarios
- Long hyperparameter runs and full optimization should remain `@pytest.mark.slow`/nightly experiments; document resource requirements.

Examples (how to run local smoke tests)
- Run MLP unit tests: `uv run pytest tests/unit/mlp -q -m unit`



---
title: Test Fixtures Catalog — StockPredictor_V1
created_by: Ronyka77
created_on: 2025-08-15
confidence: 95/100
---

Purpose
- Centralize fixtures, naming rules, lifetimes, and golden data placement so tests are reproducible and maintainable.

Canonical fixture locations
- `tests/_fixtures/conftest.py` — project-level helper fixtures (e.g., `mock_http_client`, `permission_error_simulator`).
- `tests/_fixtures/data/` — JSON, parquet, and other small golden data files used by tests. Keep files small (<5MB preferred for CI).

Common fixtures (examples)
- `mock_http_client` (return configurable responses) — used by data collectors and API client tests.
- `no_sleep` (autouse) — disables `time.sleep` to speed retry/backoff code paths.
- `freeze_time` — deterministic time control for rate limiter and time-based logic.
- `db_session` (module-scoped for polygon_news) — uses `TEST_DATABASE_URL` from `.env` or skips when not configured.
- MLP fixtures (`rng_seed`, `small_df`, `small_Xy`, `stub_model`) — used by MLP unit tests to ensure deterministic runs.

Fixture naming & lifetime
- Prefer descriptive fixture names, avoid ambiguous names like `data`.
- Use `scope='function'` for stateful fixtures that touch external resources; use `module` or `session` only when necessary and documented.

Golden data rules
- Store golden files under `tests/_fixtures/data/`.
- Provide a short README in that folder describing each file, source, and update rationale.
- For large baseline datasets, provide a reduced subset for CI.

Single-scaler policy (MLP)
- Document-only: enforce a single scaler instance per MLP pipeline (training/tuning/eval/predict). Document places to set and pass the scaler (MLPDataUtils, predictor `.set_scaler`).

Updating fixtures & golden files
- Any golden data update must include a short rationale in the PR and a reviewer sign-off in the traceability table.

Examples
- Example `mock_http_client` usage is in `tests/_fixtures/conftest.py`.



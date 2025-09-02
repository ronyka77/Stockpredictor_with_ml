# Contributing to StockPredictor_with_ml

First off, thank you for considering a contribution! This project is an end-to-end ML system for stock prediction using Polygon.io data, feature engineering, gradient-boosted models, and neural networks, with production-oriented tooling (uv, logging, MLflow, caching, and batch processing).

Please read this guide to set up your environment, work with data safely, follow our lint/format/test workflow, and submit quality pull requests.

For project overview and commands, see README.md in the repo:
- Repository: https://github.com/ronyka77/Stockpredictor_with_ml

## Table of Contents

1. Project scope and architecture
2. Environment setup (uv)
3. Data and secrets handling
4. Running pipelines and entry points
5. Notebooks policy
6. Coding standards and tooling
7. Testing strategy
8. Git workflow and commit style
9. Pull request checklist
10. Issue reporting and feature requests
11. Documentation expectations
12. Code of Conduct

---

## 1) Project scope and architecture

This repository includes:

- Data collection (Polygon.io): OHLCV, News, Fundamentals (v1, v2 staging-first).
- Feature engineering: Technical indicators with Parquet storage and year-based consolidation.
- Data utilities: Target engineering, price normalization, cleaning, and cached ML data prep.
- Models: Gradient boosting (LightGBM/XGBoost), MLP (PyTorch) with optional CUDA.
- Evaluation: Threshold optimization and profit-based metrics.
- Infra and ops: PostgreSQL schemas (sql/), centralized logging, MLflow utilities.

Key locations:
- Data collectors: src/data_collector/
- Feature engineering: src/feature_engineering/
- Data utilities: src/data_utils/
- Models and predictors: src/models/
- Evaluation: src/models/evaluation/
- SQL schemas: sql/
- Logging, MLflow, caching utils: src/utils/

Please keep new contributions consistent with this layering.

References on structure and DS best practices:
- Project organization for DS: https://gist.github.com/ericmjl/27e50331f24db3e8f957d1fe7bbbe510
- GitHub’s CONTRIBUTING.md guidance: https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/setting-guidelines-for-repository-contributors
- Mozilla guide to drafting CONTRIBUTING.md: http://mozillascience.github.io/working-open-workshop/contributing/

---

## 2) Environment setup (uv)

We standardize on uv with Python 3.12+.

- Create and sync:
  - `uv venv`
  - `uv sync` (uses pyproject.toml and uv.lock)
- Run commands:
  - `uv run <python -m module_or_script>`
- Python version is pinned via .python-version.

Dev extras are defined in pyproject.toml. If you need additional tooling for development (e.g., ruff, black, isort, mypy, pytest-cov), add them under the dev group in pyproject.toml and run `uv sync`.

Optional GPU: Install the CUDA toolkit version that matches your chosen PyTorch build per the official compatibility matrix (or use the matching CUDA-enabled PyTorch wheel/container). See the [PyTorch compatibility matrix](https://pytorch.org/get-started/previous-versions/).

---

## 3) Data and secrets handling

Never commit secrets or large data.

- Secrets: Set via environment variables or a local .env file (not committed).
  - See .environment.example for required variables; typical ones include:
    - POLYGON_API_KEY
    - DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    - FEATURES_STORAGE_PATH, FEATURE_VERSION
- Database:
  - PostgreSQL is required for data collection and some batch processes.
  - SQL schemas live in sql/ (e.g., database_schema.sql, fundamentals_v2_schema.sql).
- Data storage:
  - Feature Parquet files and cleaned data caches are local artifacts; do not commit them.
  - Cache freshness defaults to ~24h; clear via `uv run python -m src.utils.cleaned_data_cache` or delete data/cleaned_cache/.
- Accidental secret commit: Notify maintainers immediately to rotate and purge history.

---

## 4) Running pipelines and entry points

Common entry points (examples from README):

- Grouped-daily OHLCV:
  - `uv run python -m src.data_collector.polygon_data.data_pipeline`
- Technical indicators batch:
  - `uv run python -m src.feature_engineering.technical_indicators.indicator_pipeline`
- Fundamentals v2 (async staging-first):
  - `uv run python -m src.data_collector.polygon_fundamentals_v2.run_pipeline`
- Models:
  - LightGBM: `uv run python -m src.models.gradient_boosting.lightgbm_model`
  - XGBoost: `uv run python -m src.models.gradient_boosting.xgboost_model`
  - RandomForest: `uv run python -m src.models.gradient_boosting.random_forest_model`
- Predictions (best LightGBM run via MLflow):
  - `run_all_and_export_best(...)` in `src.models.predictors.lightgbm_all_run_predictor`

Please add usage examples in module docstrings when introducing new entry points.

---

## 5) Notebooks policy

- Keep exploratory notebooks under notebooks/ with numeric prefixes (e.g., 01-eda.ipynb).
- Strip outputs before commit; avoid committing large rendered artifacts.
- Refactor reusable logic into src/ modules. Notebooks should import from src/.
- If notebooks support published results, document a script or module to reproduce outputs from raw data.

---

## 6) Coding standards and tooling

- Python version: 3.12+
- Style and quality:
  - ruff for linting (ruff check .)
  - black for formatting (black .)
  - isort for import order (isort .)
  - mypy for typing (mypy src tests)
- Logging:
  - Use centralized logger (src/utils/logger.py). No print statements in production code.
- Config and constants:
  - Keep configuration co-located with feature/collector modules where relevant (e.g., src/feature_engineering/config.py).
  - Parameterize via environment variables when dealing with credentials, paths, and rate limits.

Before pushing:
- `uv run ruff check .`
- `uv run black .`
- `uv run isort .`
- `uv run mypy src tests`
- `uv run pytest -q`

If you introduce new external dependencies, add them to pyproject.toml and run `uv sync`.

---

## 7) Testing strategy

- Use pytest. Tests should mirror the src tree under tests/.
- Add unit tests for new modules and functions; include regression tests for bug fixes.
- Prefer deterministic tests; if sample data is needed, include tiny CSVs/Parquet under tests/resources/.
- Avoid hitting external services in unit tests:
  - Mock Polygon.io and DB calls via fakes/mocks.
- Coverage:
  - `uv run pytest --cov=src --cov-report=term-missing`

---

## 8) Git workflow and commit style

- Branches from master:
  - feat/<short-name>, fix/<short-name>, docs/<short-name>, chore/<short-name>, refactor/<short-name>, test/<short-name>, ci/<short-name>
- Commits:
  - Prefer Conventional Commits style (feat:, fix:, docs:, chore:, refactor:, test:, ci:).
  - Keep commits focused and descriptive.

---

## 9) Pull request checklist

Open an issue for significant changes first to align scope. For your PR:

- [ ] Includes or updates tests for changes
- [ ] Lints/formatting pass (ruff/black/isort)
- [ ] Typing passes (mypy)
- [ ] Tests pass locally (pytest)
- [ ] Updates docs/README where behavior or usage changes
- [ ] No data or secrets committed
- [ ] For pipelines/models: includes usage notes or examples in docstrings

We typically squash-and-merge.

---

## 10) Issue reporting and feature requests

- Bug reports should include:
  - Steps to reproduce, expected vs actual behavior
  - Environment (OS, Python, key package versions)
  - Relevant logs (from logs/) and minimal data snippet or synthetic example
- Security issues:
  - Do not open a public issue. Contact maintainers privately.
- Feature requests:
  - Describe the problem and proposed solution, alternatives considered, and scope/impact.

---

## 11) Documentation expectations

- Public functions and modules: docstrings (Google or NumPy style).
- Module-level entry points should include example commands or code snippets.
- Keep README and docs accurate when changing commands, configs, or outputs.
- If you add new configs or environment variables, update .environment.example.

---

## 12) Code of Conduct

This project adheres to the Code of Conduct in CODE_OF_CONDUCT.md. By participating, you agree to uphold a welcoming, harassment-free environment for everyone. Report unacceptable behavior to the maintainers.

---

## Quick start for contributors

1. Fork and clone.
2. `uv venv && uv sync`
3. Create a feature branch.
4. Export env vars or create a local .env from .environment.example.
5. Run a small pipeline (e.g., grouped-daily OHLCV) to validate setup.
6. Make changes in src/, keep notebooks clean.
7. `uv run ruff/black/isort/mypy/pytest`
8. Open a PR and request review.

Thanks for helping improve StockPredictor_with_ml!
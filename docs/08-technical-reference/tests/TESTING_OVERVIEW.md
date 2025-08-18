---
title: Testing Overview — StockPredictor_V1
created_by: Ronyka77
created_on: 2025-08-15
confidence: 95/100
---

Purpose
- Provide a concise hub for the project's test strategy, run commands, and links to module-level test guides and templates.

Scope
- Covers test types, quality gates, how to run tests locally (Windows-friendly `uv` examples), fixture conventions, and pointers to module guides.

Test types & when to use them
- **Unit tests**: fast, isolated, system logic, numeric transforms, validators. Use `@pytest.mark.unit`.
- **Integration tests**: DB access, filesystem, end-to-end pipeline behaviors, and external API connectors. Use `@pytest.mark.integration`.
- **Slow / Performance**: expensive experiments and hyperparameter optimization runs. Use `@pytest.mark.slow` and exclude from fast runs.
- **Property-based tests**: for validators and feature calculators where space of inputs is large; use Hypothesis examples from `TEST_TEMPLATES.md`.

Quality gates (documented goals)
- Pytest is the standard test runner.
- Reproducibility: tests must use controlled seeds and `freeze_time` where applicable.
- Coverage target: global target **70%** for core logic; exceptions can be documented per module.
- Unit runtime guideline: aim for **unit tests < 10s/test** on typical developer machines.

Running tests (Windows examples)
- Run all unit tests: `uv run pytest -m unit`
- Run a single test file: `uv run pytest tests/unit/mlp/test_mlp_architecture.py` 
- Run with coverage and produce HTML: `uv run pytest -m unit --cov=src --cov-report=html`
- Run integration tests only: `uv run pytest -m integration`

Fixtures & golden data
- Canonical fixtures live under `tests/_fixtures/` and data under `tests/_fixtures/data/`.
- Golden datasets should be small for CI—keep per-fixture files < 5MB where possible and provide a reduced CI subset.

Traceability
- Module guides contain a markdown traceability table mapping requirements → test IDs → file path.

Where to start (priority)
1. Read `docs/08-technical-reference/tests/TEST_FIXTURES.md` for fixture conventions.
2. Read `docs/08-technical-reference/tests/TEST_TEMPLATES.md` for copy-paste test examples.
3. Review `docs/08-technical-reference/tests/MODULE_TEST_GUIDES/mlp.md` for a worked example based on existing tests.

Contributing tests
- Follow naming and marker conventions in `TEST_NAMING_AND_MARKERS.md`.
- Use fixtures from `TEST_FIXTURES.md` whenever possible; do not create project-wide fixtures without documenting them.
- For large golden data changes document the reason and get reviewer approval.

Contact
- Doc author: `Ronyka77` (see `.reports/tests/test-plan_2025-08-15T120000.md` for owners and timeline).



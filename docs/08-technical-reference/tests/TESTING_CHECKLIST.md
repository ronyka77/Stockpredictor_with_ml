---
title: Test Review Checklist — StockPredictor_V1
created_by: Ronyka77
created_on: 2025-08-15
confidence: 95/100
---

Purpose
- Lightweight checklist to help reviewers validate test changes for correctness, reproducibility, and maintainability before merging.

Checklist (use when reviewing test PRs)
- **Run & pass**: I ran the tests locally and they passed (or I verified CLI commands in the PR description).
- **Primary marker present**: Each new/modified test has a primary marker: `@pytest.mark.unit` or `@pytest.mark.integration`.
- **Naming & docstring**: File and test function names follow naming conventions; complex tests include a short docstring with REQ IDs.
- **Fixtures**: Reused existing fixtures when possible (`mock_http_client`, `db_session`, `freeze_time`, MLP fixtures). No ad-hoc global fixtures.
- **No hard-coded secrets**: No API keys, passwords, or real credentials in tests or fixtures.
- **Deterministic**: Random seeds set or Hypothesis settings constrained; `freeze_time` used for time-dependent tests.
- **Network usage**: Unit tests must not call external networks. If external access is required, mark as `integration` and document dependencies.
- **Resource cleanup**: Tests clean up or use temporary paths (`tmp_path`) and close DB sessions or files.
- **Runtime expectation**: Unit tests should generally be fast (<10s/test). Mark long tests `@pytest.mark.slow` and exclude from fast runs.
- **Golden/guideline data**: Golden files under `tests/_fixtures/data/` are small and include README; any updates include rationale in PR.
- **No prints; use logger**: Avoid `print()` output in tests; use `src/utils/logger.py` if needed.
- **Coverage note**: New tests include a coverage note where relevant and reference requirement IDs for traceability.
- **Parametrization**: Use `@pytest.mark.parametrize` for systematic coverage rather than many repeated tests.
- **Hypothesis**: If using property tests, include `@settings(max_examples=...)` to keep test time bounded.

Reviewer actions
- If any checklist item fails: request changes in PR and provide guidance (link to templates and fixtures docs).
- If golden data updated: require a short justification and reviewer sign-off.
- If tests are flaky or non-deterministic: require immediate remediation or quarantine via `@pytest.mark.flaky` (documented) and a follow-up ticket.

Commands (quick)
- Run unit tests: `uv run pytest -m unit`
- Run all tests with coverage: `uv run pytest --cov=src --cov-report=term-missing`



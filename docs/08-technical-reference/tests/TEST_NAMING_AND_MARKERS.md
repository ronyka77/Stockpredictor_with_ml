---
title: Test Naming Conventions & Markers — StockPredictor_V1
created_by: Ronyka77
created_on: 2025-08-15
confidence: 95/100
---

Purpose
- Define clear naming conventions for test files and functions and standardize pytest markers used across the repository to improve discoverability, maintainability, and test selection.

File naming conventions
- Use this pattern for test files: `test_<module>_<feature>_<scenario>.py`.
  - Examples: `test_mlp_architecture_forward_shape.py`, `test_polygon_news_processor_missing_fields.py`.
- Keep files focused: prefer one logical feature per test file. Very small modules may group closely related tests in a single file.

Test function naming
- Use descriptive function names starting with `test_` and reflecting expected behavior: `test_<action>_<expected_result>[_<edgecase>]`.
  - Examples: `test_validate_and_clean_data_removes_nans`, `test_store_article_returns_id_on_success`.
- Include Requirement IDs in a comment or in the docstring to aid traceability: e.g. `# REQ: PN-REQ-002`.

Docstrings & metadata
- Add a one-line docstring for complex tests explaining intent and important fixtures.
- If a test exercises multiple acceptance criteria, list IDs in the docstring.

Markers (pytest)
- Standard markers and semantics:
  - `@pytest.mark.unit`: Fast, isolated unit tests. Expected runtime: unit tests < 10s/test.
  - `@pytest.mark.integration`: Tests that touch external systems (DB, filesystem, network). Can be excluded from fast runs.
  - `@pytest.mark.slow`: Long-running experiments; keep out of fast CI and run nightly.
  - `@pytest.mark.no_network`: Tests that must not access the network; ensure network calls are mocked.

How to use markers
- Always apply one primary marker per test (`unit` or `integration`). Secondary markers (e.g., `slow`, `no_network`) are permitted.
- Example:
```python
@pytest.mark.unit
def test_some_unit_case():
    ...

@pytest.mark.integration
@pytest.mark.no_network
def test_db_integration_without_external_calls(db_session):
    ...
```

Enforcement suggestions
- Code reviewers must verify presence of primary marker.
- Use CI job filters to run `-m "not integration"` for fast checks and run integration jobs separately.
- Consider adding a simple pytest plugin or pre-commit hook that warns on tests missing markers.

Marker registration
- Register markers in `pyproject.toml` pytest section (or `pytest.ini`) to avoid warnings:
```
[tool.pytest.ini_options]
markers = [
  "unit: fast unit tests",
  "integration: integration tests touching external resources",
  "slow: long-running tests",
  "no_network: tests that must not use network",
]
```

Best practices
- Use `tmp_path` for filesystem isolation and `monkeypatch` for env changes.
- Avoid network calls in unit tests: prefer `mock_http_client` or recorded responses under `tests/_fixtures/data/`.
- Keep tests deterministic: set seeds and use `freeze_time` where applicable.



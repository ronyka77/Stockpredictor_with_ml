---
title: Polygon News Module Test Guide — StockPredictor_V1
module: polygon_news
created_by: Ronyka77
created_on: 2025-08-15
confidence: 94/100
---

Module summary
- The `polygon_news` package handles fetching, processing, validating, and storing news articles from Polygon. Key files: `news_client.py`, `processor.py`, `validator.py`, `storage.py`, `news_pipeline.py`, `ticker_integration.py`.

Purpose of tests
- Validate API client behavior, processing normalization, validation rules, DB storage logic (create/update/rollback), and pipeline batching/aggregation logic.

Traceability table (requirement → tests)
| Req ID | Acceptance Criteria | Test(s) | Type | Notes |
|---|---|---:|---|---|
| PN-REQ-001 | Extract article metadata maps required fields correctly | `tests/unit/polygon_news/test_news_client.py::test_extract_article_metadata_maps_fields_correctly` | unit | Ensures mapping from raw API payload to internal schema |
| PN-REQ-002 | Processor normalizes fields and handles missing/malformed inputs | `tests/unit/polygon_news/test_processor.py::test_process_article_full`, `test_process_article_missing_fields`, `test_process_article_malformed_types_raises` | unit | Processor behavior and error propagation |
| PN-REQ-003 | Validator scores and rejects invalid articles | `tests/unit/polygon_news/test_validator.py::*` | unit | Quality scoring, missing fields, date checks |
| PN-REQ-004 | Storage create/update/rollback behaviors | `tests/unit/polygon_news/test_storage.py::*`, `test_storage_exceptions.py::test_store_article_on_exception_rolls_back_and_returns_none` | unit | DB session faults and transactional rollback tested |
| PN-REQ-005 | Pipeline collection respects batching, skipping and error handling | `tests/unit/polygon_news/test_news_pipeline.py::*`, `tests/unit/polygon_news/test_main_and_historical.py::test_collect_historical_news_respects_batching` | unit | Pipeline aggregation and batching logic |
| PN-REQ-006 | Integration DB tests require TEST_DATABASE_URL and are skip-enabled | `tests/unit/polygon_news/conftest.py::db_session` (skip behavior) | integration | Docs explain how to configure test DB locally |

Key test cases & rationale
- `news_client`: date param mapping, empty responses, multi-ticker failure handling.
- `processor`: normalize whitespace, extract insights, handle missing fields.
- `validator`: scoring, required field checks, old/future date handling.
- `storage`: insert/update flows, batch storage stats, exception rollback.
- `news_pipeline`: orchestrates collection, handles per-article failures, respects batching limits.

Fixtures & mocks
- `db_session` (function scope) — uses `.env` `TEST_DATABASE_URL` or skips tests.
- `sample_raw_article_*` fixtures under `tests/_fixtures/data/` used for processor/validator/storage tests.
- `mock_http_client` and monkeypatching are used to stub API responses.

Golden data
- Article JSON fixtures live in `tests/_fixtures/data/` (e.g., `sample_article_full.json`). Keep files small and documented.

Priority & estimates (doc-only)
- High: Add traceability IDs to any remaining tests and ensure every critical validation rule has at least one unit test (0.5 day).
- Medium: Add integration smoke job instructions for local DB setup and example `.env` (0.5 day).

Deferred / heavy scenarios
- Full historical replays and long-running pipeline integration tests are deferred to nightly/integration environments; document resource needs.

How to run
- Unit tests: `uv run pytest tests/unit/polygon_news -q -m unit`
- DB-enabled tests: set `TEST_DATABASE_URL` in `.env` then run `uv run pytest tests/unit/polygon_news -q -m unit`



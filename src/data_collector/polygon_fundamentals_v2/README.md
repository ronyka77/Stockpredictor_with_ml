# Fundamentals V2 Pipeline

Clean, layered implementation beside the existing fundamentals pipeline. No existing files modified.

Components:

- `raw_client.py`: minimal raw HTTP client returning Polygon fundamentals JSON
- `parser.py`: maps raw JSON (nested/legacy) → DTOs (`IncomeStatement`, `BalanceSheet`, `CashFlowStatement`)
- `validator_v2.py`: wraps V1 validator and adds cross-statement integrity rules
- `repository.py`: DB staging table writer (`raw_fundamental_json`) and ticker lookup
- `collector_service.py`: orchestrates cache → client → parser → validator → repository
- `processor.py`: loads active tickers and processes sequentially
- `run_pipeline.py`: entrypoint

Run (Windows, uv):

```
uv run python -m src.data_collector.polygon_fundamentals_v2.run_pipeline
```

Schema:

- See `sql/fundamentals_v2_schema.sql`. The repository also calls `ensure_schema()` to create the staging table if missing.



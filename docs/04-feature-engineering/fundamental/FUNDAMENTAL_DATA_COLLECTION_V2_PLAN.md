# Fundamental Data Collection V2 Plan (Polygon Fundamentals)

## Overview

Fundamentals V2 is a clean, layered pipeline that coexists with V1 and focuses on:

- Cache-first raw JSON ingestion and idempotent staging
- Minimal parsing to essential DTOs with quality scoring
- Cross-statement validation
- Incremental upserts of structured facts
- Sequential processing to respect 5 req/minute limits

The V2 pipeline is implemented under `src/data_collector/polygon_fundamentals_v2/` and does not modify any V1 modules.

## Architecture (Layers)

1. `raw_client.py` → Async HTTP client returning raw Polygon fundamentals JSON with retries and rate limiting
2. `parser.py` → Maps nested/legacy JSON to DTOs (`IncomeStatement`, `BalanceSheet`, `CashFlowStatement`)
3. `validator_v2.py` → Wraps V1 validator, adds cross-statement integrity rules
4. `repository.py` → DB access; ensures schema; upserts to staging (`raw_fundamental_json`) and writes facts (`fundamental_facts_v2`)
5. `collector_service.py` → Orchestrates cache → client → parser → validator → repository; updates `tickers.has_financials`
6. `extractor.py` → Extracts structured facts from raw payloads into `fundamental_facts_v2`
7. `processor.py` → Sequentially processes active tickers using the service
8. `run_pipeline.py` → Entry point (Windows-friendly via `uv`)

## API and Rate Limiting

- Limit: 5 requests/minute (free tier)
- Client: `RawPolygonFundamentalsClient` uses a basic rate limiter and exponential backoff for 429
- Processing: Sequential, one ticker at a time
- Cache: JSON cache via `FundamentalCacheManager` (1-day freshness) checked before API calls

## Database Schema

V2 introduces a staging table to persist full raw payloads and a facts table for structured fields. The repository ensures both exist.

### Staging: `raw_fundamental_json`

```sql
CREATE TABLE IF NOT EXISTS raw_fundamental_json (
    id BIGSERIAL PRIMARY KEY,
    ticker_id INTEGER NOT NULL,
    period_end DATE,
    timeframe TEXT,
    fiscal_period TEXT,
    fiscal_year TEXT,
    filing_date DATE,
    source TEXT,
    payload_json JSONB NOT NULL,
    response_hash TEXT UNIQUE,
    ingested_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_raw_fund_json_ticker_end ON raw_fundamental_json(ticker_id, period_end DESC);
CREATE INDEX IF NOT EXISTS idx_raw_fund_json_filing ON raw_fundamental_json(filing_date);
```

### Facts: `public.fundamental_facts_v2`

Essential fields across Income Statement, Balance Sheet, Cash Flow, plus quality and completeness metrics.

```sql
CREATE TABLE IF NOT EXISTS public.fundamental_facts_v2 (
    id BIGSERIAL PRIMARY KEY,
    ticker_id INTEGER NOT NULL,
    "date" DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    filing_date DATE NULL,
    fiscal_period VARCHAR(10) NULL,
    fiscal_year VARCHAR(10) NULL,
    timeframe VARCHAR(20) NULL,
    cik VARCHAR(20) NULL,
    company_name VARCHAR(255) NULL,
    source_filing_url TEXT NULL,
    source_filing_file_url TEXT NULL,
    acceptance_datetime TIMESTAMP NULL,
    sic_code VARCHAR(10) NULL,
    sic_description VARCHAR(255) NULL,
    revenues NUMERIC(15, 2) NULL,
    cost_of_revenue NUMERIC(15, 2) NULL,
    gross_profit NUMERIC(15, 2) NULL,
    operating_expenses NUMERIC(15, 2) NULL,
    selling_general_and_administrative_expenses NUMERIC(15, 2) NULL,
    research_and_development NUMERIC(15, 2) NULL,
    operating_income_loss NUMERIC(15, 2) NULL,
    nonoperating_income_loss NUMERIC(15, 2) NULL,
    income_loss_from_continuing_operations_before_tax NUMERIC(15, 2) NULL,
    income_tax_expense_benefit NUMERIC(15, 2) NULL,
    income_loss_from_continuing_operations_after_tax NUMERIC(15, 2) NULL,
    net_income_loss NUMERIC(15, 2) NULL,
    net_income_loss_attributable_to_parent NUMERIC(15, 2) NULL,
    basic_earnings_per_share NUMERIC(10, 4) NULL,
    diluted_earnings_per_share NUMERIC(10, 4) NULL,
    basic_average_shares NUMERIC(15, 2) NULL,
    diluted_average_shares NUMERIC(15, 2) NULL,
    assets NUMERIC(15, 2) NULL,
    current_assets NUMERIC(15, 2) NULL,
    noncurrent_assets NUMERIC(15, 2) NULL,
    inventory NUMERIC(15, 2) NULL,
    other_current_assets NUMERIC(15, 2) NULL,
    fixed_assets NUMERIC(15, 2) NULL,
    other_noncurrent_assets NUMERIC(15, 2) NULL,
    liabilities NUMERIC(15, 2) NULL,
    current_liabilities NUMERIC(15, 2) NULL,
    noncurrent_liabilities NUMERIC(15, 2) NULL,
    accounts_payable NUMERIC(15, 2) NULL,
    other_current_liabilities NUMERIC(15, 2) NULL,
    long_term_debt NUMERIC(15, 2) NULL,
    other_noncurrent_liabilities NUMERIC(15, 2) NULL,
    equity NUMERIC(15, 2) NULL,
    equity_attributable_to_parent NUMERIC(15, 2) NULL,
    net_cash_flow_from_operating_activities NUMERIC(15, 2) NULL,
    net_cash_flow_from_investing_activities NUMERIC(15, 2) NULL,
    net_cash_flow_from_financing_activities NUMERIC(15, 2) NULL,
    net_cash_flow NUMERIC(15, 2) NULL,
    net_cash_flow_continuing NUMERIC(15, 2) NULL,
    net_cash_flow_from_operating_activities_continuing NUMERIC(15, 2) NULL,
    net_cash_flow_from_investing_activities_continuing NUMERIC(15, 2) NULL,
    net_cash_flow_from_financing_activities_continuing NUMERIC(15, 2) NULL,
    comprehensive_income_loss NUMERIC(15, 2) NULL,
    comprehensive_income_loss_attributable_to_parent NUMERIC(15, 2) NULL,
    other_comprehensive_income_loss NUMERIC(15, 2) NULL,
    other_comprehensive_income_loss_attributable_to_parent NUMERIC(15, 2) NULL,
    data_quality_score NUMERIC(5, 4) NULL,
    missing_data_count INT DEFAULT 0 NULL,
    direct_report_fields_count INT DEFAULT 0 NULL,
    imputed_fields_count INT DEFAULT 0 NULL,
    derived_fields_count INT DEFAULT 0 NULL,
    total_fields_count INT DEFAULT 0 NULL,
    data_completeness_percentage NUMERIC(5, 2) NULL,
    completeness_score NUMERIC(5, 4) GENERATED ALWAYS AS (
        CASE WHEN total_fields_count = 0 THEN 0.0
             ELSE (total_fields_count - missing_data_count)::NUMERIC / total_fields_count::NUMERIC
        END
    ) STORED,
    data_source_confidence NUMERIC(3, 2) NULL,
    CONSTRAINT fundamental_facts_v2_unique UNIQUE (ticker_id, date),
    CONSTRAINT fk_facts_v2_ticker_id FOREIGN KEY (ticker_id) REFERENCES public.tickers(id) ON DELETE CASCADE
);
```

## Orchestration and Flow

For each ticker:

1. Repository `ensure_schema()` creates tables if missing
2. Lookup `ticker_id`; if unknown → return error
3. Cache-first: try JSON cache via `FundamentalCacheManager`
4. If cache miss, fetch raw JSON via `RawPolygonFundamentalsClient` (429-aware)
5. Persist raw payload to `raw_fundamental_json` with `response_hash` and `source` (cache/api)
6. Extract structured facts to `fundamental_facts_v2` via `extractor.py` (idempotent upsert)
7. Parse to DTOs and validate via `validator_v2` (adds cross-statement warnings)
8. If no results, mark `tickers.has_financials = false`

## Processor and Runner

- `FundamentalsProcessor.process_all()` loads active tickers (sequential) and calls service per ticker
- `run_pipeline.py` is the entry point; logs summary and elapsed time

## Usage (Windows, uv)

```bash
uv run python -m src.data_collector.polygon_fundamentals_v2.run_pipeline
```

Programmatic examples:

```python
from src.data_collector.polygon_fundamentals_v2.collector_service import FundamentalsCollectorService

svc = FundamentalsCollectorService()
result = await svc.process_ticker("AAPL")
print(result)
```

## Validation and Data Quality

- V2 validator wraps V1 and adds:
  - Period alignment checks (IS↔BS, IS↔CF)
  - Balance sheet equation tolerance (~2%)
  - Cash flow component sum tolerance (~5%)
  - Produces warnings (does not fail ingestion unless base validation fails)

## Caching

- Uses `FundamentalCacheManager` (1-day freshness)
- Cache names like `TICKER_financials_YYYYMMDD.json` under `data/cache/fundamentals/`
- Service records `source` as `cache` or `api` in staging

## Performance Characteristics

- Sequential processing; safe for 5 req/minute
- Cache significantly reduces API calls on repeated runs
- Idempotent staging via `response_hash`

## Differences vs V1

- Adds raw JSON staging table and idempotent ingestion
- Clean layering (client → parser → validator → repository → service → processor → runner)
- Cross-statement validation in V2
- Facts table `fundamental_facts_v2` separate from V1 storage
- Coexists with V1; no modifications to V1 modules

## Implementation Status

- All V2 components implemented and runnable
- Entry point available via `uv run` command above



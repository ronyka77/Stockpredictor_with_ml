-- Fundamentals V2 staging schema (non-breaking; coexists with current tables)

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


-- Structured facts table for Fundamentals V2
CREATE TABLE IF NOT EXISTS public.fundamental_facts_v2 (
    id BIGSERIAL PRIMARY KEY,
    ticker_id INTEGER NOT NULL,
    "date" DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- metadata
    filing_date DATE NULL,
    fiscal_period VARCHAR(10) NULL,
    fiscal_year VARCHAR(10) NULL,
    timeframe VARCHAR(20) NULL,
    
    source_filing_url TEXT NULL,
    source_filing_file_url TEXT NULL,
    acceptance_datetime TIMESTAMP NULL,
    
    
    -- income statement
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

    -- balance sheet
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

    -- cash flow
    net_cash_flow_from_operating_activities NUMERIC(15, 2) NULL,
    net_cash_flow_from_investing_activities NUMERIC(15, 2) NULL,
    net_cash_flow_from_financing_activities NUMERIC(15, 2) NULL,
    net_cash_flow NUMERIC(15, 2) NULL,
    net_cash_flow_continuing NUMERIC(15, 2) NULL,
    net_cash_flow_from_operating_activities_continuing NUMERIC(15, 2) NULL,
    net_cash_flow_from_investing_activities_continuing NUMERIC(15, 2) NULL,
    net_cash_flow_from_financing_activities_continuing NUMERIC(15, 2) NULL,

    -- comprehensive income
    comprehensive_income_loss NUMERIC(15, 2) NULL,
    comprehensive_income_loss_attributable_to_parent NUMERIC(15, 2) NULL,
    other_comprehensive_income_loss NUMERIC(15, 2) NULL,
    other_comprehensive_income_loss_attributable_to_parent NUMERIC(15, 2) NULL,

    -- quality / completeness
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

CREATE INDEX IF NOT EXISTS idx_facts_v2_ticker_date ON public.fundamental_facts_v2 (ticker_id, date);
CREATE INDEX IF NOT EXISTS idx_facts_v2_filing_date ON public.fundamental_facts_v2 (filing_date);

-- Optional: keep updated_at in sync if trigger function exists
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_proc p JOIN pg_namespace n ON p.pronamespace = n.oid
        WHERE p.proname = 'update_updated_at_column' AND n.nspname = 'public'
    ) THEN
        CREATE TRIGGER update_facts_v2_updated_at
        BEFORE UPDATE ON public.fundamental_facts_v2
        FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
    END IF;
END
$$;


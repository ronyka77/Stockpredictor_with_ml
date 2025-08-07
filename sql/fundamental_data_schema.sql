-- Raw fundamental data storage table for optimized collection
CREATE TABLE IF NOT EXISTS public.raw_fundamental_data (
    id serial4 NOT NULL,
    ticker_id int4 NOT NULL,
    "date" date NOT NULL,
    created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
    updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
    filing_date date NULL,
    fiscal_period varchar(10) NULL,
    fiscal_year varchar(10) NULL,
    timeframe varchar(20) NULL,
    cik varchar(20) NULL,
    company_name varchar(255) NULL,
    source_filing_url text NULL,
    source_filing_file_url text NULL,
    -- Income Statement Fields
    revenues numeric(15, 2) NULL,
    cost_of_revenue numeric(15, 2) NULL,
    gross_profit numeric(15, 2) NULL,
    operating_expenses numeric(15, 2) NULL,
    selling_general_and_administrative_expenses numeric(15, 2) NULL,
    research_and_development numeric(15, 2) NULL,
    operating_income_loss numeric(15, 2) NULL,
    nonoperating_income_loss numeric(15, 2) NULL,
    income_loss_from_continuing_operations_before_tax numeric(15, 2) NULL,
    income_tax_expense_benefit numeric(15, 2) NULL,
    income_loss_from_continuing_operations_after_tax numeric(15, 2) NULL,
    net_income_loss numeric(15, 2) NULL,
    net_income_loss_attributable_to_parent numeric(15, 2) NULL,
    basic_earnings_per_share numeric(10, 4) NULL,
    diluted_earnings_per_share numeric(10, 4) NULL,
    basic_average_shares numeric(15, 2) NULL,
    diluted_average_shares numeric(15, 2) NULL,
    -- Balance Sheet Fields
    assets numeric(15, 2) NULL,
    current_assets numeric(15, 2) NULL,
    noncurrent_assets numeric(15, 2) NULL,
    inventory numeric(15, 2) NULL,
    other_current_assets numeric(15, 2) NULL,
    fixed_assets numeric(15, 2) NULL,
    other_noncurrent_assets numeric(15, 2) NULL,
    liabilities numeric(15, 2) NULL,
    current_liabilities numeric(15, 2) NULL,
    noncurrent_liabilities numeric(15, 2) NULL,
    accounts_payable numeric(15, 2) NULL,
    other_current_liabilities numeric(15, 2) NULL,
    long_term_debt numeric(15, 2) NULL,
    other_noncurrent_liabilities numeric(15, 2) NULL,
    equity numeric(15, 2) NULL,
    equity_attributable_to_parent numeric(15, 2) NULL,
    -- Cash Flow Fields
    net_cash_flow_from_operating_activities numeric(15, 2) NULL,
    net_cash_flow_from_investing_activities numeric(15, 2) NULL,
    net_cash_flow_from_financing_activities numeric(15, 2) NULL,
    net_cash_flow numeric(15, 2) NULL,
    net_cash_flow_continuing numeric(15, 2) NULL,
    net_cash_flow_from_operating_activities_continuing numeric(15, 2) NULL,
    net_cash_flow_from_investing_activities_continuing numeric(15, 2) NULL,
    net_cash_flow_from_financing_activities_continuing numeric(15, 2) NULL,
    -- Comprehensive Income Fields
    comprehensive_income_loss numeric(15, 2) NULL,
    comprehensive_income_loss_attributable_to_parent numeric(15, 2) NULL,
    other_comprehensive_income_loss numeric(15, 2) NULL,
    other_comprehensive_income_loss_attributable_to_parent numeric(15, 2) NULL,
    -- Data Quality
    data_quality_score numeric(5, 4) NULL,
    missing_data_count int4 DEFAULT 0 NULL,
    CONSTRAINT raw_fundamental_data_pkey PRIMARY KEY (id),
    CONSTRAINT unique_ticker_date_raw_fundamental UNIQUE (ticker_id, date),
    CONSTRAINT valid_date_raw_fundamental CHECK ((date >= '2020-01-01'::date)),
    CONSTRAINT fk_raw_fundamental_ticker_id FOREIGN KEY (ticker_id) REFERENCES public.tickers(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_raw_fundamental_created_at ON public.raw_fundamental_data USING btree (created_at);
CREATE INDEX IF NOT EXISTS idx_raw_fundamental_date ON public.raw_fundamental_data USING btree (date);
CREATE INDEX IF NOT EXISTS idx_raw_fundamental_ticker_id ON public.raw_fundamental_data USING btree (ticker_id);
CREATE INDEX IF NOT EXISTS idx_raw_fundamental_ticker_date ON public.raw_fundamental_data USING btree (ticker_id, date);
CREATE INDEX IF NOT EXISTS idx_raw_fundamental_filing_date ON public.raw_fundamental_data USING btree (filing_date);

-- Table Triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_raw_fundamental_updated_at 
    BEFORE UPDATE ON public.raw_fundamental_data 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column(); 


-- Add missing columns to existing table
ALTER TABLE public.raw_fundamental_data 
ADD COLUMN IF NOT EXISTS benefits_costs_expenses numeric(15, 2) NULL,
ADD COLUMN IF NOT EXISTS income_loss_from_continuing_operations_before_tax numeric(15, 2) NULL,
ADD COLUMN IF NOT EXISTS income_loss_from_continuing_operations_after_tax numeric(15, 2) NULL,
ADD COLUMN IF NOT EXISTS preferred_stock_dividends_and_other_adjustments numeric(15, 2) NULL,
ADD COLUMN IF NOT EXISTS acceptance_datetime timestamp NULL,
ADD COLUMN IF NOT EXISTS sic_code varchar(10) NULL,
ADD COLUMN IF NOT EXISTS sic_description varchar(255) NULL,
ADD COLUMN IF NOT EXISTS direct_report_fields_count int4 DEFAULT 0 NULL,
ADD COLUMN IF NOT EXISTS imputed_fields_count int4 DEFAULT 0 NULL,
ADD COLUMN IF NOT EXISTS derived_fields_count int4 DEFAULT 0 NULL,
ADD COLUMN IF NOT EXISTS total_fields_count int4 DEFAULT 0 NULL,
ADD COLUMN IF NOT EXISTS data_completeness_percentage numeric(5,2) NULL;

-- Add indexes for new columns
CREATE INDEX IF NOT EXISTS idx_raw_fundamental_sic_code 
ON public.raw_fundamental_data USING btree (sic_code);

-- Add computed column for data completeness
ALTER TABLE public.raw_fundamental_data 
ADD COLUMN completeness_score numeric(5,4) GENERATED ALWAYS AS (
    CASE 
        WHEN total_fields_count = 0 THEN 0.0
        ELSE (total_fields_count - missing_data_count)::numeric / total_fields_count
    END
) STORED;
-- DROP SCHEMA public;

CREATE SCHEMA IF NOT EXISTS public AUTHORIZATION pg_database_owner;

-- DROP FUNCTION public.migrate_technical_features();

CREATE OR REPLACE FUNCTION public.migrate_technical_features()
 RETURNS text
 LANGUAGE plpgsql
AS $function$
DECLARE
    result_text TEXT := '';
BEGIN
    -- This function would contain the migration logic
    -- from the old technical_features table to the new structure
    
    result_text := 'Migration function created. Implement migration logic as needed.';
    RETURN result_text;
END;
$function$
;

-- DROP FUNCTION public.update_updated_at_column();

CREATE OR REPLACE FUNCTION public.update_updated_at_column()
 RETURNS trigger
 LANGUAGE plpgsql
AS $function$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$function$
;

-- DROP SEQUENCE public.feature_calculation_jobs_id_seq;

CREATE SEQUENCE public.feature_calculation_jobs_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.fundamental_facts_v2_id_seq;

CREATE SEQUENCE public.fundamental_facts_v2_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.historical_prices_id_seq;

CREATE SEQUENCE public.historical_prices_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.polygon_news_articles_id_seq;

CREATE SEQUENCE public.polygon_news_articles_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.polygon_news_insights_id_seq;

CREATE SEQUENCE public.polygon_news_insights_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.polygon_news_tickers_id_seq;

CREATE SEQUENCE public.polygon_news_tickers_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.raw_fundamental_data_id_seq;

CREATE SEQUENCE public.raw_fundamental_data_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.raw_fundamental_json_id_seq;

CREATE SEQUENCE public.raw_fundamental_json_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.system_config_id_seq;

CREATE SEQUENCE public.system_config_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.technical_features_momentum_id_seq;

CREATE SEQUENCE public.technical_features_momentum_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.technical_features_trend_id_seq;

CREATE SEQUENCE public.technical_features_trend_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.technical_features_volatility_id_seq;

CREATE SEQUENCE public.technical_features_volatility_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.technical_features_volume_id_seq;

CREATE SEQUENCE public.technical_features_volume_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.tickers_id_seq;

CREATE SEQUENCE public.tickers_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;-- public.feature_calculation_jobs definition

-- Drop table

-- DROP TABLE public.feature_calculation_jobs;

CREATE TABLE public.feature_calculation_jobs ( id serial4 NOT NULL, job_id varchar(50) NOT NULL, ticker varchar(10) NULL, start_date date NULL, end_date date NULL, feature_categories _text NULL, status varchar(20) DEFAULT 'pending'::character varying NOT NULL, total_features_calculated int4 DEFAULT 0 NULL, total_warnings int4 DEFAULT 0 NULL, error_message text NULL, started_at timestamp NULL, completed_at timestamp NULL, created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL, CONSTRAINT feature_calculation_jobs_job_id_key UNIQUE (job_id), CONSTRAINT feature_calculation_jobs_pkey PRIMARY KEY (id), CONSTRAINT feature_calculation_jobs_status_check CHECK (((status)::text = ANY ((ARRAY['pending'::character varying, 'running'::character varying, 'completed'::character varying, 'failed'::character varying, 'cancelled'::character varying])::text[]))));
CREATE INDEX idx_feature_calculation_jobs_created_at ON public.feature_calculation_jobs USING btree (created_at);
CREATE INDEX idx_feature_calculation_jobs_job_id ON public.feature_calculation_jobs USING btree (job_id);
CREATE INDEX idx_feature_calculation_jobs_status ON public.feature_calculation_jobs USING btree (status);
CREATE INDEX idx_feature_calculation_jobs_ticker ON public.feature_calculation_jobs USING btree (ticker);


-- public.historical_prices definition

-- Drop table

-- DROP TABLE public.historical_prices;

CREATE TABLE public.historical_prices ( id serial4 NOT NULL, ticker varchar(10) NOT NULL, "date" date NOT NULL, "open" numeric NOT NULL, high numeric NOT NULL, low numeric NOT NULL, "close" numeric NOT NULL, volume int8 NOT NULL, adjusted_close numeric NULL, vwap numeric NULL, created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL, updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL, CONSTRAINT historical_prices_pkey PRIMARY KEY (id), CONSTRAINT historical_prices_ticker_date_key UNIQUE (ticker, date));
CREATE INDEX idx_historical_prices_created_at ON public.historical_prices USING btree (created_at);
CREATE INDEX idx_historical_prices_date ON public.historical_prices USING btree (date);
CREATE INDEX idx_historical_prices_ticker ON public.historical_prices USING btree (ticker);
CREATE INDEX idx_historical_prices_ticker_date ON public.historical_prices USING btree (ticker, date);

-- Table Triggers

create trigger update_historical_prices_updated_at before
update
    on
    public.historical_prices for each row execute function update_updated_at_column();


-- public.polygon_news_articles definition

-- Drop table

-- DROP TABLE public.polygon_news_articles;

CREATE TABLE public.polygon_news_articles ( id serial4 NOT NULL, polygon_id varchar(100) NOT NULL, title varchar(1000) NOT NULL, description text NULL, article_url varchar(2000) NOT NULL, amp_url varchar(2000) NULL, image_url varchar(2000) NULL, author varchar(200) NULL, published_utc timestamptz NOT NULL, publisher_name varchar(200) NULL, publisher_homepage_url varchar(500) NULL, publisher_logo_url varchar(500) NULL, publisher_favicon_url varchar(500) NULL, keywords _text NULL, created_at timestamptz DEFAULT CURRENT_TIMESTAMP NULL, updated_at timestamptz DEFAULT CURRENT_TIMESTAMP NULL, is_processed bool DEFAULT false NULL, processing_errors text NULL, quality_score float8 NULL, relevance_score float8 NULL, CONSTRAINT polygon_news_articles_pkey PRIMARY KEY (id), CONSTRAINT polygon_news_articles_polygon_id_key UNIQUE (polygon_id), CONSTRAINT valid_quality_score CHECK (((quality_score >= (0)::double precision) AND (quality_score <= (1)::double precision))), CONSTRAINT valid_relevance_score CHECK (((relevance_score >= (0)::double precision) AND (relevance_score <= (1)::double precision))));
CREATE INDEX idx_polygon_news_polygon_id ON public.polygon_news_articles USING btree (polygon_id);
CREATE INDEX idx_polygon_news_processed ON public.polygon_news_articles USING btree (is_processed);
CREATE INDEX idx_polygon_news_published_utc ON public.polygon_news_articles USING btree (published_utc);
CREATE INDEX idx_polygon_news_publisher ON public.polygon_news_articles USING btree (publisher_name);
CREATE INDEX idx_polygon_news_quality ON public.polygon_news_articles USING btree (quality_score);


-- public.raw_fundamental_json definition

-- Drop table

-- DROP TABLE public.raw_fundamental_json;

CREATE TABLE public.raw_fundamental_json ( id bigserial NOT NULL, ticker_id int4 NOT NULL, period_end date NULL, timeframe text NULL, fiscal_period text NULL, fiscal_year text NULL, filing_date date NULL, "source" text NULL, payload_json jsonb NOT NULL, response_hash text NULL, ingested_at timestamptz DEFAULT CURRENT_TIMESTAMP NULL, CONSTRAINT raw_fundamental_json_pkey PRIMARY KEY (id), CONSTRAINT raw_fundamental_json_response_hash_key UNIQUE (response_hash));
CREATE INDEX idx_raw_fund_json_filing ON public.raw_fundamental_json USING btree (filing_date);
CREATE INDEX idx_raw_fund_json_ticker_end ON public.raw_fundamental_json USING btree (ticker_id, period_end DESC);


-- public.system_config definition

-- Drop table

-- DROP TABLE public.system_config;

CREATE TABLE public.system_config ( id serial4 NOT NULL, config_key varchar(100) NOT NULL, config_value text NOT NULL, config_type varchar(20) DEFAULT 'string'::character varying NULL, description text NULL, is_active bool DEFAULT true NULL, created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL, updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL, CONSTRAINT system_config_config_key_key UNIQUE (config_key), CONSTRAINT system_config_pkey PRIMARY KEY (id));

-- Table Triggers

create trigger update_system_config_updated_at before
update
    on
    public.system_config for each row execute function update_updated_at_column();


-- public.technical_features_momentum definition

-- Drop table

-- DROP TABLE public.technical_features_momentum;

CREATE TABLE public.technical_features_momentum ( id serial4 NOT NULL, ticker varchar(10) NOT NULL, "date" date NOT NULL, rsi_14 numeric(5, 2) NULL, rsi_14_overbought bool NULL, rsi_14_oversold bool NULL, rsi_21 numeric(5, 2) NULL, rsi_21_overbought bool NULL, rsi_21_oversold bool NULL, stoch_k numeric(5, 2) NULL, stoch_d numeric(5, 2) NULL, stoch_overbought bool NULL, stoch_oversold bool NULL, stoch_k_above_d bool NULL, stoch_crossover numeric(5, 2) NULL, roc_5 numeric(10, 4) NULL, roc_10 numeric(10, 4) NULL, roc_20 numeric(10, 4) NULL, roc_5_positive bool NULL, roc_10_positive bool NULL, roc_20_positive bool NULL, williams_r_14 numeric(5, 2) NULL, williams_r_14_overbought bool NULL, williams_r_14_oversold bool NULL, williams_r_14_neutral bool NULL, williams_r_21 numeric(5, 2) NULL, williams_r_21_overbought bool NULL, williams_r_21_oversold bool NULL, williams_r_21_neutral bool NULL, momentum_5 numeric(10, 4) NULL, momentum_10 numeric(10, 4) NULL, momentum_20 numeric(10, 4) NULL, momentum_trend numeric(10, 4) NULL, momentum_acceleration numeric(10, 4) NULL, momentum_divergence numeric(10, 4) NULL, quality_score numeric(5, 2) NULL, created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL, CONSTRAINT momentum_features_unique UNIQUE (ticker, date), CONSTRAINT technical_features_momentum_pkey PRIMARY KEY (id));
CREATE INDEX idx_momentum_date ON public.technical_features_momentum USING btree (date);
CREATE INDEX idx_momentum_ticker ON public.technical_features_momentum USING btree (ticker);
CREATE INDEX idx_momentum_ticker_date ON public.technical_features_momentum USING btree (ticker, date);


-- public.technical_features_trend definition

-- Drop table

-- DROP TABLE public.technical_features_trend;

CREATE TABLE public.technical_features_trend ( id serial4 NOT NULL, ticker varchar(10) NOT NULL, "date" date NOT NULL, sma_5 numeric(10, 4) NULL, sma_10 numeric(10, 4) NULL, sma_20 numeric(10, 4) NULL, sma_50 numeric(10, 4) NULL, sma_100 numeric(10, 4) NULL, sma_200 numeric(10, 4) NULL, ema_5 numeric(10, 4) NULL, ema_10 numeric(10, 4) NULL, ema_20 numeric(10, 4) NULL, ema_50 numeric(10, 4) NULL, ema_100 numeric(10, 4) NULL, ema_200 numeric(10, 4) NULL, macd numeric(10, 4) NULL, macd_signal numeric(10, 4) NULL, macd_histogram numeric(10, 4) NULL, macd_above_signal bool NULL, macd_crossover numeric(10, 4) NULL, ichimoku_tenkan numeric(10, 4) NULL, ichimoku_kijun numeric(10, 4) NULL, ichimoku_senkou_a numeric(10, 4) NULL, ichimoku_senkou_b numeric(10, 4) NULL, ichimoku_chikou numeric(10, 4) NULL, ichimoku_tenkan_above_kijun bool NULL, ichimoku_price_above_cloud bool NULL, ichimoku_price_below_cloud bool NULL, ichimoku_cloud_green bool NULL, ichimoku_cloud_thickness numeric(10, 4) NULL, quality_score numeric(5, 2) NULL, created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL, CONSTRAINT technical_features_trend_pkey PRIMARY KEY (id), CONSTRAINT trend_features_unique UNIQUE (ticker, date));
CREATE INDEX idx_trend_date ON public.technical_features_trend USING btree (date);
CREATE INDEX idx_trend_ticker ON public.technical_features_trend USING btree (ticker);
CREATE INDEX idx_trend_ticker_date ON public.technical_features_trend USING btree (ticker, date);


-- public.technical_features_volatility definition

-- Drop table

-- DROP TABLE public.technical_features_volatility;

CREATE TABLE public.technical_features_volatility ( id serial4 NOT NULL, ticker varchar(10) NOT NULL, "date" date NOT NULL, bb_lower numeric(10, 4) NULL, bb_middle numeric(10, 4) NULL, bb_upper numeric(10, 4) NULL, bb_bandwidth numeric(10, 4) NULL, bb_percent numeric(10, 4) NULL, bb_width numeric(10, 4) NULL, bb_above_upper bool NULL, bb_below_lower bool NULL, bb_squeeze bool NULL, bb_position numeric(5, 4) NULL, atr_14 numeric(10, 4) NULL, atr_21 numeric(10, 4) NULL, atr_normalized_14 numeric(10, 4) NULL, atr_normalized_21 numeric(10, 4) NULL, volatility_std_10 numeric(10, 4) NULL, volatility_std_20 numeric(10, 4) NULL, volatility_annualized_10 numeric(10, 4) NULL, volatility_annualized_20 numeric(10, 4) NULL, hl_volatility_10 numeric(10, 4) NULL, hl_volatility_20 numeric(10, 4) NULL, parkinson_vol_10 numeric(10, 4) NULL, parkinson_vol_20 numeric(10, 4) NULL, vol_regime_high bool NULL, vol_regime_low bool NULL, vol_trend_rising bool NULL, quality_score numeric(5, 2) NULL, created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL, CONSTRAINT technical_features_volatility_pkey PRIMARY KEY (id), CONSTRAINT volatility_features_unique UNIQUE (ticker, date));
CREATE INDEX idx_volatility_date ON public.technical_features_volatility USING btree (date);
CREATE INDEX idx_volatility_ticker ON public.technical_features_volatility USING btree (ticker);
CREATE INDEX idx_volatility_ticker_date ON public.technical_features_volatility USING btree (ticker, date);


-- public.technical_features_volume definition

-- Drop table

-- DROP TABLE public.technical_features_volume;

CREATE TABLE public.technical_features_volume ( id serial4 NOT NULL, ticker varchar(10) NOT NULL, "date" date NOT NULL, obv_millions numeric(10, 4) NULL, obv_sma_10_millions numeric(10, 4) NULL, obv_sma_20_millions numeric(10, 4) NULL, obv_momentum_5_millions numeric(10, 4) NULL, obv_momentum_10_millions numeric(10, 4) NULL, vpt_millions numeric(10, 4) NULL, vpt_sma_10_millions numeric(10, 4) NULL, vpt_sma_20_millions numeric(10, 4) NULL, vpt_roc_5 numeric(10, 4) NULL, vpt_roc_10 numeric(10, 4) NULL, ad_line_millions numeric(10, 4) NULL, ad_sma_10_millions numeric(10, 4) NULL, ad_sma_20_millions numeric(10, 4) NULL, ad_momentum_5_millions numeric(10, 4) NULL, ad_momentum_10_millions numeric(10, 4) NULL, ad_oscillator_millions numeric(10, 4) NULL, volume_ma_5_thousands numeric(10, 4) NULL, volume_ma_10_thousands numeric(10, 4) NULL, volume_ma_20_thousands numeric(10, 4) NULL, volume_ratio_5 numeric(10, 4) NULL, volume_ratio_10 numeric(10, 4) NULL, volume_ratio_20 numeric(10, 4) NULL, mfi_14 numeric(5, 2) NULL, mfi_overbought bool NULL, mfi_oversold bool NULL, volume_spike bool NULL, volume_above_average bool NULL, volume_trend numeric(10, 4) NULL, volume_weighted_price numeric(10, 4) NULL, volume_oscillator numeric(10, 4) NULL, volume_rate_of_change numeric(10, 4) NULL, quality_score numeric(5, 2) NULL, created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL, CONSTRAINT technical_features_volume_pkey PRIMARY KEY (id), CONSTRAINT volume_features_unique UNIQUE (ticker, date));
CREATE INDEX idx_volume_date ON public.technical_features_volume USING btree (date);
CREATE INDEX idx_volume_ticker ON public.technical_features_volume USING btree (ticker);
CREATE INDEX idx_volume_ticker_date ON public.technical_features_volume USING btree (ticker, date);


-- public.tickers definition

-- Drop table

-- DROP TABLE public.tickers;

CREATE TABLE public.tickers ( id serial4 NOT NULL, ticker varchar(10) NOT NULL, "name" varchar(255) NULL, market varchar(50) DEFAULT 'stocks'::character varying NULL, locale varchar(10) DEFAULT 'us'::character varying NULL, primary_exchange varchar(50) NULL, currency_name varchar(10) NULL, active bool DEFAULT true NOT NULL, "type" varchar(50) NULL, market_cap float8 NULL, weighted_shares_outstanding float8 NULL, round_lot int4 NULL, last_updated timestamp DEFAULT CURRENT_TIMESTAMP NOT NULL, created_at timestamp DEFAULT CURRENT_TIMESTAMP NOT NULL, cik varchar(20) NULL, composite_figi varchar(20) NULL, share_class_figi varchar(20) NULL, sic_code varchar(10) NULL, sic_description varchar(255) NULL, ticker_root varchar(10) NULL, total_employees int4 NULL, list_date date NULL, has_financials bool DEFAULT true NOT NULL, CONSTRAINT tickers_pkey PRIMARY KEY (id), CONSTRAINT tickers_ticker_key UNIQUE (ticker));
CREATE INDEX idx_tickers_active ON public.tickers USING btree (active);
CREATE INDEX idx_tickers_cik ON public.tickers USING btree (cik);
CREATE INDEX idx_tickers_composite_figi ON public.tickers USING btree (composite_figi);
CREATE INDEX idx_tickers_list_date ON public.tickers USING btree (list_date);
CREATE INDEX idx_tickers_market ON public.tickers USING btree (market);
CREATE INDEX idx_tickers_share_class_figi ON public.tickers USING btree (share_class_figi);
CREATE INDEX idx_tickers_sic_code ON public.tickers USING btree (sic_code);
CREATE INDEX idx_tickers_ticker ON public.tickers USING btree (ticker);


-- public.fundamental_facts_v2 definition

-- Drop table

-- DROP TABLE public.fundamental_facts_v2;

CREATE TABLE public.fundamental_facts_v2 ( id bigserial NOT NULL, ticker_id int4 NOT NULL, "date" date NOT NULL, created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL, updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL, filing_date date NULL, fiscal_period varchar(10) NULL, fiscal_year varchar(10) NULL, timeframe varchar(20) NULL, source_filing_url text NULL, source_filing_file_url text NULL, acceptance_datetime timestamp NULL, revenues numeric(15, 2) NULL, cost_of_revenue numeric(15, 2) NULL, gross_profit numeric(15, 2) NULL, operating_expenses numeric(15, 2) NULL, selling_general_and_administrative_expenses numeric(15, 2) NULL, research_and_development numeric(15, 2) NULL, operating_income_loss numeric(15, 2) NULL, nonoperating_income_loss numeric(15, 2) NULL, income_loss_from_continuing_operations_before_tax numeric(15, 2) NULL, income_tax_expense_benefit numeric(15, 2) NULL, income_loss_from_continuing_operations_after_tax numeric(15, 2) NULL, net_income_loss numeric(15, 2) NULL, net_income_loss_attributable_to_parent numeric(15, 2) NULL, basic_earnings_per_share numeric(10, 4) NULL, diluted_earnings_per_share numeric(10, 4) NULL, basic_average_shares numeric(15, 2) NULL, diluted_average_shares numeric(15, 2) NULL, assets numeric(15, 2) NULL, current_assets numeric(15, 2) NULL, noncurrent_assets numeric(15, 2) NULL, inventory numeric(15, 2) NULL, other_current_assets numeric(15, 2) NULL, fixed_assets numeric(15, 2) NULL, other_noncurrent_assets numeric(15, 2) NULL, liabilities numeric(15, 2) NULL, current_liabilities numeric(15, 2) NULL, noncurrent_liabilities numeric(15, 2) NULL, accounts_payable numeric(15, 2) NULL, other_current_liabilities numeric(15, 2) NULL, long_term_debt numeric(15, 2) NULL, other_noncurrent_liabilities numeric(15, 2) NULL, equity numeric(15, 2) NULL, equity_attributable_to_parent numeric(15, 2) NULL, net_cash_flow_from_operating_activities numeric(15, 2) NULL, net_cash_flow_from_investing_activities numeric(15, 2) NULL, net_cash_flow_from_financing_activities numeric(15, 2) NULL, net_cash_flow numeric(15, 2) NULL, net_cash_flow_continuing numeric(15, 2) NULL, net_cash_flow_from_operating_activities_continuing numeric(15, 2) NULL, net_cash_flow_from_investing_activities_continuing numeric(15, 2) NULL, net_cash_flow_from_financing_activities_continuing numeric(15, 2) NULL, comprehensive_income_loss numeric(15, 2) NULL, comprehensive_income_loss_attributable_to_parent numeric(15, 2) NULL, other_comprehensive_income_loss numeric(15, 2) NULL, other_comprehensive_income_loss_attributable_to_parent numeric(15, 2) NULL, data_quality_score numeric(5, 4) NULL, missing_data_count int4 DEFAULT 0 NULL, direct_report_fields_count int4 DEFAULT 0 NULL, imputed_fields_count int4 DEFAULT 0 NULL, derived_fields_count int4 DEFAULT 0 NULL, total_fields_count int4 DEFAULT 0 NULL, data_completeness_percentage numeric(5, 2) NULL, completeness_score numeric(5, 4) GENERATED ALWAYS AS (
CASE
    WHEN total_fields_count = 0 THEN 0.0
    ELSE (total_fields_count - missing_data_count)::numeric / total_fields_count::numeric
END) STORED NULL, data_source_confidence numeric(3, 2) NULL, CONSTRAINT fundamental_facts_v2_pkey PRIMARY KEY (id), CONSTRAINT fundamental_facts_v2_unique UNIQUE (ticker_id, date), CONSTRAINT fk_facts_v2_ticker_id FOREIGN KEY (ticker_id) REFERENCES public.tickers(id) ON DELETE CASCADE);


-- public.polygon_news_insights definition

-- Drop table

-- DROP TABLE public.polygon_news_insights;

CREATE TABLE public.polygon_news_insights ( id serial4 NOT NULL, article_id int4 NULL, sentiment varchar(20) NULL, sentiment_reasoning text NULL, insight_type varchar(50) NULL, insight_value text NULL, confidence_score float8 NULL, created_at timestamptz DEFAULT CURRENT_TIMESTAMP NULL, CONSTRAINT polygon_news_insights_pkey PRIMARY KEY (id), CONSTRAINT valid_confidence CHECK (((confidence_score >= (0)::double precision) AND (confidence_score <= (1)::double precision))), CONSTRAINT polygon_news_insights_article_id_fkey FOREIGN KEY (article_id) REFERENCES public.polygon_news_articles(id) ON DELETE CASCADE);
CREATE INDEX idx_polygon_insights_article ON public.polygon_news_insights USING btree (article_id);
CREATE INDEX idx_polygon_insights_sentiment ON public.polygon_news_insights USING btree (sentiment);
CREATE INDEX idx_polygon_insights_type ON public.polygon_news_insights USING btree (insight_type);


-- public.polygon_news_tickers definition

-- Drop table

-- DROP TABLE public.polygon_news_tickers;

CREATE TABLE public.polygon_news_tickers ( id serial4 NOT NULL, article_id int4 NULL, ticker varchar(10) NOT NULL, created_at timestamptz DEFAULT CURRENT_TIMESTAMP NULL, CONSTRAINT polygon_news_tickers_article_id_ticker_key UNIQUE (article_id, ticker), CONSTRAINT polygon_news_tickers_pkey PRIMARY KEY (id), CONSTRAINT polygon_news_tickers_article_id_fkey FOREIGN KEY (article_id) REFERENCES public.polygon_news_articles(id) ON DELETE CASCADE);
CREATE INDEX idx_polygon_tickers_article ON public.polygon_news_tickers USING btree (article_id);
CREATE INDEX idx_polygon_tickers_ticker ON public.polygon_news_tickers USING btree (ticker);


-- public.raw_fundamental_data definition

-- Drop table

-- DROP TABLE public.raw_fundamental_data;

CREATE TABLE public.raw_fundamental_data ( id serial4 NOT NULL, ticker_id int4 NOT NULL, "date" date NOT NULL, created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL, updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL, filing_date date NULL, fiscal_period varchar(10) NULL, fiscal_year varchar(10) NULL, timeframe varchar(20) NULL, cik varchar(20) NULL, company_name varchar(255) NULL, source_filing_url text NULL, source_filing_file_url text NULL, revenues numeric(15, 2) NULL, cost_of_revenue numeric(15, 2) NULL, gross_profit numeric(15, 2) NULL, operating_expenses numeric(15, 2) NULL, selling_general_and_administrative_expenses numeric(15, 2) NULL, research_and_development numeric(15, 2) NULL, operating_income_loss numeric(15, 2) NULL, nonoperating_income_loss numeric(15, 2) NULL, income_loss_from_continuing_operations_before_tax numeric(15, 2) NULL, income_tax_expense_benefit numeric(15, 2) NULL, income_loss_from_continuing_operations_after_tax numeric(15, 2) NULL, net_income_loss numeric(15, 2) NULL, net_income_loss_attributable_to_parent numeric(15, 2) NULL, basic_earnings_per_share numeric(10, 4) NULL, diluted_earnings_per_share numeric(10, 4) NULL, basic_average_shares numeric(15, 2) NULL, diluted_average_shares numeric(15, 2) NULL, assets numeric(15, 2) NULL, current_assets numeric(15, 2) NULL, noncurrent_assets numeric(15, 2) NULL, inventory numeric(15, 2) NULL, other_current_assets numeric(15, 2) NULL, fixed_assets numeric(15, 2) NULL, other_noncurrent_assets numeric(15, 2) NULL, liabilities numeric(15, 2) NULL, current_liabilities numeric(15, 2) NULL, noncurrent_liabilities numeric(15, 2) NULL, accounts_payable numeric(15, 2) NULL, other_current_liabilities numeric(15, 2) NULL, long_term_debt numeric(15, 2) NULL, other_noncurrent_liabilities numeric(15, 2) NULL, equity numeric(15, 2) NULL, equity_attributable_to_parent numeric(15, 2) NULL, net_cash_flow_from_operating_activities numeric(15, 2) NULL, net_cash_flow_from_investing_activities numeric(15, 2) NULL, net_cash_flow_from_financing_activities numeric(15, 2) NULL, net_cash_flow numeric(15, 2) NULL, net_cash_flow_continuing numeric(15, 2) NULL, net_cash_flow_from_operating_activities_continuing numeric(15, 2) NULL, net_cash_flow_from_investing_activities_continuing numeric(15, 2) NULL, net_cash_flow_from_financing_activities_continuing numeric(15, 2) NULL, comprehensive_income_loss numeric(15, 2) NULL, comprehensive_income_loss_attributable_to_parent numeric(15, 2) NULL, other_comprehensive_income_loss numeric(15, 2) NULL, other_comprehensive_income_loss_attributable_to_parent numeric(15, 2) NULL, data_quality_score numeric(5, 4) NULL, missing_data_count int4 DEFAULT 0 NULL, benefits_costs_expenses numeric(15, 2) NULL, preferred_stock_dividends_and_other_adjustments numeric(15, 2) NULL, acceptance_datetime timestamp NULL, sic_code varchar(10) NULL, sic_description varchar(255) NULL, direct_report_fields_count int4 DEFAULT 0 NULL, imputed_fields_count int4 DEFAULT 0 NULL, derived_fields_count int4 DEFAULT 0 NULL, total_fields_count int4 DEFAULT 0 NULL, data_completeness_percentage numeric(5, 2) NULL, completeness_score numeric(5, 4) GENERATED ALWAYS AS (
CASE
    WHEN total_fields_count = 0 THEN 0.0
    ELSE (total_fields_count - missing_data_count)::numeric / total_fields_count::numeric
END) STORED NULL, data_source_confidence numeric(3, 2) NULL, 
CONSTRAINT raw_fundamental_data_pkey PRIMARY KEY (id), 
CONSTRAINT unique_ticker_date_raw_fundamental UNIQUE (ticker_id, date), 
CONSTRAINT valid_date_raw_fundamental CHECK ((date >= '2020-01-01'::date)), 
CONSTRAINT fk_raw_fundamental_ticker_id FOREIGN KEY (ticker_id) 
    REFERENCES public.tickers(id) ON DELETE CASCADE);
CREATE INDEX idx_raw_fundamental_created_at ON public.raw_fundamental_data USING btree (created_at);
CREATE INDEX idx_raw_fundamental_date ON public.raw_fundamental_data USING btree (date);
CREATE INDEX idx_raw_fundamental_filing_date ON public.raw_fundamental_data USING btree (filing_date);
CREATE INDEX idx_raw_fundamental_sic_code ON public.raw_fundamental_data USING btree (sic_code);
CREATE INDEX idx_raw_fundamental_ticker_date ON public.raw_fundamental_data USING btree (ticker_id, date);
CREATE INDEX idx_raw_fundamental_ticker_id ON public.raw_fundamental_data USING btree (ticker_id);

-- Table Triggers

create trigger update_raw_fundamental_updated_at before
update
    on
    public.raw_fundamental_data for each row execute function update_updated_at_column();


-- public.v_feature_summary source

CREATE OR REPLACE VIEW public.v_feature_summary
AS SELECT COALESCE(t.ticker, m.ticker, vol.ticker, v.ticker) AS ticker,
    COALESCE(t.date, m.date, vol.date, v.date) AS date,
        CASE
            WHEN t.ticker IS NOT NULL THEN 27
            ELSE 0
        END AS trend_features,
        CASE
            WHEN m.ticker IS NOT NULL THEN 45
            ELSE 0
        END AS momentum_features,
        CASE
            WHEN vol.ticker IS NOT NULL THEN 55
            ELSE 0
        END AS volatility_features,
        CASE
            WHEN v.ticker IS NOT NULL THEN 50
            ELSE 0
        END AS volume_features,
        CASE
            WHEN t.ticker IS NOT NULL THEN 27
            ELSE 0
        END +
        CASE
            WHEN m.ticker IS NOT NULL THEN 45
            ELSE 0
        END +
        CASE
            WHEN vol.ticker IS NOT NULL THEN 55
            ELSE 0
        END +
        CASE
            WHEN v.ticker IS NOT NULL THEN 50
            ELSE 0
        END AS total_features,
    (COALESCE(t.quality_score, 0::numeric) + COALESCE(m.quality_score, 0::numeric) + COALESCE(vol.quality_score, 0::numeric) + COALESCE(v.quality_score, 0::numeric)) / NULLIF(
        CASE
            WHEN t.quality_score IS NOT NULL THEN 1
            ELSE 0
        END +
        CASE
            WHEN m.quality_score IS NOT NULL THEN 1
            ELSE 0
        END +
        CASE
            WHEN vol.quality_score IS NOT NULL THEN 1
            ELSE 0
        END +
        CASE
            WHEN v.quality_score IS NOT NULL THEN 1
            ELSE 0
        END, 0)::numeric AS avg_quality_score,
    GREATEST(COALESCE(t.created_at, '1900-01-01 00:00:00'::timestamp without time zone), COALESCE(m.created_at, '1900-01-01 00:00:00'::timestamp without time zone), COALESCE(vol.created_at, '1900-01-01 00:00:00'::timestamp without time zone), COALESCE(v.created_at, '1900-01-01 00:00:00'::timestamp without time zone)) AS last_calculated
   FROM technical_features_trend t
     FULL JOIN technical_features_momentum m ON t.ticker::text = m.ticker::text AND t.date = m.date
     FULL JOIN technical_features_volatility vol ON COALESCE(t.ticker, m.ticker)::text = vol.ticker::text AND COALESCE(t.date, m.date) = vol.date
     FULL JOIN technical_features_volume v ON COALESCE(t.ticker, m.ticker, vol.ticker)::text = v.ticker::text AND COALESCE(t.date, m.date, vol.date) = v.date
  WHERE COALESCE(t.ticker, m.ticker, vol.ticker, v.ticker) IS NOT NULL AND COALESCE(t.date, m.date, vol.date, v.date) IS NOT NULL
  ORDER BY (COALESCE(t.ticker, m.ticker, vol.ticker, v.ticker)), (COALESCE(t.date, m.date, vol.date, v.date));


-- public.v_table_sizes source

CREATE OR REPLACE VIEW public.v_table_sizes
AS SELECT schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(((schemaname::text || '.'::text) || tablename::text)::regclass)) AS size,
    pg_total_relation_size(((schemaname::text || '.'::text) || tablename::text)::regclass) AS size_bytes
   FROM pg_tables
  WHERE tablename ~~ 'technical_features%'::text
  ORDER BY (pg_total_relation_size(((schemaname::text || '.'::text) || tablename::text)::regclass)) DESC;


-- public.v_technical_features_unified source

CREATE OR REPLACE VIEW public.v_technical_features_unified
AS SELECT t.ticker,
    t.date,
    t.sma_20,
    t.sma_50,
    t.ema_20,
    t.ema_50,
    t.macd,
    t.macd_signal,
    t.macd_above_signal,
    m.rsi_14,
    m.rsi_14_overbought,
    m.rsi_14_oversold,
    m.stoch_k,
    m.stoch_d,
    m.stoch_overbought,
    m.stoch_oversold,
    v.bb_upper,
    v.bb_lower,
    v.bb_above_upper,
    v.bb_below_lower,
    v.atr_14,
    v.volatility_std_20,
    vol.obv_millions,
    vol.volume_ma_10_thousands,
    vol.mfi_14,
    vol.volume_spike,
    vol.volume_above_average,
    t.quality_score AS trend_quality,
    m.quality_score AS momentum_quality,
    v.quality_score AS volatility_quality,
    vol.quality_score AS volume_quality,
    GREATEST(t.created_at, m.created_at, v.created_at, vol.created_at) AS last_updated
   FROM technical_features_trend t
     FULL JOIN technical_features_momentum m ON t.ticker::text = m.ticker::text AND t.date = m.date
     FULL JOIN technical_features_volatility v ON t.ticker::text = v.ticker::text AND t.date = v.date
     FULL JOIN technical_features_volume vol ON t.ticker::text = vol.ticker::text AND t.date = vol.date
  ORDER BY t.ticker, t.date;
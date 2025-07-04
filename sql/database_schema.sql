-- DROP SCHEMA public;

CREATE SCHEMA public AUTHORIZATION pg_database_owner;

-- DROP SEQUENCE public.audit_log_id_seq;

CREATE SEQUENCE public.audit_log_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.data_quality_metrics_id_seq;

CREATE SEQUENCE public.data_quality_metrics_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.feature_calculation_jobs_id_seq;

CREATE SEQUENCE public.feature_calculation_jobs_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.fundamental_growth_metrics_id_seq;

CREATE SEQUENCE public.fundamental_growth_metrics_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.fundamental_ratios_id_seq;

CREATE SEQUENCE public.fundamental_ratios_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.fundamental_scores_id_seq;

CREATE SEQUENCE public.fundamental_scores_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.fundamental_sector_analysis_id_seq;

CREATE SEQUENCE public.fundamental_sector_analysis_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
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
-- DROP SEQUENCE public.prediction_results_id_seq;

CREATE SEQUENCE public.prediction_results_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
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
-- DROP SEQUENCE public.ticker_cache_id_seq;

CREATE SEQUENCE public.ticker_cache_id_seq
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
	NO CYCLE;-- public.audit_log definition

-- Drop table

-- DROP TABLE public.audit_log;

CREATE TABLE public.audit_log (
	id serial4 NOT NULL,
	table_name varchar(100) NULL,
	operation varchar(20) NULL,
	record_id int4 NULL,
	old_values jsonb NULL,
	new_values jsonb NULL,
	changed_by varchar(100) DEFAULT 'system'::character varying NULL,
	changed_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	change_reason text NULL,
	CONSTRAINT audit_log_pkey PRIMARY KEY (id)
);
CREATE INDEX idx_audit_log_changed_at ON public.audit_log USING btree (changed_at);
CREATE INDEX idx_audit_log_operation ON public.audit_log USING btree (operation);
CREATE INDEX idx_audit_log_table_name ON public.audit_log USING btree (table_name);


-- public.data_quality_metrics definition

-- Drop table

-- DROP TABLE public.data_quality_metrics;

CREATE TABLE public.data_quality_metrics (
	id serial4 NOT NULL,
	ticker varchar(10) NOT NULL,
	"date" date NOT NULL,
	metric_type varchar(50) NOT NULL,
	metric_value numeric(10, 4) NULL,
	metric_details jsonb NULL,
	severity varchar(20) DEFAULT 'info'::character varying NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT data_quality_metrics_pkey PRIMARY KEY (id),
	CONSTRAINT data_quality_metrics_severity_check CHECK (((severity)::text = ANY ((ARRAY['info'::character varying, 'warning'::character varying, 'error'::character varying, 'critical'::character varying])::text[])))
);
CREATE INDEX idx_data_quality_metrics_date ON public.data_quality_metrics USING btree (date);
CREATE INDEX idx_data_quality_metrics_severity ON public.data_quality_metrics USING btree (severity);
CREATE INDEX idx_data_quality_metrics_ticker ON public.data_quality_metrics USING btree (ticker);
CREATE INDEX idx_data_quality_metrics_type ON public.data_quality_metrics USING btree (metric_type);


-- public.feature_calculation_jobs definition

-- Drop table

-- DROP TABLE public.feature_calculation_jobs;

CREATE TABLE public.feature_calculation_jobs (
	id serial4 NOT NULL,
	job_id varchar(50) NOT NULL,
	ticker varchar(10) NULL,
	start_date date NULL,
	end_date date NULL,
	feature_categories _text NULL,
	status varchar(20) DEFAULT 'pending'::character varying NOT NULL,
	total_features_calculated int4 DEFAULT 0 NULL,
	total_warnings int4 DEFAULT 0 NULL,
	error_message text NULL,
	started_at timestamp NULL,
	completed_at timestamp NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT feature_calculation_jobs_job_id_key UNIQUE (job_id),
	CONSTRAINT feature_calculation_jobs_pkey PRIMARY KEY (id),
	CONSTRAINT feature_calculation_jobs_status_check CHECK (((status)::text = ANY ((ARRAY['pending'::character varying, 'running'::character varying, 'completed'::character varying, 'failed'::character varying, 'cancelled'::character varying])::text[])))
);
CREATE INDEX idx_feature_calculation_jobs_created_at ON public.feature_calculation_jobs USING btree (created_at);
CREATE INDEX idx_feature_calculation_jobs_job_id ON public.feature_calculation_jobs USING btree (job_id);
CREATE INDEX idx_feature_calculation_jobs_status ON public.feature_calculation_jobs USING btree (status);
CREATE INDEX idx_feature_calculation_jobs_ticker ON public.feature_calculation_jobs USING btree (ticker);


-- public.fundamental_growth_metrics definition

-- Drop table

-- DROP TABLE public.fundamental_growth_metrics;

CREATE TABLE public.fundamental_growth_metrics (
	id serial4 NOT NULL,
	ticker varchar(10) NOT NULL,
	"date" date NOT NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	revenue_growth_1y numeric(10, 4) NULL,
	revenue_growth_3y numeric(10, 4) NULL,
	earnings_growth_1y numeric(10, 4) NULL,
	earnings_growth_3y numeric(10, 4) NULL,
	book_value_growth_1y numeric(10, 4) NULL,
	book_value_growth_3y numeric(10, 4) NULL,
	asset_growth_1y numeric(10, 4) NULL,
	asset_growth_3y numeric(10, 4) NULL,
	asset_turnover numeric(10, 4) NULL,
	inventory_turnover numeric(10, 4) NULL,
	receivables_turnover numeric(10, 4) NULL,
	working_capital_turnover numeric(10, 4) NULL,
	cash_conversion_cycle numeric(10, 4) NULL,
	days_sales_outstanding numeric(10, 4) NULL,
	days_inventory_outstanding numeric(10, 4) NULL,
	data_quality_score numeric(5, 4) NULL,
	missing_data_count int4 DEFAULT 0 NULL,
	CONSTRAINT fundamental_growth_metrics_pkey PRIMARY KEY (id),
	CONSTRAINT unique_ticker_date_growth UNIQUE (ticker, date),
	CONSTRAINT valid_date_growth CHECK ((date >= '2020-01-01'::date)),
	CONSTRAINT valid_ticker_growth CHECK ((length((ticker)::text) >= 1))
);
CREATE INDEX idx_fundamental_growth_created_at ON public.fundamental_growth_metrics USING btree (created_at);
CREATE INDEX idx_fundamental_growth_date ON public.fundamental_growth_metrics USING btree (date);
CREATE INDEX idx_fundamental_growth_ticker ON public.fundamental_growth_metrics USING btree (ticker);
CREATE INDEX idx_fundamental_growth_ticker_date ON public.fundamental_growth_metrics USING btree (ticker, date);

-- Table Triggers

create trigger update_fundamental_growth_updated_at before
update
    on
    public.fundamental_growth_metrics for each row execute function update_updated_at_column();


-- public.fundamental_ratios definition

-- Drop table

-- DROP TABLE public.fundamental_ratios;

CREATE TABLE public.fundamental_ratios (
	id serial4 NOT NULL,
	ticker varchar(10) NOT NULL,
	"date" date NOT NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	pe_ratio numeric(10, 4) NULL,
	pb_ratio numeric(10, 4) NULL,
	ps_ratio numeric(10, 4) NULL,
	ev_ebitda numeric(10, 4) NULL,
	peg_ratio numeric(10, 4) NULL,
	roe numeric(10, 4) NULL,
	roa numeric(10, 4) NULL,
	roi numeric(10, 4) NULL,
	gross_margin numeric(10, 4) NULL,
	operating_margin numeric(10, 4) NULL,
	net_margin numeric(10, 4) NULL,
	current_ratio numeric(10, 4) NULL,
	quick_ratio numeric(10, 4) NULL,
	cash_ratio numeric(10, 4) NULL,
	debt_to_equity numeric(10, 4) NULL,
	interest_coverage numeric(10, 4) NULL,
	debt_to_assets numeric(10, 4) NULL,
	data_quality_score numeric(5, 4) NULL,
	missing_data_count int4 DEFAULT 0 NULL,
	CONSTRAINT fundamental_ratios_pkey PRIMARY KEY (id),
	CONSTRAINT unique_ticker_date_ratios UNIQUE (ticker, date),
	CONSTRAINT valid_date_ratios CHECK ((date >= '2020-01-01'::date)),
	CONSTRAINT valid_ticker_ratios CHECK ((length((ticker)::text) >= 1))
);
CREATE INDEX idx_fundamental_ratios_created_at ON public.fundamental_ratios USING btree (created_at);
CREATE INDEX idx_fundamental_ratios_date ON public.fundamental_ratios USING btree (date);
CREATE INDEX idx_fundamental_ratios_ticker ON public.fundamental_ratios USING btree (ticker);
CREATE INDEX idx_fundamental_ratios_ticker_date ON public.fundamental_ratios USING btree (ticker, date);

-- Table Triggers

create trigger update_fundamental_ratios_updated_at before
update
    on
    public.fundamental_ratios for each row execute function update_updated_at_column();


-- public.fundamental_scores definition

-- Drop table

-- DROP TABLE public.fundamental_scores;

CREATE TABLE public.fundamental_scores (
	id serial4 NOT NULL,
	ticker varchar(10) NOT NULL,
	"date" date NOT NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	altman_z_score numeric(10, 4) NULL,
	piotroski_f_score int4 NULL,
	ohlson_o_score numeric(10, 4) NULL,
	working_capital_ratio numeric(10, 4) NULL,
	financial_leverage numeric(10, 4) NULL,
	market_based_financial_leverage numeric(10, 4) NULL,
	debt_to_market_cap_ratio numeric(10, 4) NULL,
	price_to_earnings_ratio numeric(10, 4) NULL,
	price_to_sales_ratio numeric(10, 4) NULL,
	market_to_book_ratio numeric(10, 4) NULL,
	market_cap_to_assets_ratio numeric(10, 4) NULL,
	revenue_per_employee numeric(15, 2) NULL,
	market_cap_per_employee numeric(15, 2) NULL,
	enterprise_value_proxy numeric(15, 2) NULL,
	ev_to_revenue_ratio numeric(10, 4) NULL,
	financial_health_composite numeric(10, 4) NULL,
	quality_composite numeric(10, 4) NULL,
	bankruptcy_risk_composite numeric(10, 4) NULL,
	piotroski_roa_positive bool NULL,
	piotroski_cfo_positive bool NULL,
	piotroski_roa_improved bool NULL,
	piotroski_cfo_vs_roa bool NULL,
	piotroski_debt_decreased bool NULL,
	piotroski_current_ratio_improved bool NULL,
	piotroski_shares_outstanding bool NULL,
	piotroski_gross_margin_improved bool NULL,
	piotroski_asset_turnover_improved bool NULL,
	data_quality_score numeric(5, 4) NULL,
	missing_data_count int4 DEFAULT 0 NULL,
	CONSTRAINT fundamental_scores_pkey PRIMARY KEY (id),
	CONSTRAINT unique_ticker_date_scores UNIQUE (ticker, date),
	CONSTRAINT valid_composite_scores CHECK ((((financial_health_composite >= (0)::numeric) AND (financial_health_composite <= (1)::numeric)) OR ((financial_health_composite IS NULL) AND ((quality_composite >= (0)::numeric) AND (quality_composite <= (1)::numeric))) OR ((quality_composite IS NULL) AND ((bankruptcy_risk_composite >= (0)::numeric) AND (bankruptcy_risk_composite <= (1)::numeric))) OR (bankruptcy_risk_composite IS NULL))),
	CONSTRAINT valid_date_scores CHECK ((date >= '2020-01-01'::date)),
	CONSTRAINT valid_market_to_book CHECK (((market_to_book_ratio > (0)::numeric) OR (market_to_book_ratio IS NULL))),
	CONSTRAINT valid_pe_ratio CHECK (((price_to_earnings_ratio > (0)::numeric) OR (price_to_earnings_ratio IS NULL))),
	CONSTRAINT valid_piotroski_score CHECK (((piotroski_f_score >= 0) AND (piotroski_f_score <= 9))),
	CONSTRAINT valid_ticker_scores CHECK ((length((ticker)::text) >= 1))
);
CREATE INDEX idx_fundamental_scores_altman_z ON public.fundamental_scores USING btree (altman_z_score);
CREATE INDEX idx_fundamental_scores_bankruptcy_risk ON public.fundamental_scores USING btree (bankruptcy_risk_composite);
CREATE INDEX idx_fundamental_scores_created_at ON public.fundamental_scores USING btree (created_at);
CREATE INDEX idx_fundamental_scores_date ON public.fundamental_scores USING btree (date);
CREATE INDEX idx_fundamental_scores_financial_health ON public.fundamental_scores USING btree (financial_health_composite);
CREATE INDEX idx_fundamental_scores_market_leverage ON public.fundamental_scores USING btree (market_based_financial_leverage);
CREATE INDEX idx_fundamental_scores_market_to_book ON public.fundamental_scores USING btree (market_to_book_ratio);
CREATE INDEX idx_fundamental_scores_ohlson_o ON public.fundamental_scores USING btree (ohlson_o_score);
CREATE INDEX idx_fundamental_scores_pe_ratio ON public.fundamental_scores USING btree (price_to_earnings_ratio);
CREATE INDEX idx_fundamental_scores_piotroski_f ON public.fundamental_scores USING btree (piotroski_f_score);
CREATE INDEX idx_fundamental_scores_quality ON public.fundamental_scores USING btree (quality_composite);
CREATE INDEX idx_fundamental_scores_ticker ON public.fundamental_scores USING btree (ticker);
CREATE INDEX idx_fundamental_scores_ticker_date ON public.fundamental_scores USING btree (ticker, date);

-- Table Triggers

create trigger update_fundamental_scores_updated_at before
update
    on
    public.fundamental_scores for each row execute function update_updated_at_column();


-- public.fundamental_sector_analysis definition

-- Drop table

-- DROP TABLE public.fundamental_sector_analysis;

CREATE TABLE public.fundamental_sector_analysis (
	id serial4 NOT NULL,
	ticker varchar(10) NOT NULL,
	"date" date NOT NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	gics_sector varchar(50) NULL,
	gics_industry_group varchar(100) NULL,
	gics_industry varchar(100) NULL,
	gics_sub_industry varchar(100) NULL,
	pe_sector_percentile numeric(5, 2) NULL,
	pb_sector_percentile numeric(5, 2) NULL,
	ps_sector_percentile numeric(5, 2) NULL,
	roe_sector_percentile numeric(5, 2) NULL,
	roa_sector_percentile numeric(5, 2) NULL,
	debt_to_equity_sector_percentile numeric(5, 2) NULL,
	pe_industry_percentile numeric(5, 2) NULL,
	pb_industry_percentile numeric(5, 2) NULL,
	roe_industry_percentile numeric(5, 2) NULL,
	pe_market_percentile numeric(5, 2) NULL,
	pb_market_percentile numeric(5, 2) NULL,
	roe_market_percentile numeric(5, 2) NULL,
	pe_relative_to_sector numeric(10, 4) NULL,
	pb_relative_to_sector numeric(10, 4) NULL,
	ps_relative_to_sector numeric(10, 4) NULL,
	roe_relative_to_sector numeric(10, 4) NULL,
	roa_relative_to_sector numeric(10, 4) NULL,
	sector_median_pe numeric(10, 4) NULL,
	sector_median_pb numeric(10, 4) NULL,
	sector_median_ps numeric(10, 4) NULL,
	sector_median_roe numeric(10, 4) NULL,
	sector_median_roa numeric(10, 4) NULL,
	industry_median_pe numeric(10, 4) NULL,
	industry_median_pb numeric(10, 4) NULL,
	industry_median_roe numeric(10, 4) NULL,
	market_median_pe numeric(10, 4) NULL,
	market_median_pb numeric(10, 4) NULL,
	market_median_roe numeric(10, 4) NULL,
	sector_rank int4 NULL,
	industry_rank int4 NULL,
	market_rank int4 NULL,
	sector_total_companies int4 NULL,
	industry_total_companies int4 NULL,
	market_total_companies int4 NULL,
	data_quality_score numeric(5, 4) NULL,
	missing_data_count int4 DEFAULT 0 NULL,
	CONSTRAINT fundamental_sector_analysis_pkey PRIMARY KEY (id),
	CONSTRAINT unique_ticker_date_sector UNIQUE (ticker, date),
	CONSTRAINT valid_date_sector CHECK ((date >= '2020-01-01'::date)),
	CONSTRAINT valid_percentiles CHECK (((pe_sector_percentile >= (0)::numeric) AND (pe_sector_percentile <= (100)::numeric) AND (pb_sector_percentile >= (0)::numeric) AND (pb_sector_percentile <= (100)::numeric) AND (roe_sector_percentile >= (0)::numeric) AND (roe_sector_percentile <= (100)::numeric))),
	CONSTRAINT valid_ticker_sector CHECK ((length((ticker)::text) >= 1))
);
CREATE INDEX idx_fundamental_sector_created_at ON public.fundamental_sector_analysis USING btree (created_at);
CREATE INDEX idx_fundamental_sector_date ON public.fundamental_sector_analysis USING btree (date);
CREATE INDEX idx_fundamental_sector_gics_industry ON public.fundamental_sector_analysis USING btree (gics_industry_group);
CREATE INDEX idx_fundamental_sector_gics_sector ON public.fundamental_sector_analysis USING btree (gics_sector);
CREATE INDEX idx_fundamental_sector_ticker ON public.fundamental_sector_analysis USING btree (ticker);
CREATE INDEX idx_fundamental_sector_ticker_date ON public.fundamental_sector_analysis USING btree (ticker, date);

-- Table Triggers

create trigger update_fundamental_sector_updated_at before
update
    on
    public.fundamental_sector_analysis for each row execute function update_updated_at_column();


-- public.historical_prices definition

-- Drop table

-- DROP TABLE public.historical_prices;

CREATE TABLE public.historical_prices (
	id serial4 NOT NULL,
	ticker varchar(10) NOT NULL,
	"date" date NOT NULL,
	"open" numeric NOT NULL,
	high numeric NOT NULL,
	low numeric NOT NULL,
	"close" numeric NOT NULL,
	volume int8 NOT NULL,
	adjusted_close numeric NULL,
	vwap numeric NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT historical_prices_pkey PRIMARY KEY (id),
	CONSTRAINT historical_prices_ticker_date_key UNIQUE (ticker, date)
);
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

CREATE TABLE public.polygon_news_articles (
	id serial4 NOT NULL,
	polygon_id varchar(100) NOT NULL,
	title varchar(1000) NOT NULL,
	description text NULL,
	article_url varchar(2000) NOT NULL,
	amp_url varchar(2000) NULL,
	image_url varchar(2000) NULL,
	author varchar(200) NULL,
	published_utc timestamptz NOT NULL,
	publisher_name varchar(200) NULL,
	publisher_homepage_url varchar(500) NULL,
	publisher_logo_url varchar(500) NULL,
	publisher_favicon_url varchar(500) NULL,
	keywords _text NULL,
	created_at timestamptz DEFAULT CURRENT_TIMESTAMP NULL,
	updated_at timestamptz DEFAULT CURRENT_TIMESTAMP NULL,
	is_processed bool DEFAULT false NULL,
	processing_errors text NULL,
	quality_score float8 NULL,
	relevance_score float8 NULL,
	CONSTRAINT polygon_news_articles_pkey PRIMARY KEY (id),
	CONSTRAINT polygon_news_articles_polygon_id_key UNIQUE (polygon_id),
	CONSTRAINT valid_quality_score CHECK (((quality_score >= (0)::double precision) AND (quality_score <= (1)::double precision))),
	CONSTRAINT valid_relevance_score CHECK (((relevance_score >= (0)::double precision) AND (relevance_score <= (1)::double precision)))
);
CREATE INDEX idx_polygon_news_polygon_id ON public.polygon_news_articles USING btree (polygon_id);
CREATE INDEX idx_polygon_news_processed ON public.polygon_news_articles USING btree (is_processed);
CREATE INDEX idx_polygon_news_published_utc ON public.polygon_news_articles USING btree (published_utc);
CREATE INDEX idx_polygon_news_publisher ON public.polygon_news_articles USING btree (publisher_name);
CREATE INDEX idx_polygon_news_quality ON public.polygon_news_articles USING btree (quality_score);


-- public.prediction_results definition

-- Drop table

-- DROP TABLE public.prediction_results;

CREATE TABLE public.prediction_results (
	id serial4 NOT NULL,
	ticker varchar(10) NOT NULL,
	prediction_date date NOT NULL,
	target_date date NOT NULL,
	model_name varchar(100) NOT NULL,
	model_version varchar(50) NULL,
	prediction_type varchar(50) NULL,
	predicted_value numeric(15, 6) NULL,
	confidence_score numeric(5, 2) NULL,
	actual_value numeric(15, 6) NULL,
	prediction_error numeric(15, 6) NULL,
	features_used jsonb NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT prediction_results_confidence_check CHECK (((confidence_score >= (0)::numeric) AND (confidence_score <= (100)::numeric))),
	CONSTRAINT prediction_results_pkey PRIMARY KEY (id),
	CONSTRAINT prediction_results_ticker_prediction_target_model_unique UNIQUE (ticker, prediction_date, target_date, model_name)
);
CREATE INDEX idx_prediction_results_model ON public.prediction_results USING btree (model_name);
CREATE INDEX idx_prediction_results_prediction_date ON public.prediction_results USING btree (prediction_date);
CREATE INDEX idx_prediction_results_target_date ON public.prediction_results USING btree (target_date);
CREATE INDEX idx_prediction_results_ticker ON public.prediction_results USING btree (ticker);


-- public.system_config definition

-- Drop table

-- DROP TABLE public.system_config;

CREATE TABLE public.system_config (
	id serial4 NOT NULL,
	config_key varchar(100) NOT NULL,
	config_value text NOT NULL,
	config_type varchar(20) DEFAULT 'string'::character varying NULL,
	description text NULL,
	is_active bool DEFAULT true NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT system_config_config_key_key UNIQUE (config_key),
	CONSTRAINT system_config_pkey PRIMARY KEY (id)
);

-- Table Triggers

create trigger update_system_config_updated_at before
update
    on
    public.system_config for each row execute function update_updated_at_column();


-- public.technical_features_momentum definition

-- Drop table

-- DROP TABLE public.technical_features_momentum;

CREATE TABLE public.technical_features_momentum (
	id serial4 NOT NULL,
	ticker varchar(10) NOT NULL,
	"date" date NOT NULL,
	rsi_14 numeric(5, 2) NULL,
	rsi_14_overbought bool NULL,
	rsi_14_oversold bool NULL,
	rsi_21 numeric(5, 2) NULL,
	rsi_21_overbought bool NULL,
	rsi_21_oversold bool NULL,
	stoch_k numeric(5, 2) NULL,
	stoch_d numeric(5, 2) NULL,
	stoch_overbought bool NULL,
	stoch_oversold bool NULL,
	stoch_k_above_d bool NULL,
	stoch_crossover numeric(5, 2) NULL,
	roc_5 numeric(10, 4) NULL,
	roc_10 numeric(10, 4) NULL,
	roc_20 numeric(10, 4) NULL,
	roc_5_positive bool NULL,
	roc_10_positive bool NULL,
	roc_20_positive bool NULL,
	williams_r_14 numeric(5, 2) NULL,
	williams_r_14_overbought bool NULL,
	williams_r_14_oversold bool NULL,
	williams_r_14_neutral bool NULL,
	williams_r_21 numeric(5, 2) NULL,
	williams_r_21_overbought bool NULL,
	williams_r_21_oversold bool NULL,
	williams_r_21_neutral bool NULL,
	momentum_5 numeric(10, 4) NULL,
	momentum_10 numeric(10, 4) NULL,
	momentum_20 numeric(10, 4) NULL,
	momentum_trend numeric(10, 4) NULL,
	momentum_acceleration numeric(10, 4) NULL,
	momentum_divergence numeric(10, 4) NULL,
	quality_score numeric(5, 2) NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT momentum_features_unique UNIQUE (ticker, date),
	CONSTRAINT technical_features_momentum_pkey PRIMARY KEY (id)
);
CREATE INDEX idx_momentum_date ON public.technical_features_momentum USING btree (date);
CREATE INDEX idx_momentum_ticker ON public.technical_features_momentum USING btree (ticker);
CREATE INDEX idx_momentum_ticker_date ON public.technical_features_momentum USING btree (ticker, date);


-- public.technical_features_trend definition

-- Drop table

-- DROP TABLE public.technical_features_trend;

CREATE TABLE public.technical_features_trend (
	id serial4 NOT NULL,
	ticker varchar(10) NOT NULL,
	"date" date NOT NULL,
	sma_5 numeric(10, 4) NULL,
	sma_10 numeric(10, 4) NULL,
	sma_20 numeric(10, 4) NULL,
	sma_50 numeric(10, 4) NULL,
	sma_100 numeric(10, 4) NULL,
	sma_200 numeric(10, 4) NULL,
	ema_5 numeric(10, 4) NULL,
	ema_10 numeric(10, 4) NULL,
	ema_20 numeric(10, 4) NULL,
	ema_50 numeric(10, 4) NULL,
	ema_100 numeric(10, 4) NULL,
	ema_200 numeric(10, 4) NULL,
	macd numeric(10, 4) NULL,
	macd_signal numeric(10, 4) NULL,
	macd_histogram numeric(10, 4) NULL,
	macd_above_signal bool NULL,
	macd_crossover numeric(10, 4) NULL,
	ichimoku_tenkan numeric(10, 4) NULL,
	ichimoku_kijun numeric(10, 4) NULL,
	ichimoku_senkou_a numeric(10, 4) NULL,
	ichimoku_senkou_b numeric(10, 4) NULL,
	ichimoku_chikou numeric(10, 4) NULL,
	ichimoku_tenkan_above_kijun bool NULL,
	ichimoku_price_above_cloud bool NULL,
	ichimoku_price_below_cloud bool NULL,
	ichimoku_cloud_green bool NULL,
	ichimoku_cloud_thickness numeric(10, 4) NULL,
	quality_score numeric(5, 2) NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT technical_features_trend_pkey PRIMARY KEY (id),
	CONSTRAINT trend_features_unique UNIQUE (ticker, date)
);
CREATE INDEX idx_trend_date ON public.technical_features_trend USING btree (date);
CREATE INDEX idx_trend_ticker ON public.technical_features_trend USING btree (ticker);
CREATE INDEX idx_trend_ticker_date ON public.technical_features_trend USING btree (ticker, date);


-- public.technical_features_volatility definition

-- Drop table

-- DROP TABLE public.technical_features_volatility;

CREATE TABLE public.technical_features_volatility (
	id serial4 NOT NULL,
	ticker varchar(10) NOT NULL,
	"date" date NOT NULL,
	bb_lower numeric(10, 4) NULL,
	bb_middle numeric(10, 4) NULL,
	bb_upper numeric(10, 4) NULL,
	bb_bandwidth numeric(10, 4) NULL,
	bb_percent numeric(10, 4) NULL,
	bb_width numeric(10, 4) NULL,
	bb_above_upper bool NULL,
	bb_below_lower bool NULL,
	bb_squeeze bool NULL,
	bb_position numeric(5, 4) NULL,
	atr_14 numeric(10, 4) NULL,
	atr_21 numeric(10, 4) NULL,
	atr_normalized_14 numeric(10, 4) NULL,
	atr_normalized_21 numeric(10, 4) NULL,
	volatility_std_10 numeric(10, 4) NULL,
	volatility_std_20 numeric(10, 4) NULL,
	volatility_annualized_10 numeric(10, 4) NULL,
	volatility_annualized_20 numeric(10, 4) NULL,
	hl_volatility_10 numeric(10, 4) NULL,
	hl_volatility_20 numeric(10, 4) NULL,
	parkinson_vol_10 numeric(10, 4) NULL,
	parkinson_vol_20 numeric(10, 4) NULL,
	vol_regime_high bool NULL,
	vol_regime_low bool NULL,
	vol_trend_rising bool NULL,
	quality_score numeric(5, 2) NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT technical_features_volatility_pkey PRIMARY KEY (id),
	CONSTRAINT volatility_features_unique UNIQUE (ticker, date)
);
CREATE INDEX idx_volatility_date ON public.technical_features_volatility USING btree (date);
CREATE INDEX idx_volatility_ticker ON public.technical_features_volatility USING btree (ticker);
CREATE INDEX idx_volatility_ticker_date ON public.technical_features_volatility USING btree (ticker, date);


-- public.technical_features_volume definition

-- Drop table

-- DROP TABLE public.technical_features_volume;

CREATE TABLE public.technical_features_volume (
	id serial4 NOT NULL,
	ticker varchar(10) NOT NULL,
	"date" date NOT NULL,
	obv_millions numeric(10, 4) NULL,
	obv_sma_10_millions numeric(10, 4) NULL,
	obv_sma_20_millions numeric(10, 4) NULL,
	obv_momentum_5_millions numeric(10, 4) NULL,
	obv_momentum_10_millions numeric(10, 4) NULL,
	vpt_millions numeric(10, 4) NULL,
	vpt_sma_10_millions numeric(10, 4) NULL,
	vpt_sma_20_millions numeric(10, 4) NULL,
	vpt_roc_5 numeric(10, 4) NULL,
	vpt_roc_10 numeric(10, 4) NULL,
	ad_line_millions numeric(10, 4) NULL,
	ad_sma_10_millions numeric(10, 4) NULL,
	ad_sma_20_millions numeric(10, 4) NULL,
	ad_momentum_5_millions numeric(10, 4) NULL,
	ad_momentum_10_millions numeric(10, 4) NULL,
	ad_oscillator_millions numeric(10, 4) NULL,
	volume_ma_5_thousands numeric(10, 4) NULL,
	volume_ma_10_thousands numeric(10, 4) NULL,
	volume_ma_20_thousands numeric(10, 4) NULL,
	volume_ratio_5 numeric(10, 4) NULL,
	volume_ratio_10 numeric(10, 4) NULL,
	volume_ratio_20 numeric(10, 4) NULL,
	mfi_14 numeric(5, 2) NULL,
	mfi_overbought bool NULL,
	mfi_oversold bool NULL,
	volume_spike bool NULL,
	volume_above_average bool NULL,
	volume_trend numeric(10, 4) NULL,
	volume_weighted_price numeric(10, 4) NULL,
	volume_oscillator numeric(10, 4) NULL,
	volume_rate_of_change numeric(10, 4) NULL,
	quality_score numeric(5, 2) NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT technical_features_volume_pkey PRIMARY KEY (id),
	CONSTRAINT volume_features_unique UNIQUE (ticker, date)
);
CREATE INDEX idx_volume_date ON public.technical_features_volume USING btree (date);
CREATE INDEX idx_volume_ticker ON public.technical_features_volume USING btree (ticker);
CREATE INDEX idx_volume_ticker_date ON public.technical_features_volume USING btree (ticker, date);


-- public.ticker_cache definition

-- Drop table

-- DROP TABLE public.ticker_cache;

CREATE TABLE public.ticker_cache (
	id serial4 NOT NULL,
	cache_key varchar(255) NOT NULL,
	cache_data jsonb NOT NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	expires_at timestamp NOT NULL,
	CONSTRAINT ticker_cache_cache_key_key UNIQUE (cache_key),
	CONSTRAINT ticker_cache_pkey PRIMARY KEY (id)
);
CREATE INDEX idx_ticker_cache_expires ON public.ticker_cache USING btree (expires_at);
CREATE INDEX idx_ticker_cache_key ON public.ticker_cache USING btree (cache_key);


-- public.tickers definition

-- Drop table

-- DROP TABLE public.tickers;

CREATE TABLE public.tickers (
	id serial4 NOT NULL,
	ticker varchar(10) NOT NULL,
	"name" varchar(255) NULL,
	market varchar(50) DEFAULT 'stocks'::character varying NULL,
	locale varchar(10) DEFAULT 'us'::character varying NULL,
	primary_exchange varchar(50) NULL,
	currency_name varchar(10) NULL,
	active bool DEFAULT true NULL,
	"type" varchar(50) NULL,
	market_cap float8 NULL,
	weighted_shares_outstanding float8 NULL,
	round_lot int4 NULL,
	is_sp500 bool DEFAULT false NULL,
	is_popular bool DEFAULT false NULL,
	last_updated timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	cik varchar(20) NULL,
	composite_figi varchar(20) NULL,
	share_class_figi varchar(20) NULL,
	sic_code varchar(10) NULL,
	sic_description varchar(255) NULL,
	ticker_root varchar(10) NULL,
	total_employees int4 NULL,
	list_date date NULL,
	last_updated_utc timestamp NULL,
	CONSTRAINT tickers_pkey PRIMARY KEY (id),
	CONSTRAINT tickers_ticker_key UNIQUE (ticker)
);
CREATE INDEX idx_tickers_active ON public.tickers USING btree (active);
CREATE INDEX idx_tickers_cik ON public.tickers USING btree (cik);
CREATE INDEX idx_tickers_composite_figi ON public.tickers USING btree (composite_figi);
CREATE INDEX idx_tickers_last_updated_utc ON public.tickers USING btree (last_updated_utc);
CREATE INDEX idx_tickers_list_date ON public.tickers USING btree (list_date);
CREATE INDEX idx_tickers_market ON public.tickers USING btree (market);
CREATE INDEX idx_tickers_popular ON public.tickers USING btree (is_popular);
CREATE INDEX idx_tickers_share_class_figi ON public.tickers USING btree (share_class_figi);
CREATE INDEX idx_tickers_sic_code ON public.tickers USING btree (sic_code);
CREATE INDEX idx_tickers_sp500 ON public.tickers USING btree (is_sp500);
CREATE INDEX idx_tickers_sp500_active ON public.tickers USING btree (is_sp500, active) WHERE ((is_sp500 = true) AND (active = true));
CREATE INDEX idx_tickers_ticker ON public.tickers USING btree (ticker);


-- public.polygon_news_insights definition

-- Drop table

-- DROP TABLE public.polygon_news_insights;

CREATE TABLE public.polygon_news_insights (
	id serial4 NOT NULL,
	article_id int4 NULL,
	sentiment varchar(20) NULL,
	sentiment_reasoning text NULL,
	insight_type varchar(50) NULL,
	insight_value text NULL,
	confidence_score float8 NULL,
	created_at timestamptz DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT polygon_news_insights_pkey PRIMARY KEY (id),
	CONSTRAINT valid_confidence CHECK (((confidence_score >= (0)::double precision) AND (confidence_score <= (1)::double precision))),
	CONSTRAINT polygon_news_insights_article_id_fkey FOREIGN KEY (article_id) REFERENCES public.polygon_news_articles(id) ON DELETE CASCADE
);
CREATE INDEX idx_polygon_insights_article ON public.polygon_news_insights USING btree (article_id);
CREATE INDEX idx_polygon_insights_sentiment ON public.polygon_news_insights USING btree (sentiment);
CREATE INDEX idx_polygon_insights_type ON public.polygon_news_insights USING btree (insight_type);


-- public.polygon_news_tickers definition

-- Drop table

-- DROP TABLE public.polygon_news_tickers;

CREATE TABLE public.polygon_news_tickers (
	id serial4 NOT NULL,
	article_id int4 NULL,
	ticker varchar(10) NOT NULL,
	created_at timestamptz DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT polygon_news_tickers_article_id_ticker_key UNIQUE (article_id, ticker),
	CONSTRAINT polygon_news_tickers_pkey PRIMARY KEY (id),
	CONSTRAINT polygon_news_tickers_article_id_fkey FOREIGN KEY (article_id) REFERENCES public.polygon_news_articles(id) ON DELETE CASCADE
);
CREATE INDEX idx_polygon_tickers_article ON public.polygon_news_tickers USING btree (article_id);
CREATE INDEX idx_polygon_tickers_ticker ON public.polygon_news_tickers USING btree (ticker);


-- public.company_market_rankings source

CREATE OR REPLACE VIEW public.company_market_rankings
AS SELECT fr.ticker,
    fr.date,
    percent_rank() OVER (ORDER BY fr.pe_ratio) * 100::double precision AS pe_market_percentile,
    percent_rank() OVER (ORDER BY fr.pb_ratio) * 100::double precision AS pb_market_percentile,
    percent_rank() OVER (ORDER BY fr.roe) * 100::double precision AS roe_market_percentile,
    percent_rank() OVER (ORDER BY t.market_cap) * 100::double precision AS market_cap_market_percentile,
    count(*) OVER () AS market_peer_count
   FROM tickers t
     JOIN ( SELECT DISTINCT ON (fundamental_ratios.ticker) fundamental_ratios.ticker,
            fundamental_ratios.date,
            fundamental_ratios.pe_ratio,
            fundamental_ratios.pb_ratio,
            fundamental_ratios.roe
           FROM fundamental_ratios
          WHERE fundamental_ratios.date >= (CURRENT_DATE - '1 year'::interval)
          ORDER BY fundamental_ratios.ticker, fundamental_ratios.date DESC) fr ON t.ticker::text = fr.ticker::text
  WHERE t.is_sp500 = true AND t.active = true
  ORDER BY fr.ticker;


-- public.company_sector_rankings source

CREATE OR REPLACE VIEW public.company_sector_rankings
AS SELECT fr.ticker,
    t.sic_code,
    t.sic_description,
    fr.date,
    percent_rank() OVER (PARTITION BY t.sic_code ORDER BY fr.pe_ratio) * 100::double precision AS pe_sector_percentile,
    percent_rank() OVER (PARTITION BY t.sic_code ORDER BY fr.pb_ratio) * 100::double precision AS pb_sector_percentile,
    percent_rank() OVER (PARTITION BY t.sic_code ORDER BY fr.roe) * 100::double precision AS roe_sector_percentile,
    percent_rank() OVER (PARTITION BY t.sic_code ORDER BY fr.roa) * 100::double precision AS roa_sector_percentile,
    percent_rank() OVER (PARTITION BY t.sic_code ORDER BY t.market_cap) * 100::double precision AS market_cap_sector_percentile,
    count(*) OVER (PARTITION BY t.sic_code) AS sector_peer_count
   FROM tickers t
     JOIN ( SELECT DISTINCT ON (fundamental_ratios.ticker) fundamental_ratios.ticker,
            fundamental_ratios.date,
            fundamental_ratios.pe_ratio,
            fundamental_ratios.pb_ratio,
            fundamental_ratios.roe,
            fundamental_ratios.roa
           FROM fundamental_ratios
          WHERE fundamental_ratios.date >= (CURRENT_DATE - '1 year'::interval)
          ORDER BY fundamental_ratios.ticker, fundamental_ratios.date DESC) fr ON t.ticker::text = fr.ticker::text
  WHERE t.sic_code IS NOT NULL AND t.active = true
  ORDER BY t.sic_code, fr.ticker;


-- public.market_peer_metrics source

CREATE OR REPLACE VIEW public.market_peer_metrics
AS SELECT 'SP500'::text AS market_segment,
    count(DISTINCT fr.ticker) AS peer_count,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (fr.pe_ratio::double precision)) AS median_pe,
    percentile_cont(0.1::double precision) WITHIN GROUP (ORDER BY (fr.pe_ratio::double precision)) AS p10_pe,
    percentile_cont(0.9::double precision) WITHIN GROUP (ORDER BY (fr.pe_ratio::double precision)) AS p90_pe,
    avg(fr.pe_ratio) AS avg_pe,
    stddev(fr.pe_ratio) AS stddev_pe,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (fr.pb_ratio::double precision)) AS median_pb,
    percentile_cont(0.1::double precision) WITHIN GROUP (ORDER BY (fr.pb_ratio::double precision)) AS p10_pb,
    percentile_cont(0.9::double precision) WITHIN GROUP (ORDER BY (fr.pb_ratio::double precision)) AS p90_pb,
    avg(fr.pb_ratio) AS avg_pb,
    stddev(fr.pb_ratio) AS stddev_pb,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (fr.roe::double precision)) AS median_roe,
    percentile_cont(0.1::double precision) WITHIN GROUP (ORDER BY (fr.roe::double precision)) AS p10_roe,
    percentile_cont(0.9::double precision) WITHIN GROUP (ORDER BY (fr.roe::double precision)) AS p90_roe,
    avg(fr.roe) AS avg_roe,
    stddev(fr.roe) AS stddev_roe,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY t.market_cap) AS median_market_cap,
    percentile_cont(0.1::double precision) WITHIN GROUP (ORDER BY t.market_cap) AS p10_market_cap,
    percentile_cont(0.9::double precision) WITHIN GROUP (ORDER BY t.market_cap) AS p90_market_cap,
    avg(t.market_cap) AS avg_market_cap,
    stddev(t.market_cap) AS stddev_market_cap,
    CURRENT_TIMESTAMP AS calculated_at
   FROM tickers t
     JOIN ( SELECT DISTINCT ON (fundamental_ratios.ticker) fundamental_ratios.ticker,
            fundamental_ratios.date,
            fundamental_ratios.pe_ratio,
            fundamental_ratios.pb_ratio,
            fundamental_ratios.roe
           FROM fundamental_ratios
          WHERE fundamental_ratios.date >= (CURRENT_DATE - '1 year'::interval)
          ORDER BY fundamental_ratios.ticker, fundamental_ratios.date DESC) fr ON t.ticker::text = fr.ticker::text
  WHERE t.is_sp500 = true AND t.active = true AND fr.pe_ratio IS NOT NULL;


-- public.sector_peer_metrics source

CREATE OR REPLACE VIEW public.sector_peer_metrics
AS SELECT t.sic_code,
    t.sic_description,
    count(DISTINCT fr.ticker) AS peer_count,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (fr.pe_ratio::double precision)) AS median_pe,
    percentile_cont(0.25::double precision) WITHIN GROUP (ORDER BY (fr.pe_ratio::double precision)) AS q1_pe,
    percentile_cont(0.75::double precision) WITHIN GROUP (ORDER BY (fr.pe_ratio::double precision)) AS q3_pe,
    avg(fr.pe_ratio) AS avg_pe,
    stddev(fr.pe_ratio) AS stddev_pe,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (fr.pb_ratio::double precision)) AS median_pb,
    percentile_cont(0.25::double precision) WITHIN GROUP (ORDER BY (fr.pb_ratio::double precision)) AS q1_pb,
    percentile_cont(0.75::double precision) WITHIN GROUP (ORDER BY (fr.pb_ratio::double precision)) AS q3_pb,
    avg(fr.pb_ratio) AS avg_pb,
    stddev(fr.pb_ratio) AS stddev_pb,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (fr.ps_ratio::double precision)) AS median_ps,
    percentile_cont(0.25::double precision) WITHIN GROUP (ORDER BY (fr.ps_ratio::double precision)) AS q1_ps,
    percentile_cont(0.75::double precision) WITHIN GROUP (ORDER BY (fr.ps_ratio::double precision)) AS q3_ps,
    avg(fr.ps_ratio) AS avg_ps,
    stddev(fr.ps_ratio) AS stddev_ps,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (fr.roe::double precision)) AS median_roe,
    percentile_cont(0.25::double precision) WITHIN GROUP (ORDER BY (fr.roe::double precision)) AS q1_roe,
    percentile_cont(0.75::double precision) WITHIN GROUP (ORDER BY (fr.roe::double precision)) AS q3_roe,
    avg(fr.roe) AS avg_roe,
    stddev(fr.roe) AS stddev_roe,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (fr.roa::double precision)) AS median_roa,
    percentile_cont(0.25::double precision) WITHIN GROUP (ORDER BY (fr.roa::double precision)) AS q1_roa,
    percentile_cont(0.75::double precision) WITHIN GROUP (ORDER BY (fr.roa::double precision)) AS q3_roa,
    avg(fr.roa) AS avg_roa,
    stddev(fr.roa) AS stddev_roa,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (fr.debt_to_equity::double precision)) AS median_debt_to_equity,
    percentile_cont(0.25::double precision) WITHIN GROUP (ORDER BY (fr.debt_to_equity::double precision)) AS q1_debt_to_equity,
    percentile_cont(0.75::double precision) WITHIN GROUP (ORDER BY (fr.debt_to_equity::double precision)) AS q3_debt_to_equity,
    avg(fr.debt_to_equity) AS avg_debt_to_equity,
    stddev(fr.debt_to_equity) AS stddev_debt_to_equity,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY t.market_cap) AS median_market_cap,
    percentile_cont(0.25::double precision) WITHIN GROUP (ORDER BY t.market_cap) AS q1_market_cap,
    percentile_cont(0.75::double precision) WITHIN GROUP (ORDER BY t.market_cap) AS q3_market_cap,
    avg(t.market_cap) AS avg_market_cap,
    stddev(t.market_cap) AS stddev_market_cap,
    avg(fr.data_quality_score) AS avg_data_quality,
    min(fr.date) AS earliest_data_date,
    max(fr.date) AS latest_data_date,
    CURRENT_TIMESTAMP AS calculated_at
   FROM tickers t
     JOIN ( SELECT DISTINCT ON (fundamental_ratios.ticker) fundamental_ratios.ticker,
            fundamental_ratios.date,
            fundamental_ratios.pe_ratio,
            fundamental_ratios.pb_ratio,
            fundamental_ratios.ps_ratio,
            fundamental_ratios.roe,
            fundamental_ratios.roa,
            fundamental_ratios.debt_to_equity,
            fundamental_ratios.data_quality_score
           FROM fundamental_ratios
          WHERE fundamental_ratios.date >= (CURRENT_DATE - '1 year'::interval)
          ORDER BY fundamental_ratios.ticker, fundamental_ratios.date DESC) fr ON t.ticker::text = fr.ticker::text
  WHERE t.sic_code IS NOT NULL AND t.active = true AND fr.pe_ratio IS NOT NULL
  GROUP BY t.sic_code, t.sic_description
 HAVING count(DISTINCT fr.ticker) >= 3
  ORDER BY (count(DISTINCT fr.ticker)) DESC;


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
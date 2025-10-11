--
-- PostgreSQL database cluster dump
--

-- Started on 2025-09-23 20:08:03

SET default_transaction_read_only = off;

SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;

--
-- Roles
--

CREATE ROLE postgres;
ALTER ROLE postgres WITH SUPERUSER INHERIT CREATEROLE CREATEDB LOGIN REPLICATION BYPASSRLS;

--
-- User Configurations
--

--
-- Databases
--

--
-- Database "template1" dump
--

\connect template1

--
-- PostgreSQL database dump
--

-- Dumped from database version 17.4
-- Dumped by pg_dump version 17.4

-- Started on 2025-09-23 20:08:04

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

-- Completed on 2025-09-23 20:08:04

--
-- PostgreSQL database dump complete
--

--
-- Database "stock_data" dump
--

--
-- PostgreSQL database dump
--

-- Dumped from database version 17.4
-- Dumped by pg_dump version 17.4

-- Started on 2025-09-23 20:08:04

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 5139 (class 1262 OID 131755)
-- Name: stock_data; Type: DATABASE; Schema: -; Owner: postgres
--

CREATE DATABASE stock_data WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'Hungarian_Hungary.1250';


ALTER DATABASE stock_data OWNER TO postgres;

\connect stock_data

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 259 (class 1255 OID 132787)
-- Name: migrate_technical_features(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.migrate_technical_features() RETURNS text
    LANGUAGE plpgsql
    AS $$
DECLARE
    result_text TEXT := '';
BEGIN
    -- This function would contain the migration logic
    -- from the old technical_features table to the new structure
    
    result_text := 'Migration function created. Implement migration logic as needed.';
    RETURN result_text;
END;
$$;


ALTER FUNCTION public.migrate_technical_features() OWNER TO postgres;

--
-- TOC entry 258 (class 1255 OID 132417)
-- Name: update_updated_at_column(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.update_updated_at_column() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_updated_at_column() OWNER TO postgres;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 246 (class 1259 OID 439946)
-- Name: dividends; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.dividends (
    id character varying(128) NOT NULL,
    ticker_id integer NOT NULL,
    cash_amount numeric(18,6) NOT NULL,
    currency character(3),
    declaration_date date,
    ex_dividend_date date,
    pay_date date,
    record_date date,
    frequency smallint,
    dividend_type character varying(16),
    raw_payload jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.dividends OWNER TO postgres;

--
-- TOC entry 228 (class 1259 OID 132640)
-- Name: feature_calculation_jobs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.feature_calculation_jobs (
    id integer NOT NULL,
    job_id character varying(50) NOT NULL,
    ticker character varying(10),
    start_date date,
    end_date date,
    feature_categories text[],
    status character varying(20) DEFAULT 'pending'::character varying NOT NULL,
    total_features_calculated integer DEFAULT 0,
    total_warnings integer DEFAULT 0,
    error_message text,
    started_at timestamp without time zone,
    completed_at timestamp without time zone,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT feature_calculation_jobs_status_check CHECK (((status)::text = ANY ((ARRAY['pending'::character varying, 'running'::character varying, 'completed'::character varying, 'failed'::character varying, 'cancelled'::character varying])::text[])))
);


ALTER TABLE public.feature_calculation_jobs OWNER TO postgres;

--
-- TOC entry 227 (class 1259 OID 132639)
-- Name: feature_calculation_jobs_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.feature_calculation_jobs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.feature_calculation_jobs_id_seq OWNER TO postgres;

--
-- TOC entry 5140 (class 0 OID 0)
-- Dependencies: 227
-- Name: feature_calculation_jobs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.feature_calculation_jobs_id_seq OWNED BY public.feature_calculation_jobs.id;


--
-- TOC entry 245 (class 1259 OID 320811)
-- Name: fundamental_facts_v2; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.fundamental_facts_v2 (
    id bigint NOT NULL,
    ticker_id integer NOT NULL,
    date date NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    filing_date date,
    fiscal_period character varying(10),
    fiscal_year character varying(10),
    timeframe character varying(20),
    source_filing_url text,
    source_filing_file_url text,
    acceptance_datetime timestamp without time zone,
    revenues numeric(15,2),
    cost_of_revenue numeric(15,2),
    gross_profit numeric(15,2),
    operating_expenses numeric(15,2),
    selling_general_and_administrative_expenses numeric(15,2),
    research_and_development numeric(15,2),
    operating_income_loss numeric(15,2),
    nonoperating_income_loss numeric(15,2),
    income_loss_from_continuing_operations_before_tax numeric(15,2),
    income_tax_expense_benefit numeric(15,2),
    income_loss_from_continuing_operations_after_tax numeric(15,2),
    net_income_loss numeric(15,2),
    net_income_loss_attributable_to_parent numeric(15,2),
    basic_earnings_per_share numeric(10,4),
    diluted_earnings_per_share numeric(10,4),
    basic_average_shares numeric(15,2),
    diluted_average_shares numeric(15,2),
    assets numeric(15,2),
    current_assets numeric(15,2),
    noncurrent_assets numeric(15,2),
    inventory numeric(15,2),
    other_current_assets numeric(15,2),
    fixed_assets numeric(15,2),
    other_noncurrent_assets numeric(15,2),
    liabilities numeric(15,2),
    current_liabilities numeric(15,2),
    noncurrent_liabilities numeric(15,2),
    accounts_payable numeric(15,2),
    other_current_liabilities numeric(15,2),
    long_term_debt numeric(15,2),
    other_noncurrent_liabilities numeric(15,2),
    equity numeric(15,2),
    equity_attributable_to_parent numeric(15,2),
    net_cash_flow_from_operating_activities numeric(15,2),
    net_cash_flow_from_investing_activities numeric(15,2),
    net_cash_flow_from_financing_activities numeric(15,2),
    net_cash_flow numeric(15,2),
    net_cash_flow_continuing numeric(15,2),
    net_cash_flow_from_operating_activities_continuing numeric(15,2),
    net_cash_flow_from_investing_activities_continuing numeric(15,2),
    net_cash_flow_from_financing_activities_continuing numeric(15,2),
    comprehensive_income_loss numeric(15,2),
    comprehensive_income_loss_attributable_to_parent numeric(15,2),
    other_comprehensive_income_loss numeric(15,2),
    other_comprehensive_income_loss_attributable_to_parent numeric(15,2),
    data_quality_score numeric(5,4),
    missing_data_count integer DEFAULT 0,
    direct_report_fields_count integer DEFAULT 0,
    imputed_fields_count integer DEFAULT 0,
    derived_fields_count integer DEFAULT 0,
    total_fields_count integer DEFAULT 0,
    data_completeness_percentage numeric(5,2),
    completeness_score numeric(5,4) GENERATED ALWAYS AS (
CASE
    WHEN (total_fields_count = 0) THEN 0.0
    ELSE (((total_fields_count - missing_data_count))::numeric / (total_fields_count)::numeric)
END) STORED,
    data_source_confidence numeric(3,2)
);


ALTER TABLE public.fundamental_facts_v2 OWNER TO postgres;

--
-- TOC entry 244 (class 1259 OID 320810)
-- Name: fundamental_facts_v2_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.fundamental_facts_v2_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.fundamental_facts_v2_id_seq OWNER TO postgres;

--
-- TOC entry 5141 (class 0 OID 0)
-- Dependencies: 244
-- Name: fundamental_facts_v2_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.fundamental_facts_v2_id_seq OWNED BY public.fundamental_facts_v2.id;


--
-- TOC entry 218 (class 1259 OID 131769)
-- Name: historical_prices; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.historical_prices (
    id integer NOT NULL,
    ticker character varying(10) NOT NULL,
    date date NOT NULL,
    open numeric NOT NULL,
    high numeric NOT NULL,
    low numeric NOT NULL,
    close numeric NOT NULL,
    volume bigint NOT NULL,
    adjusted_close numeric,
    vwap numeric,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.historical_prices OWNER TO postgres;

--
-- TOC entry 217 (class 1259 OID 131768)
-- Name: historical_prices_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.historical_prices_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.historical_prices_id_seq OWNER TO postgres;

--
-- TOC entry 5142 (class 0 OID 0)
-- Dependencies: 217
-- Name: historical_prices_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.historical_prices_id_seq OWNED BY public.historical_prices.id;


--
-- TOC entry 222 (class 1259 OID 131863)
-- Name: polygon_news_articles; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.polygon_news_articles (
    id integer NOT NULL,
    polygon_id character varying(100) NOT NULL,
    title character varying(1000) NOT NULL,
    description text,
    article_url character varying(2000) NOT NULL,
    amp_url character varying(2000),
    image_url character varying(2000),
    author character varying(200),
    published_utc timestamp with time zone NOT NULL,
    publisher_name character varying(200),
    publisher_homepage_url character varying(500),
    publisher_logo_url character varying(500),
    publisher_favicon_url character varying(500),
    keywords text[],
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    is_processed boolean DEFAULT false,
    processing_errors text,
    quality_score double precision,
    relevance_score double precision,
    CONSTRAINT valid_quality_score CHECK (((quality_score >= (0)::double precision) AND (quality_score <= (1)::double precision))),
    CONSTRAINT valid_relevance_score CHECK (((relevance_score >= (0)::double precision) AND (relevance_score <= (1)::double precision)))
);


ALTER TABLE public.polygon_news_articles OWNER TO postgres;

--
-- TOC entry 221 (class 1259 OID 131862)
-- Name: polygon_news_articles_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.polygon_news_articles_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.polygon_news_articles_id_seq OWNER TO postgres;

--
-- TOC entry 5143 (class 0 OID 0)
-- Dependencies: 221
-- Name: polygon_news_articles_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.polygon_news_articles_id_seq OWNED BY public.polygon_news_articles.id;


--
-- TOC entry 226 (class 1259 OID 131894)
-- Name: polygon_news_insights; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.polygon_news_insights (
    id integer NOT NULL,
    article_id integer,
    sentiment character varying(20),
    sentiment_reasoning text,
    insight_type character varying(50),
    insight_value text,
    confidence_score double precision,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_confidence CHECK (((confidence_score >= (0)::double precision) AND (confidence_score <= (1)::double precision)))
);


ALTER TABLE public.polygon_news_insights OWNER TO postgres;

--
-- TOC entry 225 (class 1259 OID 131893)
-- Name: polygon_news_insights_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.polygon_news_insights_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.polygon_news_insights_id_seq OWNER TO postgres;

--
-- TOC entry 5144 (class 0 OID 0)
-- Dependencies: 225
-- Name: polygon_news_insights_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.polygon_news_insights_id_seq OWNED BY public.polygon_news_insights.id;


--
-- TOC entry 224 (class 1259 OID 131879)
-- Name: polygon_news_tickers; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.polygon_news_tickers (
    id integer NOT NULL,
    article_id integer,
    ticker character varying(10) NOT NULL,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.polygon_news_tickers OWNER TO postgres;

--
-- TOC entry 223 (class 1259 OID 131878)
-- Name: polygon_news_tickers_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.polygon_news_tickers_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.polygon_news_tickers_id_seq OWNER TO postgres;

--
-- TOC entry 5145 (class 0 OID 0)
-- Dependencies: 223
-- Name: polygon_news_tickers_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.polygon_news_tickers_id_seq OWNED BY public.polygon_news_tickers.id;


--
-- TOC entry 241 (class 1259 OID 287579)
-- Name: raw_fundamental_data; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.raw_fundamental_data (
    id integer NOT NULL,
    ticker_id integer NOT NULL,
    date date NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    filing_date date,
    fiscal_period character varying(10),
    fiscal_year character varying(10),
    timeframe character varying(20),
    cik character varying(20),
    company_name character varying(255),
    source_filing_url text,
    source_filing_file_url text,
    revenues numeric(15,2),
    cost_of_revenue numeric(15,2),
    gross_profit numeric(15,2),
    operating_expenses numeric(15,2),
    selling_general_and_administrative_expenses numeric(15,2),
    research_and_development numeric(15,2),
    operating_income_loss numeric(15,2),
    nonoperating_income_loss numeric(15,2),
    income_loss_from_continuing_operations_before_tax numeric(15,2),
    income_tax_expense_benefit numeric(15,2),
    income_loss_from_continuing_operations_after_tax numeric(15,2),
    net_income_loss numeric(15,2),
    net_income_loss_attributable_to_parent numeric(15,2),
    basic_earnings_per_share numeric(10,4),
    diluted_earnings_per_share numeric(10,4),
    basic_average_shares numeric(15,2),
    diluted_average_shares numeric(15,2),
    assets numeric(15,2),
    current_assets numeric(15,2),
    noncurrent_assets numeric(15,2),
    inventory numeric(15,2),
    other_current_assets numeric(15,2),
    fixed_assets numeric(15,2),
    other_noncurrent_assets numeric(15,2),
    liabilities numeric(15,2),
    current_liabilities numeric(15,2),
    noncurrent_liabilities numeric(15,2),
    accounts_payable numeric(15,2),
    other_current_liabilities numeric(15,2),
    long_term_debt numeric(15,2),
    other_noncurrent_liabilities numeric(15,2),
    equity numeric(15,2),
    equity_attributable_to_parent numeric(15,2),
    net_cash_flow_from_operating_activities numeric(15,2),
    net_cash_flow_from_investing_activities numeric(15,2),
    net_cash_flow_from_financing_activities numeric(15,2),
    net_cash_flow numeric(15,2),
    net_cash_flow_continuing numeric(15,2),
    net_cash_flow_from_operating_activities_continuing numeric(15,2),
    net_cash_flow_from_investing_activities_continuing numeric(15,2),
    net_cash_flow_from_financing_activities_continuing numeric(15,2),
    comprehensive_income_loss numeric(15,2),
    comprehensive_income_loss_attributable_to_parent numeric(15,2),
    other_comprehensive_income_loss numeric(15,2),
    other_comprehensive_income_loss_attributable_to_parent numeric(15,2),
    data_quality_score numeric(5,4),
    missing_data_count integer DEFAULT 0,
    benefits_costs_expenses numeric(15,2),
    preferred_stock_dividends_and_other_adjustments numeric(15,2),
    acceptance_datetime timestamp without time zone,
    sic_code character varying(10),
    sic_description character varying(255),
    direct_report_fields_count integer DEFAULT 0,
    imputed_fields_count integer DEFAULT 0,
    derived_fields_count integer DEFAULT 0,
    total_fields_count integer DEFAULT 0,
    data_completeness_percentage numeric(5,2),
    completeness_score numeric(5,4) GENERATED ALWAYS AS (
CASE
    WHEN (total_fields_count = 0) THEN 0.0
    ELSE (((total_fields_count - missing_data_count))::numeric / (total_fields_count)::numeric)
END) STORED,
    data_source_confidence numeric(3,2),
    CONSTRAINT valid_date_raw_fundamental CHECK ((date >= '2020-01-01'::date))
);


ALTER TABLE public.raw_fundamental_data OWNER TO postgres;

--
-- TOC entry 240 (class 1259 OID 287578)
-- Name: raw_fundamental_data_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.raw_fundamental_data_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.raw_fundamental_data_id_seq OWNER TO postgres;

--
-- TOC entry 5146 (class 0 OID 0)
-- Dependencies: 240
-- Name: raw_fundamental_data_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.raw_fundamental_data_id_seq OWNED BY public.raw_fundamental_data.id;


--
-- TOC entry 243 (class 1259 OID 312310)
-- Name: raw_fundamental_json; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.raw_fundamental_json (
    id bigint NOT NULL,
    ticker_id integer NOT NULL,
    period_end date,
    timeframe text,
    fiscal_period text,
    fiscal_year text,
    filing_date date,
    source text,
    payload_json jsonb NOT NULL,
    response_hash text,
    ingested_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.raw_fundamental_json OWNER TO postgres;

--
-- TOC entry 242 (class 1259 OID 312309)
-- Name: raw_fundamental_json_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.raw_fundamental_json_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.raw_fundamental_json_id_seq OWNER TO postgres;

--
-- TOC entry 5147 (class 0 OID 0)
-- Dependencies: 242
-- Name: raw_fundamental_json_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.raw_fundamental_json_id_seq OWNED BY public.raw_fundamental_json.id;


--
-- TOC entry 232 (class 1259 OID 132744)
-- Name: technical_features_momentum; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.technical_features_momentum (
    id integer NOT NULL,
    ticker character varying(10) NOT NULL,
    date date NOT NULL,
    rsi_14 numeric(5,2),
    rsi_14_overbought boolean,
    rsi_14_oversold boolean,
    rsi_21 numeric(5,2),
    rsi_21_overbought boolean,
    rsi_21_oversold boolean,
    stoch_k numeric(5,2),
    stoch_d numeric(5,2),
    stoch_overbought boolean,
    stoch_oversold boolean,
    stoch_k_above_d boolean,
    stoch_crossover numeric(5,2),
    roc_5 numeric(10,4),
    roc_10 numeric(10,4),
    roc_20 numeric(10,4),
    roc_5_positive boolean,
    roc_10_positive boolean,
    roc_20_positive boolean,
    williams_r_14 numeric(5,2),
    williams_r_14_overbought boolean,
    williams_r_14_oversold boolean,
    williams_r_14_neutral boolean,
    williams_r_21 numeric(5,2),
    williams_r_21_overbought boolean,
    williams_r_21_oversold boolean,
    williams_r_21_neutral boolean,
    momentum_5 numeric(10,4),
    momentum_10 numeric(10,4),
    momentum_20 numeric(10,4),
    momentum_trend numeric(10,4),
    momentum_acceleration numeric(10,4),
    momentum_divergence numeric(10,4),
    quality_score numeric(5,2),
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.technical_features_momentum OWNER TO postgres;

--
-- TOC entry 231 (class 1259 OID 132743)
-- Name: technical_features_momentum_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.technical_features_momentum_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.technical_features_momentum_id_seq OWNER TO postgres;

--
-- TOC entry 5148 (class 0 OID 0)
-- Dependencies: 231
-- Name: technical_features_momentum_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.technical_features_momentum_id_seq OWNED BY public.technical_features_momentum.id;


--
-- TOC entry 230 (class 1259 OID 132731)
-- Name: technical_features_trend; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.technical_features_trend (
    id integer NOT NULL,
    ticker character varying(10) NOT NULL,
    date date NOT NULL,
    sma_5 numeric(10,4),
    sma_10 numeric(10,4),
    sma_20 numeric(10,4),
    sma_50 numeric(10,4),
    sma_100 numeric(10,4),
    sma_200 numeric(10,4),
    ema_5 numeric(10,4),
    ema_10 numeric(10,4),
    ema_20 numeric(10,4),
    ema_50 numeric(10,4),
    ema_100 numeric(10,4),
    ema_200 numeric(10,4),
    macd numeric(10,4),
    macd_signal numeric(10,4),
    macd_histogram numeric(10,4),
    macd_above_signal boolean,
    macd_crossover numeric(10,4),
    ichimoku_tenkan numeric(10,4),
    ichimoku_kijun numeric(10,4),
    ichimoku_senkou_a numeric(10,4),
    ichimoku_senkou_b numeric(10,4),
    ichimoku_chikou numeric(10,4),
    ichimoku_tenkan_above_kijun boolean,
    ichimoku_price_above_cloud boolean,
    ichimoku_price_below_cloud boolean,
    ichimoku_cloud_green boolean,
    ichimoku_cloud_thickness numeric(10,4),
    quality_score numeric(5,2),
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.technical_features_trend OWNER TO postgres;

--
-- TOC entry 229 (class 1259 OID 132730)
-- Name: technical_features_trend_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.technical_features_trend_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.technical_features_trend_id_seq OWNER TO postgres;

--
-- TOC entry 5149 (class 0 OID 0)
-- Dependencies: 229
-- Name: technical_features_trend_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.technical_features_trend_id_seq OWNED BY public.technical_features_trend.id;


--
-- TOC entry 234 (class 1259 OID 132757)
-- Name: technical_features_volatility; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.technical_features_volatility (
    id integer NOT NULL,
    ticker character varying(10) NOT NULL,
    date date NOT NULL,
    bb_lower numeric(10,4),
    bb_middle numeric(10,4),
    bb_upper numeric(10,4),
    bb_bandwidth numeric(10,4),
    bb_percent numeric(10,4),
    bb_width numeric(10,4),
    bb_above_upper boolean,
    bb_below_lower boolean,
    bb_squeeze boolean,
    bb_position numeric(5,4),
    atr_14 numeric(10,4),
    atr_21 numeric(10,4),
    atr_normalized_14 numeric(10,4),
    atr_normalized_21 numeric(10,4),
    volatility_std_10 numeric(10,4),
    volatility_std_20 numeric(10,4),
    volatility_annualized_10 numeric(10,4),
    volatility_annualized_20 numeric(10,4),
    hl_volatility_10 numeric(10,4),
    hl_volatility_20 numeric(10,4),
    parkinson_vol_10 numeric(10,4),
    parkinson_vol_20 numeric(10,4),
    vol_regime_high boolean,
    vol_regime_low boolean,
    vol_trend_rising boolean,
    quality_score numeric(5,2),
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.technical_features_volatility OWNER TO postgres;

--
-- TOC entry 233 (class 1259 OID 132756)
-- Name: technical_features_volatility_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.technical_features_volatility_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.technical_features_volatility_id_seq OWNER TO postgres;

--
-- TOC entry 5150 (class 0 OID 0)
-- Dependencies: 233
-- Name: technical_features_volatility_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.technical_features_volatility_id_seq OWNED BY public.technical_features_volatility.id;


--
-- TOC entry 236 (class 1259 OID 132770)
-- Name: technical_features_volume; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.technical_features_volume (
    id integer NOT NULL,
    ticker character varying(10) NOT NULL,
    date date NOT NULL,
    obv_millions numeric(10,4),
    obv_sma_10_millions numeric(10,4),
    obv_sma_20_millions numeric(10,4),
    obv_momentum_5_millions numeric(10,4),
    obv_momentum_10_millions numeric(10,4),
    vpt_millions numeric(10,4),
    vpt_sma_10_millions numeric(10,4),
    vpt_sma_20_millions numeric(10,4),
    vpt_roc_5 numeric(10,4),
    vpt_roc_10 numeric(10,4),
    ad_line_millions numeric(10,4),
    ad_sma_10_millions numeric(10,4),
    ad_sma_20_millions numeric(10,4),
    ad_momentum_5_millions numeric(10,4),
    ad_momentum_10_millions numeric(10,4),
    ad_oscillator_millions numeric(10,4),
    volume_ma_5_thousands numeric(10,4),
    volume_ma_10_thousands numeric(10,4),
    volume_ma_20_thousands numeric(10,4),
    volume_ratio_5 numeric(10,4),
    volume_ratio_10 numeric(10,4),
    volume_ratio_20 numeric(10,4),
    mfi_14 numeric(5,2),
    mfi_overbought boolean,
    mfi_oversold boolean,
    volume_spike boolean,
    volume_above_average boolean,
    volume_trend numeric(10,4),
    volume_weighted_price numeric(10,4),
    volume_oscillator numeric(10,4),
    volume_rate_of_change numeric(10,4),
    quality_score numeric(5,2),
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.technical_features_volume OWNER TO postgres;

--
-- TOC entry 235 (class 1259 OID 132769)
-- Name: technical_features_volume_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.technical_features_volume_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.technical_features_volume_id_seq OWNER TO postgres;

--
-- TOC entry 5151 (class 0 OID 0)
-- Dependencies: 235
-- Name: technical_features_volume_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.technical_features_volume_id_seq OWNED BY public.technical_features_volume.id;


--
-- TOC entry 220 (class 1259 OID 131783)
-- Name: tickers; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.tickers (
    id integer NOT NULL,
    ticker character varying(10) NOT NULL,
    name character varying(255),
    market character varying(50) DEFAULT 'stocks'::character varying,
    locale character varying(10) DEFAULT 'us'::character varying,
    primary_exchange character varying(50),
    currency_name character varying(10),
    active boolean DEFAULT true NOT NULL,
    type character varying(50),
    market_cap double precision,
    weighted_shares_outstanding double precision,
    round_lot integer,
    last_updated timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    cik character varying(20),
    composite_figi character varying(20),
    share_class_figi character varying(20),
    sic_code character varying(10),
    sic_description character varying(255),
    ticker_root character varying(10),
    total_employees integer,
    list_date date,
    has_financials boolean DEFAULT true NOT NULL
);


ALTER TABLE public.tickers OWNER TO postgres;

--
-- TOC entry 5152 (class 0 OID 0)
-- Dependencies: 220
-- Name: TABLE tickers; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.tickers IS 'Comprehensive ticker information from Polygon.io API including company details, identifiers, and metadata';


--
-- TOC entry 5153 (class 0 OID 0)
-- Dependencies: 220
-- Name: COLUMN tickers.cik; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.tickers.cik IS 'Central Index Key (CIK) from SEC';


--
-- TOC entry 5154 (class 0 OID 0)
-- Dependencies: 220
-- Name: COLUMN tickers.composite_figi; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.tickers.composite_figi IS 'Financial Instrument Global Identifier';


--
-- TOC entry 5155 (class 0 OID 0)
-- Dependencies: 220
-- Name: COLUMN tickers.share_class_figi; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.tickers.share_class_figi IS 'Share class specific FIGI';


--
-- TOC entry 5156 (class 0 OID 0)
-- Dependencies: 220
-- Name: COLUMN tickers.sic_code; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.tickers.sic_code IS 'Standard Industrial Classification code';


--
-- TOC entry 5157 (class 0 OID 0)
-- Dependencies: 220
-- Name: COLUMN tickers.sic_description; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.tickers.sic_description IS 'SIC code description';


--
-- TOC entry 5158 (class 0 OID 0)
-- Dependencies: 220
-- Name: COLUMN tickers.ticker_root; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.tickers.ticker_root IS 'Root ticker symbol';


--
-- TOC entry 5159 (class 0 OID 0)
-- Dependencies: 220
-- Name: COLUMN tickers.total_employees; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.tickers.total_employees IS 'Total number of employees';


--
-- TOC entry 5160 (class 0 OID 0)
-- Dependencies: 220
-- Name: COLUMN tickers.list_date; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.tickers.list_date IS 'Date when the security was first listed';


--
-- TOC entry 5161 (class 0 OID 0)
-- Dependencies: 220
-- Name: COLUMN tickers.has_financials; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.tickers.has_financials IS 'Has financial data';


--
-- TOC entry 219 (class 1259 OID 131782)
-- Name: tickers_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.tickers_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.tickers_id_seq OWNER TO postgres;

--
-- TOC entry 5162 (class 0 OID 0)
-- Dependencies: 219
-- Name: tickers_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.tickers_id_seq OWNED BY public.tickers.id;


--
-- TOC entry 239 (class 1259 OID 132792)
-- Name: v_feature_summary; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.v_feature_summary AS
 SELECT COALESCE(t.ticker, m.ticker, vol.ticker, v.ticker) AS ticker,
    COALESCE(t.date, m.date, vol.date, v.date) AS date,
        CASE
            WHEN (t.ticker IS NOT NULL) THEN 27
            ELSE 0
        END AS trend_features,
        CASE
            WHEN (m.ticker IS NOT NULL) THEN 45
            ELSE 0
        END AS momentum_features,
        CASE
            WHEN (vol.ticker IS NOT NULL) THEN 55
            ELSE 0
        END AS volatility_features,
        CASE
            WHEN (v.ticker IS NOT NULL) THEN 50
            ELSE 0
        END AS volume_features,
    (((
        CASE
            WHEN (t.ticker IS NOT NULL) THEN 27
            ELSE 0
        END +
        CASE
            WHEN (m.ticker IS NOT NULL) THEN 45
            ELSE 0
        END) +
        CASE
            WHEN (vol.ticker IS NOT NULL) THEN 55
            ELSE 0
        END) +
        CASE
            WHEN (v.ticker IS NOT NULL) THEN 50
            ELSE 0
        END) AS total_features,
    ((((COALESCE(t.quality_score, (0)::numeric) + COALESCE(m.quality_score, (0)::numeric)) + COALESCE(vol.quality_score, (0)::numeric)) + COALESCE(v.quality_score, (0)::numeric)) / (NULLIF((((
        CASE
            WHEN (t.quality_score IS NOT NULL) THEN 1
            ELSE 0
        END +
        CASE
            WHEN (m.quality_score IS NOT NULL) THEN 1
            ELSE 0
        END) +
        CASE
            WHEN (vol.quality_score IS NOT NULL) THEN 1
            ELSE 0
        END) +
        CASE
            WHEN (v.quality_score IS NOT NULL) THEN 1
            ELSE 0
        END), 0))::numeric) AS avg_quality_score,
    GREATEST(COALESCE(t.created_at, '1900-01-01 00:00:00'::timestamp without time zone), COALESCE(m.created_at, '1900-01-01 00:00:00'::timestamp without time zone), COALESCE(vol.created_at, '1900-01-01 00:00:00'::timestamp without time zone), COALESCE(v.created_at, '1900-01-01 00:00:00'::timestamp without time zone)) AS last_calculated
   FROM (((public.technical_features_trend t
     FULL JOIN public.technical_features_momentum m ON ((((t.ticker)::text = (m.ticker)::text) AND (t.date = m.date))))
     FULL JOIN public.technical_features_volatility vol ON ((((COALESCE(t.ticker, m.ticker))::text = (vol.ticker)::text) AND (COALESCE(t.date, m.date) = vol.date))))
     FULL JOIN public.technical_features_volume v ON ((((COALESCE(t.ticker, m.ticker, vol.ticker))::text = (v.ticker)::text) AND (COALESCE(t.date, m.date, vol.date) = v.date))))
  WHERE ((COALESCE(t.ticker, m.ticker, vol.ticker, v.ticker) IS NOT NULL) AND (COALESCE(t.date, m.date, vol.date, v.date) IS NOT NULL))
  ORDER BY COALESCE(t.ticker, m.ticker, vol.ticker, v.ticker), COALESCE(t.date, m.date, vol.date, v.date);


ALTER VIEW public.v_feature_summary OWNER TO postgres;

--
-- TOC entry 238 (class 1259 OID 132788)
-- Name: v_table_sizes; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.v_table_sizes AS
 SELECT schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(((((schemaname)::text || '.'::text) || (tablename)::text))::regclass)) AS size,
    pg_total_relation_size(((((schemaname)::text || '.'::text) || (tablename)::text))::regclass) AS size_bytes
   FROM pg_tables
  ORDER BY (pg_total_relation_size(((((schemaname)::text || '.'::text) || (tablename)::text))::regclass)) DESC;


ALTER VIEW public.v_table_sizes OWNER TO postgres;

--
-- TOC entry 237 (class 1259 OID 132782)
-- Name: v_technical_features_unified; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.v_technical_features_unified AS
 SELECT t.ticker,
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
   FROM (((public.technical_features_trend t
     FULL JOIN public.technical_features_momentum m ON ((((t.ticker)::text = (m.ticker)::text) AND (t.date = m.date))))
     FULL JOIN public.technical_features_volatility v ON ((((t.ticker)::text = (v.ticker)::text) AND (t.date = v.date))))
     FULL JOIN public.technical_features_volume vol ON ((((t.ticker)::text = (vol.ticker)::text) AND (t.date = vol.date))))
  ORDER BY t.ticker, t.date;


ALTER VIEW public.v_technical_features_unified OWNER TO postgres;

--
-- TOC entry 4838 (class 2604 OID 132643)
-- Name: feature_calculation_jobs id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.feature_calculation_jobs ALTER COLUMN id SET DEFAULT nextval('public.feature_calculation_jobs_id_seq'::regclass);


--
-- TOC entry 4862 (class 2604 OID 320814)
-- Name: fundamental_facts_v2 id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.fundamental_facts_v2 ALTER COLUMN id SET DEFAULT nextval('public.fundamental_facts_v2_id_seq'::regclass);


--
-- TOC entry 4820 (class 2604 OID 131772)
-- Name: historical_prices id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.historical_prices ALTER COLUMN id SET DEFAULT nextval('public.historical_prices_id_seq'::regclass);


--
-- TOC entry 4830 (class 2604 OID 131866)
-- Name: polygon_news_articles id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.polygon_news_articles ALTER COLUMN id SET DEFAULT nextval('public.polygon_news_articles_id_seq'::regclass);


--
-- TOC entry 4836 (class 2604 OID 131897)
-- Name: polygon_news_insights id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.polygon_news_insights ALTER COLUMN id SET DEFAULT nextval('public.polygon_news_insights_id_seq'::regclass);


--
-- TOC entry 4834 (class 2604 OID 131882)
-- Name: polygon_news_tickers id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.polygon_news_tickers ALTER COLUMN id SET DEFAULT nextval('public.polygon_news_tickers_id_seq'::regclass);


--
-- TOC entry 4851 (class 2604 OID 287582)
-- Name: raw_fundamental_data id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.raw_fundamental_data ALTER COLUMN id SET DEFAULT nextval('public.raw_fundamental_data_id_seq'::regclass);


--
-- TOC entry 4860 (class 2604 OID 312313)
-- Name: raw_fundamental_json id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.raw_fundamental_json ALTER COLUMN id SET DEFAULT nextval('public.raw_fundamental_json_id_seq'::regclass);


--
-- TOC entry 4845 (class 2604 OID 132747)
-- Name: technical_features_momentum id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.technical_features_momentum ALTER COLUMN id SET DEFAULT nextval('public.technical_features_momentum_id_seq'::regclass);


--
-- TOC entry 4843 (class 2604 OID 132734)
-- Name: technical_features_trend id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.technical_features_trend ALTER COLUMN id SET DEFAULT nextval('public.technical_features_trend_id_seq'::regclass);


--
-- TOC entry 4847 (class 2604 OID 132760)
-- Name: technical_features_volatility id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.technical_features_volatility ALTER COLUMN id SET DEFAULT nextval('public.technical_features_volatility_id_seq'::regclass);


--
-- TOC entry 4849 (class 2604 OID 132773)
-- Name: technical_features_volume id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.technical_features_volume ALTER COLUMN id SET DEFAULT nextval('public.technical_features_volume_id_seq'::regclass);


--
-- TOC entry 4823 (class 2604 OID 131786)
-- Name: tickers id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.tickers ALTER COLUMN id SET DEFAULT nextval('public.tickers_id_seq'::regclass);


--
-- TOC entry 4975 (class 2606 OID 439954)
-- Name: dividends dividends_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.dividends
    ADD CONSTRAINT dividends_pkey PRIMARY KEY (id);


--
-- TOC entry 4919 (class 2606 OID 132654)
-- Name: feature_calculation_jobs feature_calculation_jobs_job_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.feature_calculation_jobs
    ADD CONSTRAINT feature_calculation_jobs_job_id_key UNIQUE (job_id);


--
-- TOC entry 4921 (class 2606 OID 132652)
-- Name: feature_calculation_jobs feature_calculation_jobs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.feature_calculation_jobs
    ADD CONSTRAINT feature_calculation_jobs_pkey PRIMARY KEY (id);


--
-- TOC entry 4971 (class 2606 OID 320826)
-- Name: fundamental_facts_v2 fundamental_facts_v2_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.fundamental_facts_v2
    ADD CONSTRAINT fundamental_facts_v2_pkey PRIMARY KEY (id);


--
-- TOC entry 4973 (class 2606 OID 320828)
-- Name: fundamental_facts_v2 fundamental_facts_v2_unique; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.fundamental_facts_v2
    ADD CONSTRAINT fundamental_facts_v2_unique UNIQUE (ticker_id, date);


--
-- TOC entry 4879 (class 2606 OID 131776)
-- Name: historical_prices historical_prices_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.historical_prices
    ADD CONSTRAINT historical_prices_pkey PRIMARY KEY (id);


--
-- TOC entry 4881 (class 2606 OID 131778)
-- Name: historical_prices historical_prices_ticker_date_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.historical_prices
    ADD CONSTRAINT historical_prices_ticker_date_key UNIQUE (ticker, date);


--
-- TOC entry 4937 (class 2606 OID 132752)
-- Name: technical_features_momentum momentum_features_unique; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.technical_features_momentum
    ADD CONSTRAINT momentum_features_unique UNIQUE (ticker, date);


--
-- TOC entry 4904 (class 2606 OID 131875)
-- Name: polygon_news_articles polygon_news_articles_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.polygon_news_articles
    ADD CONSTRAINT polygon_news_articles_pkey PRIMARY KEY (id);


--
-- TOC entry 4906 (class 2606 OID 131877)
-- Name: polygon_news_articles polygon_news_articles_polygon_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.polygon_news_articles
    ADD CONSTRAINT polygon_news_articles_polygon_id_key UNIQUE (polygon_id);


--
-- TOC entry 4917 (class 2606 OID 131903)
-- Name: polygon_news_insights polygon_news_insights_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.polygon_news_insights
    ADD CONSTRAINT polygon_news_insights_pkey PRIMARY KEY (id);


--
-- TOC entry 4910 (class 2606 OID 131887)
-- Name: polygon_news_tickers polygon_news_tickers_article_id_ticker_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.polygon_news_tickers
    ADD CONSTRAINT polygon_news_tickers_article_id_ticker_key UNIQUE (article_id, ticker);


--
-- TOC entry 4912 (class 2606 OID 131885)
-- Name: polygon_news_tickers polygon_news_tickers_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.polygon_news_tickers
    ADD CONSTRAINT polygon_news_tickers_pkey PRIMARY KEY (id);


--
-- TOC entry 4961 (class 2606 OID 287590)
-- Name: raw_fundamental_data raw_fundamental_data_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.raw_fundamental_data
    ADD CONSTRAINT raw_fundamental_data_pkey PRIMARY KEY (id);


--
-- TOC entry 4967 (class 2606 OID 312318)
-- Name: raw_fundamental_json raw_fundamental_json_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.raw_fundamental_json
    ADD CONSTRAINT raw_fundamental_json_pkey PRIMARY KEY (id);


--
-- TOC entry 4969 (class 2606 OID 312320)
-- Name: raw_fundamental_json raw_fundamental_json_response_hash_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.raw_fundamental_json
    ADD CONSTRAINT raw_fundamental_json_response_hash_key UNIQUE (response_hash);


--
-- TOC entry 4939 (class 2606 OID 132750)
-- Name: technical_features_momentum technical_features_momentum_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.technical_features_momentum
    ADD CONSTRAINT technical_features_momentum_pkey PRIMARY KEY (id);


--
-- TOC entry 4930 (class 2606 OID 132737)
-- Name: technical_features_trend technical_features_trend_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.technical_features_trend
    ADD CONSTRAINT technical_features_trend_pkey PRIMARY KEY (id);


--
-- TOC entry 4944 (class 2606 OID 132763)
-- Name: technical_features_volatility technical_features_volatility_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.technical_features_volatility
    ADD CONSTRAINT technical_features_volatility_pkey PRIMARY KEY (id);


--
-- TOC entry 4951 (class 2606 OID 132776)
-- Name: technical_features_volume technical_features_volume_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.technical_features_volume
    ADD CONSTRAINT technical_features_volume_pkey PRIMARY KEY (id);


--
-- TOC entry 4895 (class 2606 OID 131795)
-- Name: tickers tickers_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.tickers
    ADD CONSTRAINT tickers_pkey PRIMARY KEY (id);


--
-- TOC entry 4897 (class 2606 OID 131797)
-- Name: tickers tickers_ticker_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.tickers
    ADD CONSTRAINT tickers_ticker_key UNIQUE (ticker);


--
-- TOC entry 4932 (class 2606 OID 132739)
-- Name: technical_features_trend trend_features_unique; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.technical_features_trend
    ADD CONSTRAINT trend_features_unique UNIQUE (ticker, date);


--
-- TOC entry 4963 (class 2606 OID 287592)
-- Name: raw_fundamental_data unique_ticker_date_raw_fundamental; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.raw_fundamental_data
    ADD CONSTRAINT unique_ticker_date_raw_fundamental UNIQUE (ticker_id, date);


--
-- TOC entry 4946 (class 2606 OID 132765)
-- Name: technical_features_volatility volatility_features_unique; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.technical_features_volatility
    ADD CONSTRAINT volatility_features_unique UNIQUE (ticker, date);


--
-- TOC entry 4953 (class 2606 OID 132778)
-- Name: technical_features_volume volume_features_unique; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.technical_features_volume
    ADD CONSTRAINT volume_features_unique UNIQUE (ticker, date);


--
-- TOC entry 4976 (class 1259 OID 439961)
-- Name: idx_dividends_ex_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_dividends_ex_date ON public.dividends USING btree (ex_dividend_date);


--
-- TOC entry 4977 (class 1259 OID 439960)
-- Name: idx_dividends_ticker; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_dividends_ticker ON public.dividends USING btree (ticker_id);


--
-- TOC entry 4922 (class 1259 OID 132658)
-- Name: idx_feature_calculation_jobs_created_at; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_feature_calculation_jobs_created_at ON public.feature_calculation_jobs USING btree (created_at);


--
-- TOC entry 4923 (class 1259 OID 132655)
-- Name: idx_feature_calculation_jobs_job_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_feature_calculation_jobs_job_id ON public.feature_calculation_jobs USING btree (job_id);


--
-- TOC entry 4924 (class 1259 OID 132657)
-- Name: idx_feature_calculation_jobs_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_feature_calculation_jobs_status ON public.feature_calculation_jobs USING btree (status);


--
-- TOC entry 4925 (class 1259 OID 132656)
-- Name: idx_feature_calculation_jobs_ticker; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_feature_calculation_jobs_ticker ON public.feature_calculation_jobs USING btree (ticker);


--
-- TOC entry 4882 (class 1259 OID 132475)
-- Name: idx_historical_prices_created_at; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_historical_prices_created_at ON public.historical_prices USING btree (created_at);


--
-- TOC entry 4883 (class 1259 OID 131780)
-- Name: idx_historical_prices_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_historical_prices_date ON public.historical_prices USING btree (date);


--
-- TOC entry 4884 (class 1259 OID 131781)
-- Name: idx_historical_prices_ticker; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_historical_prices_ticker ON public.historical_prices USING btree (ticker);


--
-- TOC entry 4885 (class 1259 OID 131779)
-- Name: idx_historical_prices_ticker_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_historical_prices_ticker_date ON public.historical_prices USING btree (ticker, date);


--
-- TOC entry 4933 (class 1259 OID 132754)
-- Name: idx_momentum_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_momentum_date ON public.technical_features_momentum USING btree (date);


--
-- TOC entry 4934 (class 1259 OID 132753)
-- Name: idx_momentum_ticker; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_momentum_ticker ON public.technical_features_momentum USING btree (ticker);


--
-- TOC entry 4935 (class 1259 OID 132755)
-- Name: idx_momentum_ticker_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_momentum_ticker_date ON public.technical_features_momentum USING btree (ticker, date);


--
-- TOC entry 4913 (class 1259 OID 131917)
-- Name: idx_polygon_insights_article; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_polygon_insights_article ON public.polygon_news_insights USING btree (article_id);


--
-- TOC entry 4914 (class 1259 OID 131916)
-- Name: idx_polygon_insights_sentiment; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_polygon_insights_sentiment ON public.polygon_news_insights USING btree (sentiment);


--
-- TOC entry 4915 (class 1259 OID 131918)
-- Name: idx_polygon_insights_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_polygon_insights_type ON public.polygon_news_insights USING btree (insight_type);


--
-- TOC entry 4898 (class 1259 OID 131913)
-- Name: idx_polygon_news_polygon_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_polygon_news_polygon_id ON public.polygon_news_articles USING btree (polygon_id);


--
-- TOC entry 4899 (class 1259 OID 131911)
-- Name: idx_polygon_news_processed; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_polygon_news_processed ON public.polygon_news_articles USING btree (is_processed);


--
-- TOC entry 4900 (class 1259 OID 131909)
-- Name: idx_polygon_news_published_utc; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_polygon_news_published_utc ON public.polygon_news_articles USING btree (published_utc);


--
-- TOC entry 4901 (class 1259 OID 131910)
-- Name: idx_polygon_news_publisher; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_polygon_news_publisher ON public.polygon_news_articles USING btree (publisher_name);


--
-- TOC entry 4902 (class 1259 OID 131912)
-- Name: idx_polygon_news_quality; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_polygon_news_quality ON public.polygon_news_articles USING btree (quality_score);


--
-- TOC entry 4907 (class 1259 OID 131915)
-- Name: idx_polygon_tickers_article; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_polygon_tickers_article ON public.polygon_news_tickers USING btree (article_id);


--
-- TOC entry 4908 (class 1259 OID 131914)
-- Name: idx_polygon_tickers_ticker; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_polygon_tickers_ticker ON public.polygon_news_tickers USING btree (ticker);


--
-- TOC entry 4964 (class 1259 OID 312322)
-- Name: idx_raw_fund_json_filing; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_raw_fund_json_filing ON public.raw_fundamental_json USING btree (filing_date);


--
-- TOC entry 4965 (class 1259 OID 312321)
-- Name: idx_raw_fund_json_ticker_end; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_raw_fund_json_ticker_end ON public.raw_fundamental_json USING btree (ticker_id, period_end DESC);


--
-- TOC entry 4954 (class 1259 OID 287598)
-- Name: idx_raw_fundamental_created_at; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_raw_fundamental_created_at ON public.raw_fundamental_data USING btree (created_at);


--
-- TOC entry 4955 (class 1259 OID 287599)
-- Name: idx_raw_fundamental_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_raw_fundamental_date ON public.raw_fundamental_data USING btree (date);


--
-- TOC entry 4956 (class 1259 OID 287602)
-- Name: idx_raw_fundamental_filing_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_raw_fundamental_filing_date ON public.raw_fundamental_data USING btree (filing_date);


--
-- TOC entry 4957 (class 1259 OID 287700)
-- Name: idx_raw_fundamental_sic_code; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_raw_fundamental_sic_code ON public.raw_fundamental_data USING btree (sic_code);


--
-- TOC entry 4958 (class 1259 OID 287601)
-- Name: idx_raw_fundamental_ticker_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_raw_fundamental_ticker_date ON public.raw_fundamental_data USING btree (ticker_id, date);


--
-- TOC entry 4959 (class 1259 OID 287600)
-- Name: idx_raw_fundamental_ticker_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_raw_fundamental_ticker_id ON public.raw_fundamental_data USING btree (ticker_id);


--
-- TOC entry 4886 (class 1259 OID 131812)
-- Name: idx_tickers_active; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_tickers_active ON public.tickers USING btree (active);


--
-- TOC entry 4887 (class 1259 OID 133115)
-- Name: idx_tickers_cik; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_tickers_cik ON public.tickers USING btree (cik);


--
-- TOC entry 4888 (class 1259 OID 133116)
-- Name: idx_tickers_composite_figi; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_tickers_composite_figi ON public.tickers USING btree (composite_figi);


--
-- TOC entry 4889 (class 1259 OID 133119)
-- Name: idx_tickers_list_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_tickers_list_date ON public.tickers USING btree (list_date);


--
-- TOC entry 4890 (class 1259 OID 131811)
-- Name: idx_tickers_market; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_tickers_market ON public.tickers USING btree (market);


--
-- TOC entry 4891 (class 1259 OID 133117)
-- Name: idx_tickers_share_class_figi; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_tickers_share_class_figi ON public.tickers USING btree (share_class_figi);


--
-- TOC entry 4892 (class 1259 OID 133118)
-- Name: idx_tickers_sic_code; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_tickers_sic_code ON public.tickers USING btree (sic_code);


--
-- TOC entry 4893 (class 1259 OID 131810)
-- Name: idx_tickers_ticker; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_tickers_ticker ON public.tickers USING btree (ticker);


--
-- TOC entry 4926 (class 1259 OID 132741)
-- Name: idx_trend_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_trend_date ON public.technical_features_trend USING btree (date);


--
-- TOC entry 4927 (class 1259 OID 132740)
-- Name: idx_trend_ticker; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_trend_ticker ON public.technical_features_trend USING btree (ticker);


--
-- TOC entry 4928 (class 1259 OID 132742)
-- Name: idx_trend_ticker_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_trend_ticker_date ON public.technical_features_trend USING btree (ticker, date);


--
-- TOC entry 4940 (class 1259 OID 132767)
-- Name: idx_volatility_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_volatility_date ON public.technical_features_volatility USING btree (date);


--
-- TOC entry 4941 (class 1259 OID 132766)
-- Name: idx_volatility_ticker; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_volatility_ticker ON public.technical_features_volatility USING btree (ticker);


--
-- TOC entry 4942 (class 1259 OID 132768)
-- Name: idx_volatility_ticker_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_volatility_ticker_date ON public.technical_features_volatility USING btree (ticker, date);


--
-- TOC entry 4947 (class 1259 OID 132780)
-- Name: idx_volume_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_volume_date ON public.technical_features_volume USING btree (date);


--
-- TOC entry 4948 (class 1259 OID 132779)
-- Name: idx_volume_ticker; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_volume_ticker ON public.technical_features_volume USING btree (ticker);


--
-- TOC entry 4949 (class 1259 OID 132781)
-- Name: idx_volume_ticker_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_volume_ticker_date ON public.technical_features_volume USING btree (ticker, date);


--
-- TOC entry 4978 (class 1259 OID 439962)
-- Name: ux_dividends_ticker_ex_pay_amt; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX ux_dividends_ticker_ex_pay_amt ON public.dividends USING btree (ticker_id, ex_dividend_date, pay_date, cash_amount);


--
-- TOC entry 4984 (class 2620 OID 132722)
-- Name: historical_prices update_historical_prices_updated_at; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER update_historical_prices_updated_at BEFORE UPDATE ON public.historical_prices FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- TOC entry 4985 (class 2620 OID 287603)
-- Name: raw_fundamental_data update_raw_fundamental_updated_at; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER update_raw_fundamental_updated_at BEFORE UPDATE ON public.raw_fundamental_data FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- TOC entry 4982 (class 2606 OID 320829)
-- Name: fundamental_facts_v2 fk_facts_v2_ticker_id; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.fundamental_facts_v2
    ADD CONSTRAINT fk_facts_v2_ticker_id FOREIGN KEY (ticker_id) REFERENCES public.tickers(id) ON DELETE CASCADE;


--
-- TOC entry 4981 (class 2606 OID 287593)
-- Name: raw_fundamental_data fk_raw_fundamental_ticker_id; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.raw_fundamental_data
    ADD CONSTRAINT fk_raw_fundamental_ticker_id FOREIGN KEY (ticker_id) REFERENCES public.tickers(id) ON DELETE CASCADE;


--
-- TOC entry 4983 (class 2606 OID 439955)
-- Name: dividends fk_ticker; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.dividends
    ADD CONSTRAINT fk_ticker FOREIGN KEY (ticker_id) REFERENCES public.tickers(id);


--
-- TOC entry 4980 (class 2606 OID 131904)
-- Name: polygon_news_insights polygon_news_insights_article_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.polygon_news_insights
    ADD CONSTRAINT polygon_news_insights_article_id_fkey FOREIGN KEY (article_id) REFERENCES public.polygon_news_articles(id) ON DELETE CASCADE;


--
-- TOC entry 4979 (class 2606 OID 131888)
-- Name: polygon_news_tickers polygon_news_tickers_article_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.polygon_news_tickers
    ADD CONSTRAINT polygon_news_tickers_article_id_fkey FOREIGN KEY (article_id) REFERENCES public.polygon_news_articles(id) ON DELETE CASCADE;


-- Completed on 2025-09-23 20:08:04

--
-- PostgreSQL database dump complete
--

-- Completed on 2025-09-23 20:08:04

--
-- PostgreSQL database cluster dump complete
--


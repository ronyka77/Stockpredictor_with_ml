"""
SQLAlchemy Models for Fundamental Features

This module defines the database models for fundamental analysis data
including ratios, growth metrics, scores, and sector analysis.
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Date,
    DECIMAL,
    Boolean,
    TIMESTAMP,
    CheckConstraint,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import date
from typing import Optional

# Database constraint constants
TICKER_LENGTH_CONSTRAINT = "LENGTH(ticker) >= 1"
MIN_DATE_CONSTRAINT = "date >= '2020-01-01'"

Base = declarative_base()


class FundamentalRatios(Base):
    """Core financial ratios including valuation, profitability, liquidity, and leverage"""

    __tablename__ = "fundamental_ratios"

    # Primary key and identifiers
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())
    updated_at = Column(
        TIMESTAMP, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    # Valuation Ratios
    pe_ratio = Column(DECIMAL(10, 4))
    pb_ratio = Column(DECIMAL(10, 4))
    ps_ratio = Column(DECIMAL(10, 4))
    ev_ebitda = Column(DECIMAL(10, 4))
    peg_ratio = Column(DECIMAL(10, 4))

    # Profitability Ratios
    roe = Column(DECIMAL(10, 4))
    roa = Column(DECIMAL(10, 4))
    roi = Column(DECIMAL(10, 4))
    gross_margin = Column(DECIMAL(10, 4))
    operating_margin = Column(DECIMAL(10, 4))
    net_margin = Column(DECIMAL(10, 4))

    # Liquidity Ratios
    current_ratio = Column(DECIMAL(10, 4))
    quick_ratio = Column(DECIMAL(10, 4))
    cash_ratio = Column(DECIMAL(10, 4))

    # Leverage Ratios
    debt_to_equity = Column(DECIMAL(10, 4))
    interest_coverage = Column(DECIMAL(10, 4))
    debt_to_assets = Column(DECIMAL(10, 4))

    # Data Quality Metrics
    data_quality_score = Column(DECIMAL(5, 4))
    missing_data_count = Column(Integer, default=0)

    # Constraints
    __table_args__ = (
        UniqueConstraint("ticker", "date", name="unique_ticker_date_ratios"),
        CheckConstraint(TICKER_LENGTH_CONSTRAINT, name="valid_ticker_ratios"),
        CheckConstraint(MIN_DATE_CONSTRAINT, name="valid_date_ratios"),
    )

    def __repr__(self):
        return f"<FundamentalRatios(ticker='{self.ticker}', date='{self.date}', pe_ratio={self.pe_ratio})>"


class FundamentalGrowthMetrics(Base):
    """Growth rates and efficiency metrics across multiple periods"""

    __tablename__ = "fundamental_growth_metrics"

    # Primary key and identifiers
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())
    updated_at = Column(
        TIMESTAMP, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    # Revenue Growth Rates (CAGR)
    revenue_growth_1y = Column(DECIMAL(10, 4))
    revenue_growth_3y = Column(DECIMAL(10, 4))

    # Earnings Growth Rates (CAGR)
    earnings_growth_1y = Column(DECIMAL(10, 4))
    earnings_growth_3y = Column(DECIMAL(10, 4))

    # Book Value Growth Rates (CAGR)
    book_value_growth_1y = Column(DECIMAL(10, 4))
    book_value_growth_3y = Column(DECIMAL(10, 4))

    # Asset Growth Rates
    asset_growth_1y = Column(DECIMAL(10, 4))
    asset_growth_3y = Column(DECIMAL(10, 4))

    # Efficiency Metrics
    asset_turnover = Column(DECIMAL(10, 4))
    inventory_turnover = Column(DECIMAL(10, 4))
    receivables_turnover = Column(DECIMAL(10, 4))
    working_capital_turnover = Column(DECIMAL(10, 4))

    # Additional Efficiency Ratios
    cash_conversion_cycle = Column(DECIMAL(10, 4))
    days_sales_outstanding = Column(DECIMAL(10, 4))
    days_inventory_outstanding = Column(DECIMAL(10, 4))

    # Data Quality Metrics
    data_quality_score = Column(DECIMAL(5, 4))
    missing_data_count = Column(Integer, default=0)

    # Constraints
    __table_args__ = (
        UniqueConstraint("ticker", "date", name="unique_ticker_date_growth"),
        CheckConstraint(TICKER_LENGTH_CONSTRAINT, name="valid_ticker_growth"),
        CheckConstraint(MIN_DATE_CONSTRAINT, name="valid_date_growth"),
    )

    def __repr__(self):
        return f"<FundamentalGrowthMetrics(ticker='{self.ticker}', date='{self.date}', revenue_growth_1y={self.revenue_growth_1y})>"


class FundamentalScores(Base):
    """Advanced scoring systems and financial health metrics"""

    __tablename__ = "fundamental_scores"

    # Primary key and identifiers
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())
    updated_at = Column(
        TIMESTAMP, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    # Bankruptcy Prediction Scores
    altman_z_score = Column(DECIMAL(10, 4))
    piotroski_f_score = Column(Integer)
    ohlson_o_score = Column(DECIMAL(10, 4))
    zmijewski_score = Column(DECIMAL(10, 4))

    # Financial Health Metrics
    earnings_quality_score = Column(DECIMAL(10, 4))
    cash_conversion_ratio = Column(DECIMAL(10, 4))
    working_capital_ratio = Column(DECIMAL(10, 4))
    financial_leverage = Column(DECIMAL(10, 4))

    # Quality Metrics
    accruals_ratio = Column(DECIMAL(10, 4))
    earnings_persistence = Column(DECIMAL(10, 4))
    earnings_predictability = Column(DECIMAL(10, 4))

    # Composite Scores
    financial_health_composite = Column(DECIMAL(10, 4))
    quality_composite = Column(DECIMAL(10, 4))
    bankruptcy_risk_composite = Column(DECIMAL(10, 4))

    # Individual Piotroski Components (for transparency)
    piotroski_roa_positive = Column(Boolean)
    piotroski_cfo_positive = Column(Boolean)
    piotroski_roa_improved = Column(Boolean)
    piotroski_cfo_vs_roa = Column(Boolean)
    piotroski_debt_decreased = Column(Boolean)
    piotroski_current_ratio_improved = Column(Boolean)
    piotroski_shares_outstanding = Column(Boolean)
    piotroski_gross_margin_improved = Column(Boolean)
    piotroski_asset_turnover_improved = Column(Boolean)

    # Data Quality Metrics
    data_quality_score = Column(DECIMAL(5, 4))
    missing_data_count = Column(Integer, default=0)

    # Constraints
    __table_args__ = (
        UniqueConstraint("ticker", "date", name="unique_ticker_date_scores"),
        CheckConstraint(TICKER_LENGTH_CONSTRAINT, name="valid_ticker_scores"),
        CheckConstraint(MIN_DATE_CONSTRAINT, name="valid_date_scores"),
        CheckConstraint(
            "piotroski_f_score >= 0 AND piotroski_f_score <= 9", name="valid_piotroski_score"
        ),
    )

    def __repr__(self):
        return f"<FundamentalScores(ticker='{self.ticker}', date='{self.date}', altman_z_score={self.altman_z_score}, piotroski_f_score={self.piotroski_f_score})>"


class FundamentalSectorAnalysis(Base):
    """Cross-sectional analysis and sector-relative metrics"""

    __tablename__ = "fundamental_sector_analysis"

    # Primary key and identifiers
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())
    updated_at = Column(
        TIMESTAMP, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    # GICS Classification
    gics_sector = Column(String(50))
    gics_industry_group = Column(String(100))
    gics_industry = Column(String(100))
    gics_sub_industry = Column(String(100))

    # Sector Percentile Rankings (0-100)
    pe_sector_percentile = Column(DECIMAL(5, 2))
    pb_sector_percentile = Column(DECIMAL(5, 2))
    ps_sector_percentile = Column(DECIMAL(5, 2))
    roe_sector_percentile = Column(DECIMAL(5, 2))
    roa_sector_percentile = Column(DECIMAL(5, 2))
    debt_to_equity_sector_percentile = Column(DECIMAL(5, 2))

    # Industry Percentile Rankings (0-100)
    pe_industry_percentile = Column(DECIMAL(5, 2))
    pb_industry_percentile = Column(DECIMAL(5, 2))
    roe_industry_percentile = Column(DECIMAL(5, 2))

    # Market Percentile Rankings (0-100)
    pe_market_percentile = Column(DECIMAL(5, 2))
    pb_market_percentile = Column(DECIMAL(5, 2))
    roe_market_percentile = Column(DECIMAL(5, 2))

    # Sector Relative Ratios (Individual / Sector Median)
    pe_relative_to_sector = Column(DECIMAL(10, 4))
    pb_relative_to_sector = Column(DECIMAL(10, 4))
    ps_relative_to_sector = Column(DECIMAL(10, 4))
    roe_relative_to_sector = Column(DECIMAL(10, 4))
    roa_relative_to_sector = Column(DECIMAL(10, 4))

    # Sector Benchmark Values
    sector_median_pe = Column(DECIMAL(10, 4))
    sector_median_pb = Column(DECIMAL(10, 4))
    sector_median_ps = Column(DECIMAL(10, 4))
    sector_median_roe = Column(DECIMAL(10, 4))
    sector_median_roa = Column(DECIMAL(10, 4))

    # Industry Benchmark Values
    industry_median_pe = Column(DECIMAL(10, 4))
    industry_median_pb = Column(DECIMAL(10, 4))
    industry_median_roe = Column(DECIMAL(10, 4))

    # Market Benchmark Values
    market_median_pe = Column(DECIMAL(10, 4))
    market_median_pb = Column(DECIMAL(10, 4))
    market_median_roe = Column(DECIMAL(10, 4))

    # Cross-Sectional Rankings
    sector_rank = Column(Integer)
    industry_rank = Column(Integer)
    market_rank = Column(Integer)
    sector_total_companies = Column(Integer)
    industry_total_companies = Column(Integer)
    market_total_companies = Column(Integer)

    # Data Quality Metrics
    data_quality_score = Column(DECIMAL(5, 4))
    missing_data_count = Column(Integer, default=0)

    # Constraints
    __table_args__ = (
        UniqueConstraint("ticker", "date", name="unique_ticker_date_sector"),
        CheckConstraint(TICKER_LENGTH_CONSTRAINT, name="valid_ticker_sector"),
        CheckConstraint(MIN_DATE_CONSTRAINT, name="valid_date_sector"),
        CheckConstraint(
            "pe_sector_percentile >= 0 AND pe_sector_percentile <= 100 AND "
            "pb_sector_percentile >= 0 AND pb_sector_percentile <= 100 AND "
            "roe_sector_percentile >= 0 AND roe_sector_percentile <= 100",
            name="valid_percentiles",
        ),
    )

    def __repr__(self):
        return f"<FundamentalSectorAnalysis(ticker='{self.ticker}', date='{self.date}', gics_sector='{self.gics_sector}')>"


# Utility functions for model operations
def get_latest_fundamental_data(session, ticker: str) -> Optional[dict]:
    """Get the latest fundamental data for a ticker across all tables"""

    # Get latest dates for each table
    latest_ratios = (
        session.query(FundamentalRatios)
        .filter_by(ticker=ticker)
        .order_by(FundamentalRatios.date.desc())
        .first()
    )
    latest_growth = (
        session.query(FundamentalGrowthMetrics)
        .filter_by(ticker=ticker)
        .order_by(FundamentalGrowthMetrics.date.desc())
        .first()
    )
    latest_scores = (
        session.query(FundamentalScores)
        .filter_by(ticker=ticker)
        .order_by(FundamentalScores.date.desc())
        .first()
    )
    latest_sector = (
        session.query(FundamentalSectorAnalysis)
        .filter_by(ticker=ticker)
        .order_by(FundamentalSectorAnalysis.date.desc())
        .first()
    )

    if not latest_ratios:
        return None

    return {
        "ticker": ticker,
        "ratios": latest_ratios,
        "growth": latest_growth,
        "scores": latest_scores,
        "sector": latest_sector,
    }


def get_fundamental_data_by_date(session, ticker: str, target_date: date) -> Optional[dict]:
    """Get fundamental data for a specific ticker and date"""

    ratios = session.query(FundamentalRatios).filter_by(ticker=ticker, date=target_date).first()
    growth = (
        session.query(FundamentalGrowthMetrics).filter_by(ticker=ticker, date=target_date).first()
    )
    scores = session.query(FundamentalScores).filter_by(ticker=ticker, date=target_date).first()
    sector = (
        session.query(FundamentalSectorAnalysis).filter_by(ticker=ticker, date=target_date).first()
    )

    if not ratios:
        return None

    return {
        "ticker": ticker,
        "date": target_date,
        "ratios": ratios,
        "growth": growth,
        "scores": scores,
        "sector": sector,
    }


def get_sector_companies(session, gics_sector: str, target_date: date) -> list:
    """Get all companies in a specific GICS sector for a given date"""

    return (
        session.query(FundamentalSectorAnalysis)
        .filter_by(gics_sector=gics_sector, date=target_date)
        .all()
    )


def calculate_data_quality_score(missing_count: int, total_fields: int) -> float:
    """Calculate data quality score based on missing data count"""
    if total_fields == 0:
        return 0.0

    completeness = (total_fields - missing_count) / total_fields
    return round(completeness, 4)

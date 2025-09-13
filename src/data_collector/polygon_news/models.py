"""
SQLAlchemy models for Polygon news data storage
Optimized for Polygon.io News API response structure with 2-year data retention
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone

from src.data_collector.config import config
from src.utils.logger import get_logger
from src.database.connection import execute

logger = get_logger(__name__, utility="data_collector")


# Raw SQL DDL for required tables
_CREATE_ARTICLES_SQL = """
CREATE TABLE IF NOT EXISTS polygon_news_articles (
    id SERIAL PRIMARY KEY,
    polygon_id VARCHAR(100) UNIQUE NOT NULL,
    title VARCHAR(1000) NOT NULL,
    description TEXT,
    article_url VARCHAR(2000) NOT NULL,
    amp_url VARCHAR(2000),
    image_url VARCHAR(2000),
    author VARCHAR(200),
    published_utc TIMESTAMPTZ NOT NULL,
    publisher_name VARCHAR(200),
    publisher_homepage_url VARCHAR(500),
    publisher_logo_url VARCHAR(500),
    publisher_favicon_url VARCHAR(500),
    keywords TEXT[],
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    is_processed BOOLEAN DEFAULT FALSE,
    processing_errors TEXT,
    quality_score DOUBLE PRECISION,
    relevance_score DOUBLE PRECISION
);
"""

_CREATE_TICKERS_SQL = """
CREATE TABLE IF NOT EXISTS polygon_news_tickers (
    id SERIAL PRIMARY KEY,
    article_id INTEGER NOT NULL REFERENCES polygon_news_articles(id) ON DELETE CASCADE,
    ticker VARCHAR(10) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);
"""

_CREATE_INSIGHTS_SQL = """
CREATE TABLE IF NOT EXISTS polygon_news_insights (
    id SERIAL PRIMARY KEY,
    article_id INTEGER NOT NULL REFERENCES polygon_news_articles(id) ON DELETE CASCADE,
    sentiment VARCHAR(20),
    sentiment_reasoning TEXT,
    insight_type VARCHAR(50) NOT NULL,
    insight_value TEXT,
    confidence_score DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT now()
);
"""


def create_tables() -> None:
    """Create required tables using global DB pool."""
    # Use execute helper to run DDL statements
    execute(_CREATE_ARTICLES_SQL, params=None, commit=True)
    execute(_CREATE_TICKERS_SQL, params=None, commit=True)
    execute(_CREATE_INSIGHTS_SQL, params=None, commit=True)
    logger.info("Created Polygon news database tables (if not present)")


def get_retention_cutoff_date(retention_days: Optional[int] = None) -> datetime:
    """Get cutoff date for data retention (default: from config)"""
    if retention_days is None:
        retention_days = config.NEWS_RETENTION_YEARS * 365
    return datetime.now(timezone.utc) - timedelta(days=retention_days)


def validate_article_data(article_data: Dict[str, Any]) -> bool:
    """Validate required fields for article data"""
    required_fields = ["polygon_id", "title", "article_url", "published_utc"]

    for field in required_fields:
        if not article_data.get(field):
            logger.warning(f"Missing required field: {field}")
            return False

    return True

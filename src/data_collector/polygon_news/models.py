"""
SQLAlchemy models for Polygon news data storage
Optimized for Polygon.io News API response structure with 2-year data retention
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    Float,
    ForeignKey,
)
from src.data_collector.config import config
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from src.utils.logger import get_logger

Base = declarative_base()
logger = get_logger(__name__, utility="data_collector")


class PolygonNewsArticle(Base):
    """
    Main news article model for Polygon news data
    """

    __tablename__ = "polygon_news_articles"

    id = Column(Integer, primary_key=True)

    # Core article data from Polygon API
    polygon_id = Column(String(100), unique=True, nullable=False, index=True)
    title = Column(String(1000), nullable=False)
    description = Column(Text)
    article_url = Column(String(2000), nullable=False)
    amp_url = Column(String(2000))
    image_url = Column(String(2000))
    author = Column(String(200))

    # Publishing information
    published_utc = Column(DateTime(timezone=True), nullable=False, index=True)
    publisher_name = Column(String(200), index=True)
    publisher_homepage_url = Column(String(500))
    publisher_logo_url = Column(String(500))
    publisher_favicon_url = Column(String(500))

    # Content metadata
    # Store keywords as Postgres text[] (ARRAY of text). Production DB is Postgres so use ARRAY.
    keywords = Column(postgresql.ARRAY(String))

    # Processing metadata
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now()
    )
    is_processed = Column(Boolean, default=False, index=True)
    processing_errors = Column(Text)

    # Quality metrics (our custom scoring)
    quality_score = Column(Float)
    relevance_score = Column(Float)

    # Relationships
    tickers = relationship(
        "PolygonNewsTicker", back_populates="article", cascade="all, delete-orphan"
    )
    insights = relationship(
        "PolygonNewsInsight", back_populates="article", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<PolygonNewsArticle(id={self.id}, polygon_id='{self.polygon_id}', title='{self.title[:50]}...')>"

    @classmethod
    def get_latest_date_for_ticker(
        cls, session: Session, ticker: str
    ) -> Optional[datetime]:
        """Get the latest published date for articles associated with a ticker"""
        result = (
            session.query(func.max(cls.published_utc))
            .join(PolygonNewsTicker)
            .filter(PolygonNewsTicker.ticker == ticker.upper())
            .scalar()
        )
        return result

    @classmethod
    def get_latest_date_overall(cls, session: Session) -> Optional[datetime]:
        """Get the latest published date across all articles"""
        result = session.query(func.max(cls.published_utc)).scalar()
        return result

    @classmethod
    def cleanup_old_articles(
        cls, session: Session, cutoff_date: Optional[datetime] = None
    ) -> int:
        """Remove articles older than cutoff date (default: from config)"""
        if cutoff_date is None:
            from datetime import timezone

            cutoff_date = datetime.now(timezone.utc) - timedelta(
                days=config.NEWS_RETENTION_YEARS * 365
            )

        # Count articles to be deleted
        count = session.query(cls).filter(cls.published_utc < cutoff_date).count()

        if count > 0:
            # Delete old articles (cascades to related records)
            session.query(cls).filter(cls.published_utc < cutoff_date).delete()
            session.commit()
            logger.info(f"Cleaned up {count} articles older than {cutoff_date.date()}")

        return count

    @classmethod
    def get_article_by_polygon_id(
        cls, session: Session, polygon_id: str
    ) -> Optional["PolygonNewsArticle"]:
        """Get article by Polygon ID"""
        return session.query(cls).filter(cls.polygon_id == polygon_id).first()

    def get_ticker_list(self) -> List[str]:
        """Get list of associated ticker symbols"""
        return [ticker.ticker for ticker in self.tickers]

    def get_primary_sentiment(self) -> Optional[Dict[str, Any]]:
        """Get primary sentiment insight"""
        for insight in self.insights:
            if insight.insight_type == "sentiment" and insight.sentiment:
                return {
                    "sentiment": insight.sentiment,
                    "reasoning": insight.sentiment_reasoning,
                    "confidence": insight.confidence_score,
                }
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert article to dictionary representation"""
        return {
            "id": self.id,
            "polygon_id": self.polygon_id,
            "title": self.title,
            "description": self.description,
            "article_url": self.article_url,
            "amp_url": self.amp_url,
            "image_url": self.image_url,
            "author": self.author,
            "published_utc": self.published_utc.isoformat()
            if self.published_utc
            else None,
            "publisher_name": self.publisher_name,
            "keywords": self.keywords,
            "quality_score": self.quality_score,
            "relevance_score": self.relevance_score,
            "tickers": self.get_ticker_list(),
            "sentiment": self.get_primary_sentiment(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class PolygonNewsTicker(Base):
    """
    Association between news articles and stock tickers
    """

    __tablename__ = "polygon_news_tickers"

    id = Column(Integer, primary_key=True)
    article_id = Column(
        Integer,
        ForeignKey("polygon_news_articles.id", ondelete="CASCADE"),
        nullable=False,
    )
    ticker = Column(String(10), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now())

    # Relationships
    article = relationship("PolygonNewsArticle", back_populates="tickers")

    # Unique constraint
    __table_args__ = {"extend_existing": True}

    def __repr__(self):
        return (
            f"<PolygonNewsTicker(article_id={self.article_id}, ticker='{self.ticker}')>"
        )

    @classmethod
    def get_articles_for_ticker(
        cls,
        session: Session,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[PolygonNewsArticle]:
        """Get all articles associated with a ticker within date range"""
        query = (
            session.query(PolygonNewsArticle)
            .join(cls)
            .filter(cls.ticker == ticker.upper())
        )

        if start_date:
            query = query.filter(PolygonNewsArticle.published_utc >= start_date)

        if end_date:
            query = query.filter(PolygonNewsArticle.published_utc <= end_date)

        return query.order_by(PolygonNewsArticle.published_utc.desc()).all()

    @classmethod
    def get_ticker_counts(
        cls,
        session: Session,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[tuple]:
        """Get article counts by ticker within date range"""
        query = session.query(cls.ticker, func.count(cls.id).label("count")).join(
            PolygonNewsArticle
        )

        if start_date:
            query = query.filter(PolygonNewsArticle.published_utc >= start_date)

        if end_date:
            query = query.filter(PolygonNewsArticle.published_utc <= end_date)

        return query.group_by(cls.ticker).order_by(func.count(cls.id).desc()).all()


class PolygonNewsInsight(Base):
    """
    Sentiment analysis and insights from Polygon API
    """

    __tablename__ = "polygon_news_insights"

    id = Column(Integer, primary_key=True)
    article_id = Column(
        Integer,
        ForeignKey("polygon_news_articles.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Sentiment analysis from Polygon
    sentiment = Column(String(20), index=True)  # 'positive', 'negative', 'neutral'
    sentiment_reasoning = Column(Text)

    # Additional insight fields
    insight_type = Column(String(50), nullable=False, index=True)
    insight_value = Column(Text)
    confidence_score = Column(Float)

    created_at = Column(DateTime(timezone=True), default=func.now())

    # Relationships
    article = relationship("PolygonNewsArticle", back_populates="insights")

    def __repr__(self):
        return f"<PolygonNewsInsight(article_id={self.article_id}, type='{self.insight_type}', sentiment='{self.sentiment}')>"

    @classmethod
    def get_sentiment_distribution(
        cls,
        session: Session,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, int]:
        """Get distribution of sentiment across articles"""
        query = (
            session.query(cls.sentiment, func.count(cls.id))
            .filter(cls.insight_type == "sentiment", cls.sentiment.isnot(None))
            .join(PolygonNewsArticle)
        )

        if start_date:
            query = query.filter(PolygonNewsArticle.published_utc >= start_date)

        if end_date:
            query = query.filter(PolygonNewsArticle.published_utc <= end_date)

        results = query.group_by(cls.sentiment).all()
        return {sentiment: count for sentiment, count in results}

    @classmethod
    def get_sentiment_for_ticker(
        cls,
        session: Session,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List["PolygonNewsInsight"]:
        """Get sentiment insights for a specific ticker"""
        query = (
            session.query(cls)
            .join(PolygonNewsArticle)
            .join(PolygonNewsTicker)
            .filter(
                PolygonNewsTicker.ticker == ticker.upper(),
                cls.insight_type == "sentiment",
                cls.sentiment.isnot(None),
            )
        )

        if start_date:
            query = query.filter(PolygonNewsArticle.published_utc >= start_date)

        if end_date:
            query = query.filter(PolygonNewsArticle.published_utc <= end_date)

        return query.order_by(PolygonNewsArticle.published_utc.desc()).all()

    def to_dict(self) -> Dict[str, Any]:
        """Convert insight to dictionary representation"""
        return {
            "id": self.id,
            "article_id": self.article_id,
            "sentiment": self.sentiment,
            "sentiment_reasoning": self.sentiment_reasoning,
            "insight_type": self.insight_type,
            "insight_value": self.insight_value,
            "confidence_score": self.confidence_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# Utility functions for model operations
def create_tables(engine):
    """Create all tables in the database"""
    Base.metadata.create_all(engine)
    logger.info("Created Polygon news database tables")


def get_retention_cutoff_date(retention_days: Optional[int] = None) -> datetime:
    """Get cutoff date for data retention (default: from config)"""
    if retention_days is None:
        retention_days = config.NEWS_RETENTION_YEARS * 365
    from datetime import timezone

    return datetime.now(timezone.utc) - timedelta(days=retention_days)


def validate_article_data(article_data: Dict[str, Any]) -> bool:
    """Validate required fields for article data"""
    required_fields = ["polygon_id", "title", "article_url", "published_utc"]

    for field in required_fields:
        if not article_data.get(field):
            logger.warning(f"Missing required field: {field}")
            return False

    return True

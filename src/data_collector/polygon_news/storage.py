"""
Database storage operations for Polygon news data
Handles upserts, batch processing, and data maintenance
"""

from sqlalchemy.orm import Session
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

from src.data_collector.polygon_news.models import PolygonNewsArticle, PolygonNewsTicker, PolygonNewsInsight, validate_article_data
from src.data_collector.config import config
from src.utils.logger import get_polygon_logger

logger = get_polygon_logger(__name__)


class PolygonNewsStorage:
    """
    Database storage operations for Polygon news data
    """
    
    def __init__(self, session: Session):
        """
        Initialize storage with database session
        
        Args:
            session: SQLAlchemy database session
        """
        self.session = session
        self.logger = get_polygon_logger(self.__class__.__name__)
    
    def store_article(self, article_data: Dict[str, Any]) -> Optional[int]:
        """
        Store a single article with upsert logic
        
        Args:
            article_data: Processed article data from Polygon API
            
        Returns:
            Article ID if successful, None if skipped/failed
        """
        try:
            # Validate article data
            if not validate_article_data(article_data):
                self.logger.warning(f"Invalid article data: {article_data.get('polygon_id', 'unknown')}")
                return None
            
            # Check if article already exists
            existing_article = PolygonNewsArticle.get_article_by_polygon_id(
                self.session, article_data['polygon_id']
            )
            
            if existing_article:
                # Update existing article if needed
                updated = self._update_existing_article(existing_article, article_data)
                if updated:
                    self.logger.info(f"Updated existing article: {article_data['polygon_id']}")
                return existing_article.id
            
            # Create new article
            article_id = self._create_new_article(article_data)
            if article_id:
                self.logger.info(f"Created new article: {article_data['polygon_id']}")
            
            return article_id
            
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Error storing article {article_data.get('polygon_id', 'unknown')}: {e}")
            return None
    
    def store_articles_batch(self, articles_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Store multiple articles in batch with transaction management
        
        Args:
            articles_data: List of processed article data
            
        Returns:
            Statistics dictionary with counts
        """
        stats = {
            'total_processed': 0,
            'new_articles': 0,
            'updated_articles': 0,
            'skipped_articles': 0,
            'failed_articles': 0
        }
        
        self.logger.info(f"Starting batch storage of {len(articles_data)} articles")
        
        for article_data in articles_data:
            stats['total_processed'] += 1
            
            try:
                # Check if article exists
                existing_article = PolygonNewsArticle.get_article_by_polygon_id(
                    self.session, article_data['polygon_id']
                )
                
                if existing_article:
                    # Update existing
                    updated = self._update_existing_article(existing_article, article_data)
                    if updated:
                        stats['updated_articles'] += 1
                    else:
                        stats['skipped_articles'] += 1
                else:
                    # Create new
                    article_id = self._create_new_article(article_data)
                    if article_id:
                        stats['new_articles'] += 1
                    else:
                        stats['failed_articles'] += 1
                
                # Commit every 50 articles to avoid large transactions
                if stats['total_processed'] % 50 == 0:
                    self.session.commit()
                    self.logger.info(f"Committed batch at {stats['total_processed']} articles")
                
            except Exception as e:
                stats['failed_articles'] += 1
                self.logger.error(f"Failed to process article {article_data.get('polygon_id', 'unknown')}: {e}")
                self.session.rollback()
        
        # Final commit
        try:
            self.session.commit()
            self.logger.info(f"Batch storage completed: {stats}")
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Failed to commit final batch: {e}")
            stats['failed_articles'] += stats['new_articles'] + stats['updated_articles']
            stats['new_articles'] = 0
            stats['updated_articles'] = 0
        
        return stats
    
    def _create_new_article(self, article_data: Dict[str, Any]) -> Optional[int]:
        """Create new article with all related data"""
        try:
            # Parse published date
            published_utc = self._parse_datetime(article_data['published_utc'])
            if not published_utc:
                self.logger.warning(f"Invalid published_utc: {article_data['published_utc']}")
                return None
            
            # Create article
            article = PolygonNewsArticle(
                polygon_id=article_data['polygon_id'],
                title=article_data['title'][:1000],  # Truncate if too long
                description=article_data.get('description'),
                article_url=article_data['article_url'],
                amp_url=article_data.get('amp_url'),
                image_url=article_data.get('image_url'),
                author=article_data.get('author'),
                published_utc=published_utc,
                publisher_name=article_data.get('publisher_name'),
                publisher_homepage_url=article_data.get('publisher_homepage_url'),
                publisher_logo_url=article_data.get('publisher_logo_url'),
                publisher_favicon_url=article_data.get('publisher_favicon_url'),
                keywords=article_data.get('keywords', []),
                quality_score=article_data.get('quality_score'),
                relevance_score=article_data.get('relevance_score')
            )
            
            self.session.add(article)
            self.session.flush()  # Get the ID
            
            # Add ticker associations
            tickers = article_data.get('tickers', [])
            for ticker in tickers:
                ticker_obj = PolygonNewsTicker(
                    article_id=article.id,
                    ticker=ticker.upper()
                )
                self.session.add(ticker_obj)
            
            # Add insights
            insights = article_data.get('insights', [])
            for insight_data in insights:
                insight = PolygonNewsInsight(
                    article_id=article.id,
                    sentiment=insight_data.get('sentiment'),
                    sentiment_reasoning=insight_data.get('sentiment_reasoning'),
                    insight_type=insight_data.get('insight_type', 'sentiment'),
                    insight_value=insight_data.get('insight_value'),
                    confidence_score=insight_data.get('confidence_score')
                )
                self.session.add(insight)
            
            return article.id
            
        except Exception as e:
            self.logger.error(f"Error creating article: {e}")
            return None
    
    def _update_existing_article(self, article: PolygonNewsArticle, 
                                article_data: Dict[str, Any]) -> bool:
        """Update existing article if data has changed"""
        updated = False
        
        try:
            # Update basic fields if they've changed
            if article.title != article_data['title'][:1000]:
                article.title = article_data['title'][:1000]
                updated = True
            
            if article.description != article_data.get('description'):
                article.description = article_data.get('description')
                updated = True
            
            # Update quality scores
            if article.quality_score != article_data.get('quality_score'):
                article.quality_score = article_data.get('quality_score')
                updated = True
            
            if article.relevance_score != article_data.get('relevance_score'):
                article.relevance_score = article_data.get('relevance_score')
                updated = True
            
            # Update keywords if different
            new_keywords = article_data.get('keywords', [])
            if article.keywords != new_keywords:
                article.keywords = new_keywords
                updated = True
            
            # Update tickers if different
            existing_tickers = set(ticker.ticker for ticker in article.tickers)
            new_tickers = set(ticker.upper() for ticker in article_data.get('tickers', []))
            
            if existing_tickers != new_tickers:
                # Remove old ticker associations
                for ticker_obj in article.tickers:
                    self.session.delete(ticker_obj)
                
                # Add new ticker associations
                for ticker in new_tickers:
                    ticker_obj = PolygonNewsTicker(
                        article_id=article.id,
                        ticker=ticker
                    )
                    self.session.add(ticker_obj)
                
                updated = True
            
            # Update insights if different
            new_insights = article_data.get('insights', [])
            if len(article.insights) != len(new_insights):
                # Remove old insights
                for insight in article.insights:
                    self.session.delete(insight)
                
                # Add new insights
                for insight_data in new_insights:
                    insight = PolygonNewsInsight(
                        article_id=article.id,
                        sentiment=insight_data.get('sentiment'),
                        sentiment_reasoning=insight_data.get('sentiment_reasoning'),
                        insight_type=insight_data.get('insight_type', 'sentiment'),
                        insight_value=insight_data.get('insight_value'),
                        confidence_score=insight_data.get('confidence_score')
                    )
                    self.session.add(insight)
                
                updated = True
            
            if updated:
                from datetime import timezone
                article.updated_at = datetime.now(timezone.utc)
            
            return updated
            
        except Exception as e:
            self.logger.error(f"Error updating article {article.polygon_id}: {e}")
            return False
    
    def get_latest_date_for_ticker(self, ticker: str) -> Optional[datetime]:
        """Get latest article date for a specific ticker"""
        return PolygonNewsArticle.get_latest_date_for_ticker(self.session, ticker)
    
    def get_latest_date_overall(self) -> Optional[datetime]:
        """Get latest article date across all articles"""
        return PolygonNewsArticle.get_latest_date_overall(self.session)
    
    def cleanup_old_articles(self, retention_days: Optional[int] = None) -> int:
        """Remove articles older than retention period"""
        if retention_days is None:
            retention_days = config.NEWS_RETENTION_YEARS * 365
        from datetime import timezone
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
        return PolygonNewsArticle.cleanup_old_articles(self.session, cutoff_date)
    
    def get_articles_for_ticker(self, ticker: str, 
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None,
                                limit: Optional[int] = None) -> List[PolygonNewsArticle]:
        """Get articles for a specific ticker"""
        articles = PolygonNewsTicker.get_articles_for_ticker(
            self.session, ticker, start_date, end_date
        )
        
        if limit:
            articles = articles[:limit]
        
        return articles
    
    def get_article_statistics(self, 
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get comprehensive statistics about stored articles"""
        try:
            query = self.session.query(PolygonNewsArticle)
            
            if start_date:
                query = query.filter(PolygonNewsArticle.published_utc >= start_date)
            
            if end_date:
                query = query.filter(PolygonNewsArticle.published_utc <= end_date)
            
            total_articles = query.count()
            
            # Get ticker counts
            ticker_counts = PolygonNewsTicker.get_ticker_counts(
                self.session, start_date, end_date
            )
            
            # Get sentiment distribution
            sentiment_dist = PolygonNewsInsight.get_sentiment_distribution(
                self.session, start_date, end_date
            )
            
            # Get publisher counts
            publisher_counts = self.session.query(
                PolygonNewsArticle.publisher_name,
                self.session.query(PolygonNewsArticle).filter(
                    PolygonNewsArticle.publisher_name == PolygonNewsArticle.publisher_name
                ).count().label('count')
            ).group_by(PolygonNewsArticle.publisher_name).limit(10).all()
            
            return {
                'total_articles': total_articles,
                'date_range': {
                    'start': start_date.isoformat() if start_date else None,
                    'end': end_date.isoformat() if end_date else None
                },
                'top_tickers': dict(ticker_counts[:10]),
                'sentiment_distribution': sentiment_dist,
                'top_publishers': dict(publisher_counts),
                'latest_article_date': self.get_latest_date_overall()
            }
        
        except Exception as e:
            self.logger.error(f"Error getting article statistics: {e}")
            return {
                'total_articles': 0,
                'date_range': {
                    'start': start_date.isoformat() if start_date else None,
                    'end': end_date.isoformat() if end_date else None
                },
                'top_tickers': {},
                'sentiment_distribution': {},
                'top_publishers': {},
                'latest_article_date': None,
                'error': str(e)
            }
    
    def _parse_datetime(self, date_str: str) -> Optional[datetime]:
        """Parse datetime string from Polygon API"""
        if not date_str:
            return None
        
        try:
            # Handle ISO format with Z suffix
            if date_str.endswith('Z'):
                date_str = date_str[:-1] + '+00:00'
            
            return datetime.fromisoformat(date_str)
        except ValueError as e:
            self.logger.warning(f"Failed to parse datetime: {date_str} - {e}")
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on database connection and data"""
        try:
            # Test basic query
            article_count = self.session.query(PolygonNewsArticle).count()
            latest_date = self.get_latest_date_overall()
            
            # Check for recent data (within last 7 days)
            from datetime import timezone
            recent_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            recent_count = self.session.query(PolygonNewsArticle).filter(
                PolygonNewsArticle.published_utc >= recent_cutoff
            ).count()
            
            return {
                'status': 'healthy',
                'total_articles': article_count,
                'latest_article_date': latest_date.isoformat() if latest_date else None,
                'recent_articles_7d': recent_count,
                'database_connection': 'ok'
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'database_connection': 'failed'
            }
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type):
        """Context manager exit with cleanup"""
        if exc_type:
            self.session.rollback()
        else:
            try:
                self.session.commit()
            except Exception as e:
                self.logger.error(f"Failed to commit session: {e}")
                self.session.rollback() 
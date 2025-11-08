"""
Database storage operations for Polygon news data
Handles upserts, batch processing, and data maintenance
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta, timezone

from src.data_collector.polygon_news.models import validate_article_data
from src.data_collector.config import config
from src.utils.logger import get_logger
from src.database.connection import fetch_one, fetch_all, execute, run_in_transaction

logger = get_logger(__name__, utility="data_collector")


class PolygonNewsStorage:
    """
    Database storage operations for Polygon news data using global DB pool
    """

    def __init__(self):
        """
        Initialize storage using module-level DB helpers
        """
        self.logger = get_logger(__name__, utility="data_collector")

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
                self.logger.warning(
                    f"Invalid article data: {article_data.get('polygon_id', 'unknown')}"
                )
                return None

            # Check if article already exists
            existing = fetch_one(
                "SELECT id, title, description, article_url, published_utc, publisher_name, keywords, quality_score, relevance_score FROM polygon_news_articles WHERE polygon_id = %s",
                params=(article_data["polygon_id"],),
            )

            if existing:
                updated = self._update_existing_article(existing, article_data)
                if updated:
                    self.logger.info(f"Updated existing article: {article_data['polygon_id']}")
                return existing.get("id")

            # Create new article
            article_id = self._create_new_article(article_data)
            if article_id:
                self.logger.info(f"Created new article: {article_data['polygon_id']}")

            return article_id

        except Exception as e:
            self.logger.error(
                f"Error storing article {article_data.get('polygon_id', 'unknown')}: {e}"
            )
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
            "total_processed": 0,
            "new_articles": 0,
            "updated_articles": 0,
            "skipped_articles": 0,
            "failed_articles": 0,
        }

        self.logger.info(f"Starting batch storage of {len(articles_data)} articles")

        for article_data in articles_data:
            stats["total_processed"] += 1

            try:
                # Check existence
                existing = fetch_one(
                    "SELECT id, title, description, article_url, published_utc, publisher_name, keywords, quality_score, relevance_score FROM polygon_news_articles WHERE polygon_id = %s",
                    params=(article_data["polygon_id"],),
                )

                if existing:
                    updated = self._update_existing_article(existing, article_data)
                    if updated:
                        stats["updated_articles"] += 1
                    else:
                        stats["skipped_articles"] += 1
                else:
                    aid = self._create_new_article(article_data)
                    if aid:
                        stats["new_articles"] += 1
                    else:
                        stats["failed_articles"] += 1

            except Exception as e:
                stats["failed_articles"] += 1
                self.logger.error(
                    f"Failed to process article {article_data.get('polygon_id', 'unknown')}: {e}"
                )

        self.logger.info(f"Batch storage completed: {stats}")
        return stats

    def _create_new_article(self, article_data: Dict[str, Any]) -> Optional[int]:
        """Create new article with all related data"""
        try:
            # Parse published date
            published_utc = self._parse_datetime(article_data["published_utc"])
            if not published_utc:
                self.logger.warning(f"Invalid published_utc: {article_data['published_utc']}")
                return None

            import json

            # Ensure keywords are a Python list when inserting into Postgres text[]
            keywords_val = article_data.get("keywords", [])
            if isinstance(keywords_val, str):
                try:
                    keywords_val = json.loads(keywords_val)
                except Exception:
                    # Fallback: wrap single string in list
                    keywords_val = [keywords_val]

            # Insert article and related rows inside a single transaction so
            # FK constraints don't fail due to cross-connection visibility.
            def _txn(conn, cur):
                insert_sql = (
                    "INSERT INTO polygon_news_articles (polygon_id, title, description, article_url, amp_url, image_url, author, published_utc, publisher_name, publisher_homepage_url, publisher_logo_url, publisher_favicon_url, keywords, quality_score, relevance_score) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id"
                )

                cur.execute(
                    insert_sql,
                    (
                        article_data["polygon_id"],
                        article_data["title"][:1000],
                        article_data.get("description"),
                        article_data["article_url"],
                        article_data.get("amp_url"),
                        article_data.get("image_url"),
                        article_data.get("author"),
                        published_utc,
                        article_data.get("publisher_name"),
                        article_data.get("publisher_homepage_url"),
                        article_data.get("publisher_logo_url"),
                        article_data.get("publisher_favicon_url"),
                        keywords_val,
                        article_data.get("quality_score"),
                        article_data.get("relevance_score"),
                    ),
                )

                row = cur.fetchone()
                if not row:
                    raise RuntimeError("Failed to insert article")

                # handle RealDictCursor or tuple cursor
                if isinstance(row, dict):
                    article_id = row.get("id")
                else:
                    article_id = row[0]

                # Insert tickers
                tickers = article_data.get("tickers", [])
                for ticker in tickers:
                    cur.execute(
                        "INSERT INTO polygon_news_tickers (article_id, ticker) VALUES (%s, %s)",
                        (article_id, ticker.upper()),
                    )

                # Insert insights
                insights = article_data.get("insights", [])
                for insight_data in insights:
                    cur.execute(
                        "INSERT INTO polygon_news_insights (article_id, sentiment, sentiment_reasoning, insight_type, insight_value, confidence_score) VALUES (%s, %s, %s, %s, %s, %s)",
                        (
                            article_id,
                            insight_data.get("sentiment"),
                            insight_data.get("sentiment_reasoning"),
                            insight_data.get("insight_type", "sentiment"),
                            insight_data.get("insight_value"),
                            insight_data.get("confidence_score"),
                        ),
                    )

                return article_id

            article_id = run_in_transaction(_txn)
            return article_id

        except Exception as e:
            self.logger.error(f"Error creating article: {e}")
            return None

    def _update_existing_article(
        self, article: Dict[str, Any], article_data: Dict[str, Any]
    ) -> bool:
        """Update existing article if data has changed using SQL"""
        try:
            article_id = article.get("id")
            fields = []
            params: List[Any] = []

            # Compare and prepare updates
            if article.get("title") != article_data["title"][:1000]:
                fields.append("title = %s")
                params.append(article_data["title"][:1000])

            if article.get("description") != article_data.get("description"):
                fields.append("description = %s")
                params.append(article_data.get("description"))

            if article.get("quality_score") != article_data.get("quality_score"):
                fields.append("quality_score = %s")
                params.append(article_data.get("quality_score"))

            if article.get("relevance_score") != article_data.get("relevance_score"):
                fields.append("relevance_score = %s")
                params.append(article_data.get("relevance_score"))

            new_keywords = article_data.get("keywords", [])
            if article.get("keywords") != new_keywords:
                fields.append("keywords = %s")
                params.append(new_keywords)

            updated = False
            if fields:
                params.append(article_id)
                sql = f"UPDATE polygon_news_articles SET {', '.join(fields)}, updated_at = now() WHERE id = %s"
                execute(sql, params=tuple(params), commit=True)
                updated = True

            # Replace tickers
            new_tickers = set(t.upper() for t in article_data.get("tickers", []))
            if new_tickers:
                # delete existing
                execute(
                    "DELETE FROM polygon_news_tickers WHERE article_id = %s",
                    params=(article_id,),
                    commit=True,
                )
                for t in new_tickers:
                    execute(
                        "INSERT INTO polygon_news_tickers (article_id, ticker) VALUES (%s, %s)",
                        params=(article_id, t),
                        commit=True,
                    )
                updated = True

            # Replace insights
            new_insights = article_data.get("insights", [])
            if new_insights:
                execute(
                    "DELETE FROM polygon_news_insights WHERE article_id = %s",
                    params=(article_id,),
                    commit=True,
                )
                for insight in new_insights:
                    execute(
                        "INSERT INTO polygon_news_insights (article_id, sentiment, sentiment_reasoning, insight_type, insight_value, confidence_score) VALUES (%s, %s, %s, %s, %s, %s)",
                        params=(
                            article_id,
                            insight.get("sentiment"),
                            insight.get("sentiment_reasoning"),
                            insight.get("insight_type", "sentiment"),
                            insight.get("insight_value"),
                            insight.get("confidence_score"),
                        ),
                        commit=True,
                    )
                updated = True

            if updated:
                return True
            return False

        except Exception as e:
            self.logger.error(f"Error updating article {article.get('polygon_id')}: {e}")
            return False

    def get_latest_date_for_ticker(self, ticker: str) -> Optional[datetime]:
        """Get latest article date for a specific ticker"""
        row = fetch_one(
            "SELECT MAX(a.published_utc) as latest FROM polygon_news_articles a JOIN polygon_news_tickers t ON a.id = t.article_id WHERE t.ticker = %s",
            params=(ticker.upper(),),
        )
        return row.get("latest") if row else None

    def get_latest_date_overall(self) -> Optional[datetime]:
        """Get latest article date across all articles"""
        row = fetch_one("SELECT MAX(published_utc) as latest FROM polygon_news_articles")
        return row.get("latest") if row else None

    def cleanup_old_articles(self, retention_days: Optional[int] = None) -> int:
        """Remove articles older than retention period"""
        if retention_days is None:
            retention_days = config.NEWS_RETENTION_YEARS * 365
        from datetime import timezone

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
        execute(
            "DELETE FROM polygon_news_articles WHERE published_utc < %s",
            params=(cutoff_date,),
            commit=True,
        )
        # execute helper does not return affected count; return 0 as unknown
        return 0

    def get_articles_for_ticker(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get articles for a specific ticker"""
        sql = (
            "SELECT a.* FROM polygon_news_articles a JOIN polygon_news_tickers t ON a.id = t.article_id "
            "WHERE t.ticker = %s"
        )
        params: List[Any] = [ticker.upper()]
        if start_date:
            sql += " AND a.published_utc >= %s"
            params.append(start_date)
        if end_date:
            sql += " AND a.published_utc <= %s"
            params.append(end_date)
        sql += " ORDER BY a.published_utc DESC"
        if limit:
            sql += " LIMIT %s"
            params.append(limit)

        rows = fetch_all(sql, params=tuple(params), dict_cursor=True)
        return rows

    def get_article_statistics(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get comprehensive statistics about stored articles"""
        try:
            # Total articles
            sql_total = "SELECT COUNT(*) as cnt FROM polygon_news_articles"
            total_row = fetch_one(sql_total)
            total_articles = total_row.get("cnt", 0) if total_row else 0
            logger.info(f"Total articles: {total_articles}")

            # Top tickers
            try:
                sql_tickers = "SELECT t.ticker, COUNT(*) as cnt FROM polygon_news_tickers t JOIN polygon_news_articles a ON a.id = t.article_id"
                sql_tickers += " GROUP BY t.ticker ORDER BY cnt DESC"
                ticker_rows = fetch_all(sql_tickers, dict_cursor=True)
                ticker_counts = (
                    [(r.get("ticker"), r.get("cnt")) for r in ticker_rows] if ticker_rows else []
                )
            except Exception as e:
                logger.error(f"Error getting top tickers: {e}")
                ticker_counts = []

            # Sentiment distribution
            try:
                sql_sent = "SELECT sentiment, COUNT(*) as cnt FROM polygon_news_insights i JOIN polygon_news_articles a ON a.id = i.article_id GROUP BY sentiment"
                sent_rows = fetch_all(sql_sent, dict_cursor=True)
                sentiment_dist = (
                    {r.get("sentiment"): r.get("cnt") for r in sent_rows} if sent_rows else {}
                )
            except Exception as e:
                logger.error(f"Error getting sentiment distribution: {e}")
                sentiment_dist = {}

            # Top publishers
            try:
                sql_pub = "SELECT publisher_name, COUNT(*) as cnt FROM polygon_news_articles GROUP BY publisher_name ORDER BY cnt DESC LIMIT 10"
                pub_rows = fetch_all(sql_pub, dict_cursor=True)
                top_publishers = (
                    {r.get("publisher_name"): r.get("cnt") for r in pub_rows} if pub_rows else {}
                )
                logger.info(f"Top publishers: {top_publishers}")
                return {
                    "total_articles": total_articles,
                    "date_range": {
                        "start": start_date.isoformat() if start_date else None,
                        "end": end_date.isoformat() if end_date else None,
                    },
                    "top_tickers": dict(ticker_counts[:10]),
                    "sentiment_distribution": sentiment_dist,
                    "top_publishers": top_publishers,
                    "latest_article_date": self.get_latest_date_overall(),
                }
            except Exception as e:
                logger.error(f"Error getting top publishers: {e}")
                pub_rows = []

        except Exception as e:
            self.logger.error(f"Error getting article statistics: {e}")
            return {
                "total_articles": 0,
                "date_range": {
                    "start": start_date.isoformat() if start_date else None,
                    "end": end_date.isoformat() if end_date else None,
                },
                "top_tickers": {},
                "sentiment_distribution": {},
                "top_publishers": {},
                "latest_article_date": None,
                "error": str(e),
            }

    def _parse_datetime(self, date_str: str) -> Optional[datetime]:
        """Parse datetime string from Polygon API"""
        if not date_str:
            return None

        # If already a datetime object, return as-is
        if isinstance(date_str, datetime):
            return date_str

        try:
            # Handle ISO format with Z suffix for UTC
            if isinstance(date_str, str) and date_str.endswith("Z"):
                date_str = date_str[:-1] + "+00:00"

            return datetime.fromisoformat(date_str)
        except Exception as e:
            # Last resort: try common formats
            try:
                # Try parsing date-only strings
                return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S%z")
            except Exception:
                self.logger.warning(f"Failed to parse datetime: {date_str} - {e}")
                return None

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on database connection and data"""
        try:
            # Test basic queries via pool
            row = fetch_one("SELECT COUNT(*) as cnt FROM polygon_news_articles")
            article_count = row.get("cnt", 0) if row else 0
            latest_date = self.get_latest_date_overall()

            recent_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            recent_row = fetch_one(
                "SELECT COUNT(*) as cnt FROM polygon_news_articles WHERE published_utc >= %s",
                params=(recent_cutoff,),
            )
            recent_count = recent_row.get("cnt", 0) if recent_row else 0

            return {
                "status": "healthy",
                "total_articles": article_count,
                "latest_article_date": latest_date.isoformat() if latest_date else None,
                "recent_articles_7d": recent_count,
                "database_connection": "ok",
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "database_connection": "failed"}

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup (no-op for pool-based storage)"""
        if exc_type:
            self.logger.warning("Exiting PolygonNewsStorage context with exception")
        # Nothing to commit/rollback when using pooled connections via helpers
        return False

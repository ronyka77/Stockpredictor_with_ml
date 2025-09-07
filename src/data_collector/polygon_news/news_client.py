"""
Polygon.io News API client extending the base PolygonDataClient
Specialized for news data collection with comprehensive error handling
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from src.data_collector.polygon_data.client import PolygonDataClient, PolygonAPIError
from src.data_collector.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__, utility="data_collector")


class PolygonNewsClient(PolygonDataClient):
    """
    Specialized client for Polygon.io News API

    Extends the base PolygonDataClient with news-specific functionality
    including ticker-based queries, date filtering, and sentiment analysis.
    """

    def __init__(
        self, api_key: Optional[str] = None, requests_per_minute: Optional[int] = None
    ):
        """
        Initialize the Polygon News client

        Args:
            api_key: Polygon.io API key (defaults to config)
            requests_per_minute: Rate limit for API requests (defaults to config)
        """
        if api_key is None:
            api_key = config.API_KEY
        if requests_per_minute is None:
            requests_per_minute = config.REQUESTS_PER_MINUTE

        super().__init__(api_key, requests_per_minute)
        logger.info(
            f"Polygon News client initialized with {requests_per_minute} requests/minute"
        )

    def get_news_for_ticker(
        self,
        ticker: str,
        published_utc_gte: Optional[datetime] = None,
        published_utc_lte: Optional[datetime] = None,
        order: str = "desc",
        limit: int = 1000,
        sort: str = "published_utc",
    ) -> List[Dict[str, Any]]:
        """
        Get news articles for a specific ticker

        Args:
            ticker: Stock ticker symbol (case-sensitive)
            published_utc_gte: Return results published on or after this date
            published_utc_lte: Return results published on or before this date
            order: Order results (asc/desc)
            limit: Limit number of results (max 1000)
            sort: Sort field (published_utc)

        Returns:
            List of news articles with full metadata

        Raises:
            PolygonAPIError: For API-related errors
        """
        endpoint = "/v2/reference/news"

        # Build query parameters
        params = {
            "ticker": ticker,
            "order": order,
            "limit": min(limit, 1000),  # Polygon max limit
            "sort": sort,
        }

        # Add date filters if provided
        if published_utc_gte:
            params["published_utc.gte"] = published_utc_gte.strftime("%Y-%m-%d")

        if published_utc_lte:
            params["published_utc.lte"] = published_utc_lte.strftime("%Y-%m-%d")

        logger.info(f"Fetching news for ticker {ticker} with {len(params)} parameters")

        try:
            # Use pagination to get all results
            all_articles = self._fetch_paginated_data(endpoint, params)

            logger.info(
                f"Successfully fetched {len(all_articles)} articles for {ticker}"
            )
            return all_articles

        except PolygonAPIError as e:
            logger.error(f"Failed to fetch news for ticker {ticker}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching news for {ticker}: {e}")
            raise PolygonAPIError(f"Unexpected error: {str(e)}")

    def get_news_for_multiple_tickers(
        self,
        tickers: List[str],
        published_utc_gte: Optional[datetime] = None,
        published_utc_lte: Optional[datetime] = None,
        order: str = "desc",
        limit_per_ticker: int = 100,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get news articles for multiple tickers

        Args:
            tickers: List of stock ticker symbols
            published_utc_gte: Return results published on or after this date
            published_utc_lte: Return results published on or before this date
            order: Order results (asc/desc)
            limit_per_ticker: Limit number of results per ticker

        Returns:
            Dictionary mapping ticker to list of articles
        """
        results = {}
        failed_tickers = []

        logger.info(f"Fetching news for {len(tickers)} tickers")

        for ticker in tickers:
            try:
                articles = self.get_news_for_ticker(
                    ticker=ticker,
                    published_utc_gte=published_utc_gte,
                    published_utc_lte=published_utc_lte,
                    order=order,
                    limit=limit_per_ticker,
                )
                results[ticker] = articles

            except PolygonAPIError as e:
                logger.warning(f"Failed to fetch news for ticker {ticker}: {e}")
                failed_tickers.append(ticker)
                results[ticker] = []

        if failed_tickers:
            logger.warning(
                f"Failed to fetch news for {len(failed_tickers)} tickers: {failed_tickers}"
            )

        total_articles = sum(len(articles) for articles in results.values())
        logger.info(
            f"Successfully fetched {total_articles} total articles for {len(tickers)} tickers"
        )

        return results

    def get_recent_market_news(
        self,
        days_back: int = 7,
        major_tickers: Optional[List[str]] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Get recent market news across major tickers

        Args:
            days_back: Number of days to look back
            major_tickers: List of major tickers to focus on
            limit: Total limit of articles to return

        Returns:
            List of recent news articles
        """
        if major_tickers is None:
            # Default major tickers for market news
            major_tickers = [
                "AAPL",
                "GOOGL",
                "MSFT",
                "AMZN",
                "TSLA",
                "META",
                "NVDA",
                "SPY",
                "QQQ",
            ]

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        logger.info(
            f"Fetching recent market news for {days_back} days across {len(major_tickers)} tickers"
        )

        all_articles = []
        articles_per_ticker = max(1, limit // len(major_tickers))

        ticker_results = self.get_news_for_multiple_tickers(
            tickers=major_tickers,
            published_utc_gte=start_date,
            published_utc_lte=end_date,
            limit_per_ticker=articles_per_ticker,
        )

        # Combine and deduplicate articles
        seen_ids = set()
        for ticker, articles in ticker_results.items():
            for article in articles:
                article_id = article.get("id")
                if article_id and article_id not in seen_ids:
                    seen_ids.add(article_id)
                    all_articles.append(article)

        # Sort by publication date (most recent first)
        all_articles.sort(key=lambda x: x.get("published_utc", ""), reverse=True)

        # Limit total results
        all_articles = all_articles[:limit]

        logger.info(f"Retrieved {len(all_articles)} unique recent market news articles")
        return all_articles

    def get_news_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        tickers: Optional[List[str]] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Get news articles within a specific date range

        Args:
            start_date: Start date for news search
            end_date: End date for news search
            tickers: Optional list of specific tickers to focus on
            limit: Maximum number of articles to return

        Returns:
            List of news articles within the date range
        """
        logger.info(f"Fetching news from {start_date.date()} to {end_date.date()}")

        if tickers:
            # Get news for specific tickers
            ticker_results = self.get_news_for_multiple_tickers(
                tickers=tickers,
                published_utc_gte=start_date,
                published_utc_lte=end_date,
                limit_per_ticker=limit // len(tickers) if len(tickers) > 0 else limit,
            )

            # Combine results
            all_articles = []
            seen_ids = set()

            for ticker, articles in ticker_results.items():
                for article in articles:
                    article_id = article.get("id")
                    if article_id and article_id not in seen_ids:
                        seen_ids.add(article_id)
                        all_articles.append(article)
        else:
            # Get general market news using major tickers
            all_articles = self.get_recent_market_news(
                days_back=(end_date - start_date).days, limit=limit
            )

        # Filter by exact date range
        filtered_articles = []
        for article in all_articles:
            published_str = article.get("published_utc", "")
            if published_str:
                try:
                    published_date = datetime.fromisoformat(
                        published_str.replace("Z", "+00:00")
                    )
                    if start_date <= published_date <= end_date:
                        filtered_articles.append(article)
                except ValueError:
                    logger.warning(f"Invalid date format in article: {published_str}")
                    continue

        logger.info(f"Retrieved {len(filtered_articles)} articles within date range")
        return filtered_articles

    def validate_news_response(self, article: Dict[str, Any]) -> bool:
        """
        Validate that a news article response has required fields

        Args:
            article: News article data from Polygon API

        Returns:
            True if article has required fields, False otherwise
        """
        required_fields = ["id", "title", "article_url", "published_utc"]

        for field in required_fields:
            if not article.get(field):
                logger.warning(f"Article missing required field: {field}")
                return False

        return True

    def extract_article_metadata(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and normalize metadata from Polygon news article

        Args:
            article: Raw article data from Polygon API

        Returns:
            Normalized article metadata
        """
        # Extract publisher information
        publisher = article.get("publisher", {})

        # Extract insights (sentiment analysis)
        insights = article.get("insights", [])
        sentiment_insight = None
        for insight in insights:
            if insight.get("sentiment"):
                sentiment_insight = insight
                break

        metadata = {
            "polygon_id": article.get("id"),
            "title": article.get("title", ""),
            "description": article.get("description", ""),
            "article_url": article.get("article_url", ""),
            "amp_url": article.get("amp_url"),
            "image_url": article.get("image_url"),
            "author": article.get("author", ""),
            "published_utc": article.get("published_utc"),
            "keywords": article.get("keywords", []),
            "tickers": article.get("tickers", []),
            # Publisher information
            "publisher_name": publisher.get("name", ""),
            "publisher_homepage_url": publisher.get("homepage_url"),
            "publisher_logo_url": publisher.get("logo_url"),
            "publisher_favicon_url": publisher.get("favicon_url"),
            # Sentiment analysis
            "sentiment": sentiment_insight.get("sentiment")
            if sentiment_insight
            else None,
            "sentiment_reasoning": sentiment_insight.get("sentiment_reasoning")
            if sentiment_insight
            else None,
            "insights": insights,
        }

        return metadata

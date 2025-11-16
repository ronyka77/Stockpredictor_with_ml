"""
News content processor for Polygon news data
Handles content cleaning and metadata extraction
"""

from typing import Dict, List, Any
import re

from src.utils.core.logger import get_logger

logger = get_logger(__name__, utility="data_collector")


class NewsProcessor:
    """
    Processor for Polygon news content
    Since Polygon provides pre-processed data, this focuses on normalization
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__, utility="data_collector")

    def process_article(self, article_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process article metadata from Polygon API

        Args:
            article_metadata: Raw metadata from Polygon news client

        Returns:
            Processed article data ready for storage
        """
        try:
            # Extract basic fields
            processed = {
                "polygon_id": article_metadata.get("polygon_id"),
                "title": self._clean_text(article_metadata.get("title", "")),
                "description": self._clean_text(article_metadata.get("description", "")),
                "article_url": article_metadata.get("article_url", ""),
                "amp_url": article_metadata.get("amp_url"),
                "image_url": article_metadata.get("image_url"),
                "author": self._clean_text(article_metadata.get("author", "")),
                "published_utc": article_metadata.get("published_utc"),
                "keywords": article_metadata.get("keywords", []),
                "tickers": article_metadata.get("tickers", []),
                # Publisher information
                "publisher_name": article_metadata.get("publisher_name", ""),
                "publisher_homepage_url": article_metadata.get("publisher_homepage_url"),
                "publisher_logo_url": article_metadata.get("publisher_logo_url"),
                "publisher_favicon_url": article_metadata.get("publisher_favicon_url"),
                # Process insights
                "insights": self._process_insights(article_metadata.get("insights", [])),
            }

            return processed

        except Exception as e:
            self.logger.error(f"Error processing article: {e}")
            raise

    def _clean_text(self, text: str) -> str:
        """Clean text content"""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Remove common artifacts
        text = re.sub(r"\(Reuters\)|\(AP\)|\(Bloomberg\)", "", text, flags=re.IGNORECASE)

        return text

    def _process_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process insights from Polygon API"""
        processed_insights = []

        for insight in insights:
            processed_insight = {
                "sentiment": insight.get("sentiment"),
                "sentiment_reasoning": insight.get("sentiment_reasoning"),
                "insight_type": "sentiment",
                "insight_value": insight.get("sentiment"),
                "confidence_score": None,  # Polygon doesn't provide confidence scores
            }
            processed_insights.append(processed_insight)

        return processed_insights

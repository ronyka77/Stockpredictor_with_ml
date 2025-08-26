"""
News content validator for quality assessment
"""

from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta, timezone

from src.utils.logger import get_logger

logger = get_logger(__name__, utility="data_collector")


class NewsValidator:
    """
    Validator for news article quality and completeness
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__, utility="data_collector")
        self.min_title_length = 10
        self.max_title_length = 1000
        self.required_fields = ['polygon_id', 'title', 'article_url', 'published_utc']
    
    def validate_article(self, article: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
        """
        Validate article quality
        
        Args:
            article: Processed article data
            
        Returns:
            Tuple of (is_valid, quality_score, issues)
        """
        issues = []
        quality_score = 1.0
        
        # Check required fields
        for field in self.required_fields:
            if not article.get(field):
                issues.append(f"Missing required field: {field}")
                quality_score -= 0.3
        
        # Validate title
        title = article.get('title', '')
        if title:
            if len(title) < self.min_title_length:
                issues.append("Title too short")
                quality_score -= 0.2
            elif len(title) > self.max_title_length:
                issues.append("Title too long")
                quality_score -= 0.1
        
        # Check for tickers
        tickers = article.get('tickers', [])
        if not tickers:
            issues.append("No associated tickers")
            quality_score -= 0.2
        
        # Check publication date
        published_utc = article.get('published_utc')
        if published_utc:
            try:
                if isinstance(published_utc, str):
                    pub_date = datetime.fromisoformat(published_utc.replace('Z', '+00:00'))
                else:
                    pub_date = published_utc
                
                # Check if too old (more than 2 years)
                now = datetime.now(timezone.utc)
                if pub_date < now - timedelta(days=730):
                    issues.append("Article too old")
                    quality_score -= 0.1
                
                # Check if future date
                if pub_date > now:
                    issues.append("Future publication date")
                    quality_score -= 0.2
                    
            except Exception:
                issues.append("Invalid publication date")
                quality_score -= 0.2
        
        # Ensure score doesn't go below 0
        quality_score = max(0.0, quality_score)
        
        # Article is valid if score > 0.5 and no critical issues
        critical_issues = [issue for issue in issues if "Missing required field" in issue]
        is_valid = quality_score > 0.5 and len(critical_issues) == 0
        
        return is_valid, quality_score, issues 
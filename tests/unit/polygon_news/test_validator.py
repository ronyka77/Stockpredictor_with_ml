import pytest
from datetime import datetime, timezone, timedelta

from src.data_collector.polygon_news.validator import NewsValidator


@pytest.mark.unit
def test_validate_article_returns_true_for_complete_article(processed_article_expected):
    v = NewsValidator()
    is_valid, score, issues = v.validate_article(processed_article_expected)
    assert is_valid is True
    assert score > 0.5
    assert issues == []


@pytest.mark.unit
def test_validate_article_detects_missing_required_fields(processed_article_expected):
    v = NewsValidator()
    article = processed_article_expected.copy()
    article.pop("polygon_id", None)
    is_valid, score, issues = v.validate_article(article)
    assert is_valid is False
    assert any("Missing required field" in s for s in issues)


@pytest.mark.unit
def test_validate_article_scores_length_and_source(processed_article_expected):
    v = NewsValidator()
    article = processed_article_expected.copy()
    article["title"] = "Short"
    is_valid, score, issues = v.validate_article(article)
    # Short title should be flagged and reduce quality score; it may still be valid
    assert "Title too short" in issues
    assert score < 1.0


@pytest.mark.unit
def test_validate_article_handles_old_and_future_dates(processed_article_expected):
    v = NewsValidator()
    article = processed_article_expected.copy()
    # Old date
    article["published_utc"] = (datetime.now(timezone.utc) - timedelta(days=800)).isoformat()
    is_valid_old, score_old, issues_old = v.validate_article(article)
    assert "Article too old" in issues_old or score_old < 1.0

    # Future date
    article["published_utc"] = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
    is_valid_future, score_future, issues_future = v.validate_article(article)
    assert "Future publication date" in issues_future or score_future < 1.0



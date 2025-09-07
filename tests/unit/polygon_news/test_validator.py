import pytest
from datetime import datetime, timezone, timedelta

from src.data_collector.polygon_news.validator import NewsValidator


@pytest.mark.unit
def test_validate_article_returns_true_for_complete_article(processed_article_expected):
    v = NewsValidator()
    is_valid, score, issues = v.validate_article(processed_article_expected)
    if is_valid is not True:
        raise AssertionError("Validator unexpectedly marked complete article as invalid")
    if not (score > 0.5):
        raise AssertionError("Validator returned unexpectedly low score for complete article")
    if issues != []:
        raise AssertionError("Validator reported issues for a complete article")


@pytest.mark.unit
def test_validate_article_detects_missing_required_fields(processed_article_expected):
    v = NewsValidator()
    article = processed_article_expected.copy()
    article.pop("polygon_id", None)
    is_valid, score, issues = v.validate_article(article)
    if is_valid is not False:
        raise AssertionError("Validator failed to detect missing required fields")
    if not any("Missing required field" in s for s in issues):
        raise AssertionError("Missing required field not reported in issues")


@pytest.mark.unit
def test_validate_article_scores_length_and_source(processed_article_expected):
    v = NewsValidator()
    article = processed_article_expected.copy()
    article["title"] = "Short"
    is_valid, score, issues = v.validate_article(article)
    # Short title should be flagged and reduce quality score; it may still be valid
    if "Title too short" not in issues:
        raise AssertionError("Short title not flagged as expected")
    if score >= 1.0:
        raise AssertionError("Score unexpectedly equals 1.0 for short title")


@pytest.mark.unit
def test_validate_article_handles_old_and_future_dates(processed_article_expected):
    v = NewsValidator()
    article = processed_article_expected.copy()
    # Old date
    article["published_utc"] = (
        datetime.now(timezone.utc) - timedelta(days=800)
    ).isoformat()
    is_valid_old, score_old, issues_old = v.validate_article(article)
    if not ("Article too old" in issues_old or score_old < 1.0):
        raise AssertionError("Old article not penalized as expected")

    # Future date
    article["published_utc"] = (
        datetime.now(timezone.utc) + timedelta(days=10)
    ).isoformat()
    is_valid_future, score_future, issues_future = v.validate_article(article)
    if not ("Future publication date" in issues_future or score_future < 1.0):
        raise AssertionError("Future-dated article not flagged as expected")

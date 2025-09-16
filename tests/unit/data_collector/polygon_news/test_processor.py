import pytest

from src.data_collector.polygon_news.processor import NewsProcessor


@pytest.mark.unit
def test_process_article_full(sample_raw_article_full, processed_article_expected):
    """Process full raw article and normalize fields as expected"""
    proc = NewsProcessor()
    processed = proc.process_article(sample_raw_article_full)

    # Title should have artifacts removed and trailing whitespace trimmed
    assert processed["title"].strip() == processed_article_expected["title"].strip()
    assert processed["description"] == processed_article_expected["description"]
    assert processed["publisher_name"] == processed_article_expected["publisher_name"]
    # Insights normalized
    assert processed["insights"][0]["insight_type"] == "sentiment"


@pytest.mark.unit
def test_process_article_missing_fields(sample_raw_article_missing):
    """Handle missing fields by filling defaults (empty strings, empty lists)"""
    proc = NewsProcessor()
    processed = proc.process_article(sample_raw_article_missing)

    # Missing title/description should become empty strings
    assert processed["title"] == ""
    assert processed["description"] == ""
    # tickers empty list
    assert processed["tickers"] == []


@pytest.mark.unit
def test_process_article_malformed_types_raises(sample_raw_article_malformed):
    """Raise when article fields have malformed types that cannot be processed"""
    proc = NewsProcessor()
    with pytest.raises(Exception):
        proc.process_article(sample_raw_article_malformed)

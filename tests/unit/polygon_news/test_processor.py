import pytest

from src.data_collector.polygon_news.processor import NewsProcessor


@pytest.mark.unit
def test_process_article_full(sample_raw_article_full, processed_article_expected):
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
    proc = NewsProcessor()
    processed = proc.process_article(sample_raw_article_missing)

    # Missing title/description should become empty strings
    assert processed["title"] == ""
    assert processed["description"] == ""
    # tickers empty list
    assert processed["tickers"] == []


@pytest.mark.unit
def test_process_article_malformed_types_raises(sample_raw_article_malformed):
    proc = NewsProcessor()
    with pytest.raises(Exception):
        proc.process_article(sample_raw_article_malformed)



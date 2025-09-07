import pytest

from src.data_collector.polygon_news.processor import NewsProcessor


@pytest.mark.unit
def test_process_article_full(sample_raw_article_full, processed_article_expected):
    proc = NewsProcessor()
    processed = proc.process_article(sample_raw_article_full)

    # Title should have artifacts removed and trailing whitespace trimmed
    if processed["title"].strip() != processed_article_expected["title"].strip():
        raise AssertionError("Processed title does not match expected after cleanup")
    if processed["description"] != processed_article_expected["description"]:
        raise AssertionError("Processed description mismatch")
    if processed["publisher_name"] != processed_article_expected["publisher_name"]:
        raise AssertionError("Processed publisher_name mismatch")
    # Insights normalized
    if processed["insights"][0]["insight_type"] != "sentiment":
        raise AssertionError("Insight type normalization failed")


@pytest.mark.unit
def test_process_article_missing_fields(sample_raw_article_missing):
    proc = NewsProcessor()
    processed = proc.process_article(sample_raw_article_missing)

    # Missing title/description should become empty strings
    if processed["title"] != "":
        raise AssertionError("Missing title should be converted to empty string")
    if processed["description"] != "":
        raise AssertionError("Missing description should be converted to empty string")
    # tickers empty list
    if processed["tickers"] != []:
        raise AssertionError("Missing tickers should become an empty list")


@pytest.mark.unit
def test_process_article_malformed_types_raises(sample_raw_article_malformed):
    proc = NewsProcessor()
    with pytest.raises(Exception):
        proc.process_article(sample_raw_article_malformed)

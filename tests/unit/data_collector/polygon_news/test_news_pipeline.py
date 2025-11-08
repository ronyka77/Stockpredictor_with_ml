import pytest
from datetime import datetime, timezone, timedelta

from src.data_collector.polygon_news.news_pipeline import PolygonNewsCollector
from tests.unit.data_collector.polygon_news.helpers import make_raw_article


@pytest.mark.unit
def test__collect_ticker_news_no_raw_articles(mocker):
    """When no raw articles are returned, _collect_ticker_news reports zero fetched"""
    collector = PolygonNewsCollector()

    # mocker client to return no articles
    mocker.patch.object(collector.news_client, "get_news_for_ticker", lambda **kw: [])

    stats = collector._collect_ticker_news(
        "ACME", datetime.now(timezone.utc), datetime.now(timezone.utc), 50
    )
    assert stats.get("api_calls") == 1
    assert stats.get("articles_fetched") == 0


@pytest.mark.unit
def test__collect_ticker_news_happy_flow(mocker, processed_article_expected):
    """Happy flow collects, processes and stores articles and returns counts"""
    collector = PolygonNewsCollector()
    raw_articles = [make_raw_article({"id": "a1"})]
    mocker.patch.object(collector.news_client, "get_news_for_ticker", lambda **kw: raw_articles)
    mocker.patch.object(
        collector.news_client, "extract_article_metadata", lambda raw: processed_article_expected
    )
    mocker.patch.object(
        collector.processor, "process_article", lambda meta: processed_article_expected
    )
    mocker.patch.object(collector.validator, "validate_article", lambda art: (True, 0.9, []))
    mocker.patch.object(
        collector.storage,
        "store_articles_batch",
        lambda arts: {"new_articles": 1, "updated_articles": 0, "skipped_articles": 0},
    )

    stats = collector._collect_ticker_news(
        "ACME", datetime.now(timezone.utc), datetime.now(timezone.utc), 50
    )
    assert stats.get("api_calls") == 1
    assert stats.get("articles_fetched") == 1
    assert stats.get("articles_stored") == 1


@pytest.mark.unit
def test__collect_ticker_news_skips_invalid(mocker, processed_article_expected):
    """Skip invalid processed articles while storing valid ones in a batch"""
    collector = PolygonNewsCollector()
    raw_articles = [make_raw_article({"id": "a1"}), make_raw_article({"id": "a2"})]
    mocker.patch.object(collector.news_client, "get_news_for_ticker", lambda **kw: raw_articles)
    mocker.patch.object(
        collector.news_client, "extract_article_metadata", lambda raw: processed_article_expected
    )

    # First article valid, second invalid
    def validator_seq(article):
        if article.get("polygon_id") == processed_article_expected.get("polygon_id"):
            return (True, 0.8, [])
        return (False, 0.0, ["No associated tickers"])

    mocker.patch.object(collector.validator, "validate_article", lambda art: (True, 0.8, []))
    # simulate one invalid by changing processed list mid-loop via mocker of processor
    calls = {"n": 0}

    def proc(meta):
        calls["n"] += 1
        if calls["n"] == 1:
            return processed_article_expected
        else:
            # return an article missing tickers
            a = processed_article_expected.copy()
            a["tickers"] = []
            return a

    mocker.patch.object(collector.processor, "process_article", proc)
    mocker.patch.object(
        collector.storage,
        "store_articles_batch",
        lambda arts: {"new_articles": 1, "updated_articles": 0, "skipped_articles": 1},
    )

    stats = collector._collect_ticker_news(
        "ACME", datetime.now(timezone.utc), datetime.now(timezone.utc), 50, limit=2
    )

    assert stats.get("articles_fetched") == 2
    assert stats.get("articles_stored") == 1
    assert stats.get("articles_skipped", 0) >= 1


@pytest.mark.unit
def test__collect_ticker_news_processing_exception_per_article(mocker):
    """Per-article processing exceptions are isolated and do not stop batch"""
    collector = PolygonNewsCollector()
    raw_articles = [make_raw_article({"id": "a1"}), make_raw_article({"id": "a2"})]
    mocker.patch.object(collector.news_client, "get_news_for_ticker", lambda **kw: raw_articles)

    # first article raises processing error
    def proc(meta):
        if meta.get("id") == "a1":
            raise RuntimeError("processing failed")
        return {
            "polygon_id": "art-2",
            "title": "T",
            "article_url": "u",
            "published_utc": datetime.now(timezone.utc).isoformat(),
            "tickers": ["X"],
        }

    mocker.patch.object(collector.news_client, "extract_article_metadata", lambda raw: raw)
    mocker.patch.object(collector.processor, "process_article", proc)
    mocker.patch.object(collector.validator, "validate_article", lambda art: (True, 0.8, []))
    mocker.patch.object(
        collector.storage,
        "store_articles_batch",
        lambda arts: {"new_articles": 1, "updated_articles": 0, "skipped_articles": 0},
    )

    stats = collector._collect_ticker_news(
        "ACME", datetime.now(timezone.utc), datetime.now(timezone.utc), 50, limit=2
    )

    assert stats.get("articles_fetched") == 2
    assert stats.get("articles_stored") == 1
    assert stats.get("articles_skipped", 0) >= 1


@pytest.mark.unit
def test_collect_targeted_news_happy_flow(mocker):
    """Collect targeted news using ticker info and return aggregated stats"""
    collector = PolygonNewsCollector()

    # validate_ticker_list -> one valid
    mocker.patch.object(
        collector.ticker_integration, "validate_ticker_list", lambda tickers: (tickers, [])
    )
    mocker.patch.object(
        collector.ticker_integration, "get_ticker_info", lambda t: {"priority_score": 50}
    )
    mocker.patch.object(
        collector,
        "_collect_ticker_news",
        lambda ticker, s, e, p, limit=100: {
            "api_calls": 1,
            "articles_fetched": 2,
            "articles_stored": 1,
            "articles_updated": 0,
            "articles_skipped": 0,
        },
    )

    start = datetime.now(timezone.utc) - timedelta(days=2)
    end = datetime.now(timezone.utc)
    stats = collector.collect_targeted_news(["ACME"], start, end, limit_per_ticker=10)

    assert stats.get("total_api_calls") == 1
    assert stats.get("total_articles_fetched") == 2


@pytest.mark.unit
def test_get_collection_status_handles_healthy_and_error(mocker):
    """Return collection status combining storage health and article stats; handle errors"""
    collector = PolygonNewsCollector()

    mocker.patch.object(collector.storage, "health_check", lambda: {"status": "healthy"})
    mocker.patch.object(
        collector.storage, "get_latest_date_overall", lambda: datetime.now(timezone.utc)
    )
    mocker.patch.object(
        collector.storage, "get_article_statistics", lambda start_date=None: {"total_articles": 5}
    )

    status = collector.get_collection_status()
    assert status.get("status") == "healthy"
    assert status.get("recent_statistics", {}).get("total_articles") == 5

    # Now simulate error
    def raise_err():
        raise RuntimeError("db fail")

    mocker.patch.object(collector.storage, "health_check", raise_err)
    status2 = collector.get_collection_status()
    assert status2.get("status") == "error"
    assert "error" in status2


@pytest.mark.unit
def test_context_manager_enter_exit():
    """Context manager yields a PolygonNewsCollector instance and exits cleanly"""
    with PolygonNewsCollector() as c:
        assert isinstance(c, PolygonNewsCollector)

import pytest
from datetime import datetime, timezone, timedelta

from src.data_collector.polygon_news.news_pipeline import PolygonNewsCollector


@pytest.mark.unit
def test__collect_ticker_news_no_raw_articles(db_session, monkeypatch):
    collector = PolygonNewsCollector(db_session=db_session)

    # Monkeypatch client to return no articles
    monkeypatch.setattr(collector.news_client, "get_news_for_ticker", lambda **kw: [])

    stats = collector._collect_ticker_news("ACME", datetime.now(timezone.utc), datetime.now(timezone.utc), 50)

    assert stats["api_calls"] == 1
    assert stats["articles_fetched"] == 0


@pytest.mark.unit
def test__collect_ticker_news_happy_flow(db_session, monkeypatch, processed_article_expected):
    collector = PolygonNewsCollector(db_session=db_session)

    raw_articles = [{"id": "a1"}]
    monkeypatch.setattr(collector.news_client, "get_news_for_ticker", lambda **kw: raw_articles)
    monkeypatch.setattr(collector.news_client, "extract_article_metadata", lambda raw: processed_article_expected)
    monkeypatch.setattr(collector.processor, "process_article", lambda meta: processed_article_expected)
    monkeypatch.setattr(collector.validator, "validate_article", lambda art: (True, 0.9, []))
    monkeypatch.setattr(collector.storage, "store_articles_batch", lambda arts: {"new_articles": 1, "updated_articles": 0, "skipped_articles": 0})

    stats = collector._collect_ticker_news("ACME", datetime.now(timezone.utc), datetime.now(timezone.utc), 50)

    assert stats["api_calls"] == 1
    assert stats["articles_fetched"] == 1
    assert stats["articles_stored"] == 1


@pytest.mark.unit
def test__collect_ticker_news_skips_invalid(db_session, monkeypatch, processed_article_expected):
    collector = PolygonNewsCollector(db_session=db_session)

    raw_articles = [{"id": "a1"}, {"id": "a2"}]
    monkeypatch.setattr(collector.news_client, "get_news_for_ticker", lambda **kw: raw_articles)
    monkeypatch.setattr(collector.news_client, "extract_article_metadata", lambda raw: processed_article_expected)
    # First article valid, second invalid
    def validator_seq(article):
        if article.get("polygon_id") == processed_article_expected.get("polygon_id"):
            return (True, 0.8, [])
        return (False, 0.0, ["No associated tickers"])

    monkeypatch.setattr(collector.validator, "validate_article", lambda art: (True, 0.8, []))
    # simulate one invalid by changing processed list mid-loop via monkeypatch of processor
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

    monkeypatch.setattr(collector.processor, "process_article", proc)
    monkeypatch.setattr(collector.storage, "store_articles_batch", lambda arts: {"new_articles": 1, "updated_articles": 0, "skipped_articles": 1})

    stats = collector._collect_ticker_news("ACME", datetime.now(timezone.utc), datetime.now(timezone.utc), 50, limit=2)

    assert stats["articles_fetched"] == 2
    assert stats["articles_stored"] == 1
    assert stats["articles_skipped"] >= 1


@pytest.mark.unit
def test__collect_ticker_news_processing_exception_per_article(db_session, monkeypatch):
    collector = PolygonNewsCollector(db_session=db_session)

    raw_articles = [{"id": "a1"}, {"id": "a2"}]
    monkeypatch.setattr(collector.news_client, "get_news_for_ticker", lambda **kw: raw_articles)
    # first article raises processing error
    def proc(meta):
        if meta.get("id") == "a1":
            raise RuntimeError("processing failed")
        return {"polygon_id": "art-2", "title": "T", "article_url": "u", "published_utc": datetime.now(timezone.utc).isoformat(), "tickers": ["X"]}

    monkeypatch.setattr(collector.news_client, "extract_article_metadata", lambda raw: raw)
    monkeypatch.setattr(collector.processor, "process_article", proc)
    monkeypatch.setattr(collector.validator, "validate_article", lambda art: (True, 0.8, []))
    monkeypatch.setattr(collector.storage, "store_articles_batch", lambda arts: {"new_articles": 1, "updated_articles": 0, "skipped_articles": 0})

    stats = collector._collect_ticker_news("ACME", datetime.now(timezone.utc), datetime.now(timezone.utc), 50, limit=2)

    assert stats["articles_fetched"] == 2
    assert stats["articles_stored"] == 1
    assert stats["articles_skipped"] >= 1


@pytest.mark.unit
def test_collect_targeted_news_happy_flow(db_session, monkeypatch):
    collector = PolygonNewsCollector(db_session=db_session)

    # validate_ticker_list -> one valid
    monkeypatch.setattr(collector.ticker_integration, "validate_ticker_list", lambda tickers: (tickers, []))
    monkeypatch.setattr(collector.ticker_integration, "get_ticker_info", lambda t: {"priority_score": 50})
    monkeypatch.setattr(collector, "_collect_ticker_news", lambda ticker, s, e, p, limit=100: {"api_calls": 1, "articles_fetched": 2, "articles_stored": 1, "articles_updated": 0, "articles_skipped": 0})

    start = datetime.now(timezone.utc) - timedelta(days=2)
    end = datetime.now(timezone.utc)
    stats = collector.collect_targeted_news(["ACME"], start, end, limit_per_ticker=10)

    assert stats["total_api_calls"] == 1
    assert stats["total_articles_fetched"] == 2


@pytest.mark.unit
def test_get_collection_status_handles_healthy_and_error(db_session, monkeypatch):
    collector = PolygonNewsCollector(db_session=db_session)

    monkeypatch.setattr(collector.storage, "health_check", lambda: {"status": "healthy"})
    monkeypatch.setattr(collector.storage, "get_latest_date_overall", lambda: datetime.now(timezone.utc))
    monkeypatch.setattr(collector.storage, "get_article_statistics", lambda start_date=None: {"total_articles": 5})

    status = collector.get_collection_status()
    assert status["status"] == "healthy"
    assert status["recent_statistics"]["total_articles"] == 5

    # Now simulate error
    def raise_err():
        raise RuntimeError("db fail")

    monkeypatch.setattr(collector.storage, "health_check", raise_err)
    status2 = collector.get_collection_status()
    assert status2["status"] == "error"
    assert "error" in status2


@pytest.mark.unit
def test_context_manager_enter_exit(db_session):
    with PolygonNewsCollector(db_session=db_session) as c:
        assert isinstance(c, PolygonNewsCollector)



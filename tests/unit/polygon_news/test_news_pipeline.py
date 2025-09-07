import pytest
from datetime import datetime, timezone, timedelta

from src.data_collector.polygon_news.news_pipeline import PolygonNewsCollector


@pytest.mark.unit
def test__collect_ticker_news_no_raw_articles(db_session, mocker):
    collector = PolygonNewsCollector(db_session=db_session)

    # mocker client to return no articles
    mocker.patch.object(collector.news_client, "get_news_for_ticker", lambda **kw: [])

    stats = collector._collect_ticker_news(
        "ACME", datetime.now(timezone.utc), datetime.now(timezone.utc), 50
    )

    if stats.get("api_calls") != 1:
        raise AssertionError("API calls count unexpected for empty result")
    if stats.get("articles_fetched") != 0:
        raise AssertionError("Articles fetched should be 0 when client returns no articles")


@pytest.mark.unit
def test__collect_ticker_news_happy_flow(
    db_session, mocker, processed_article_expected
):
    collector = PolygonNewsCollector(db_session=db_session)

    raw_articles = [{"id": "a1"}]
    mocker.patch.object(
        collector.news_client, "get_news_for_ticker", lambda **kw: raw_articles
    )
    mocker.patch.object(
        collector.news_client,
        "extract_article_metadata",
        lambda raw: processed_article_expected,
    )
    mocker.patch.object(
        collector.processor, "process_article", lambda meta: processed_article_expected
    )
    mocker.patch.object(
        collector.validator, "validate_article", lambda art: (True, 0.9, [])
    )
    mocker.patch.object(
        collector.storage,
        "store_articles_batch",
        lambda arts: {"new_articles": 1, "updated_articles": 0, "skipped_articles": 0},
    )

    stats = collector._collect_ticker_news(
        "ACME", datetime.now(timezone.utc), datetime.now(timezone.utc), 50
    )

    if stats.get("api_calls") != 1:
        raise AssertionError("API calls unexpected in happy flow")
    if stats.get("articles_fetched") != 1:
        raise AssertionError("Articles fetched unexpected in happy flow")
    if stats.get("articles_stored") != 1:
        raise AssertionError("Articles stored unexpected in happy flow")


@pytest.mark.unit
def test__collect_ticker_news_skips_invalid(
    db_session, mocker, processed_article_expected
):
    collector = PolygonNewsCollector(db_session=db_session)

    raw_articles = [{"id": "a1"}, {"id": "a2"}]
    mocker.patch.object(
        collector.news_client, "get_news_for_ticker", lambda **kw: raw_articles
    )
    mocker.patch.object(
        collector.news_client,
        "extract_article_metadata",
        lambda raw: processed_article_expected,
    )

    # First article valid, second invalid
    def validator_seq(article):
        if article.get("polygon_id") == processed_article_expected.get("polygon_id"):
            return (True, 0.8, [])
        return (False, 0.0, ["No associated tickers"])

    mocker.patch.object(
        collector.validator, "validate_article", lambda art: (True, 0.8, [])
    )
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

    if stats.get("articles_fetched") != 2:
        raise AssertionError("Articles fetched count mismatch for skip-flow test")
    if stats.get("articles_stored") != 1:
        raise AssertionError("Articles stored count mismatch for skip-flow test")
    if stats.get("articles_skipped", 0) < 1:
        raise AssertionError("Expected at least one skipped article in skip-flow test")


@pytest.mark.unit
def test__collect_ticker_news_processing_exception_per_article(db_session, mocker):
    collector = PolygonNewsCollector(db_session=db_session)

    raw_articles = [{"id": "a1"}, {"id": "a2"}]
    mocker.patch.object(
        collector.news_client, "get_news_for_ticker", lambda **kw: raw_articles
    )

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

    mocker.patch.object(
        collector.news_client, "extract_article_metadata", lambda raw: raw
    )
    mocker.patch.object(collector.processor, "process_article", proc)
    mocker.patch.object(
        collector.validator, "validate_article", lambda art: (True, 0.8, [])
    )
    mocker.patch.object(
        collector.storage,
        "store_articles_batch",
        lambda arts: {"new_articles": 1, "updated_articles": 0, "skipped_articles": 0},
    )

    stats = collector._collect_ticker_news(
        "ACME", datetime.now(timezone.utc), datetime.now(timezone.utc), 50, limit=2
    )

    if stats.get("articles_fetched") != 2:
        raise AssertionError("Articles fetched count mismatch for exception handling test")
    if stats.get("articles_stored") != 1:
        raise AssertionError("Articles stored count mismatch for exception handling test")
    if stats.get("articles_skipped", 0) < 1:
        raise AssertionError("Expected at least one skipped article when processing errors occur")


@pytest.mark.unit
def test_collect_targeted_news_happy_flow(db_session, mocker):
    collector = PolygonNewsCollector(db_session=db_session)

    # validate_ticker_list -> one valid
    mocker.patch.object(
        collector.ticker_integration,
        "validate_ticker_list",
        lambda tickers: (tickers, []),
    )
    mocker.patch.object(
        collector.ticker_integration,
        "get_ticker_info",
        lambda t: {"priority_score": 50},
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

    if stats.get("total_api_calls") != 1:
        raise AssertionError("Total API calls mismatch in collect_targeted_news")
    if stats.get("total_articles_fetched") != 2:
        raise AssertionError("Total articles fetched mismatch in collect_targeted_news")


@pytest.mark.unit
def test_get_collection_status_handles_healthy_and_error(db_session, mocker):
    collector = PolygonNewsCollector(db_session=db_session)

    mocker.patch.object(
        collector.storage, "health_check", lambda: {"status": "healthy"}
    )
    mocker.patch.object(
        collector.storage, "get_latest_date_overall", lambda: datetime.now(timezone.utc)
    )
    mocker.patch.object(
        collector.storage,
        "get_article_statistics",
        lambda start_date=None: {"total_articles": 5},
    )

    status = collector.get_collection_status()
    if status.get("status") != "healthy":
        raise AssertionError("Expected healthy status from storage health_check")
    if status.get("recent_statistics", {}).get("total_articles") != 5:
        raise AssertionError("Recent statistics total_articles mismatch")

    # Now simulate error
    def raise_err():
        raise RuntimeError("db fail")

    mocker.patch.object(collector.storage, "health_check", raise_err)
    status2 = collector.get_collection_status()
    if status2.get("status") != "error":
        raise AssertionError("Expected error status when health_check raises")
    if "error" not in status2:
        raise AssertionError("Expected error key in status2")


@pytest.mark.unit
def test_context_manager_enter_exit(db_session):
    with PolygonNewsCollector(db_session=db_session) as c:
        if not isinstance(c, PolygonNewsCollector):
            raise AssertionError("Context manager did not return PolygonNewsCollector instance")

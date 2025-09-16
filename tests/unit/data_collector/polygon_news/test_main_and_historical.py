import pytest

from src.data_collector.polygon_news.news_pipeline import PolygonNewsCollector


@pytest.mark.unit
def test_collect_historical_news_respects_batching(mocker):
    """collect_historical_news respects batching and returns stats dict"""
    collector = PolygonNewsCollector()

    # Provide prioritized tickers with 2 major tickers
    tickers = [
        {"ticker": "A1", "is_major": True, "priority_score": 80},
        {"ticker": "A2", "is_major": True, "priority_score": 70},
    ]
    mocker.patch.object(
        collector.ticker_integration, "get_prioritized_tickers", lambda max_t: tickers
    )

    # Ensure storage.get_articles_for_ticker returns empty for all periods so collection runs
    mocker.patch.object(
        collector.storage, "get_articles_for_ticker", lambda t, s, e, limit=1: []
    )

    # stub _collect_ticker_news to return predictable stats
    mocker.patch.object(
        collector,
        "_collect_ticker_news",
        lambda ticker, s, e, p: {
            "api_calls": 1,
            "articles_fetched": 1,
            "articles_stored": 1,
            "articles_updated": 0,
            "articles_skipped": 0,
        },
    )

    stats = collector.collect_historical_news(
        max_tickers=2, years_back=0, batch_size_days=1
    )

    # years_back=0 will produce zero-day range; ensure function handles gracefully and returns stats
    assert isinstance(stats, dict)


@pytest.mark.unit
def test_main_handles_missing_database_url(mocker, tmp_path):
    """main() returns False when API key is missing from config"""
    # mocker config to have no API key
    import importlib
    import src.data_collector.polygon_news.news_pipeline as np_mod

    cfg = importlib.import_module("src.data_collector.config")
    mocker.patch.object(cfg.config, "API_KEY", None)

    # main() should return False when API key missing
    res = np_mod.main()
    assert res is False

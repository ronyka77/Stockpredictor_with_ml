import pytest

from src.data_collector.polygon_news.ticker_integration import NewsTickerIntegration


@pytest.mark.unit
def test_get_prioritized_tickers_fallback_and_filters():
    nti = NewsTickerIntegration()
    tickers = nti.get_prioritized_tickers(max_tickers=5, include_etfs=False)
    assert isinstance(tickers, list)
    assert len(tickers) <= 5
    # No ETFs should be present (ETF names like SPY should be skipped if include_etfs=False)
    assert all(not nti._is_etf(t["ticker"]) for t in tickers)


@pytest.mark.unit
def test_validate_ticker_list_filters_invalid():
    nti = NewsTickerIntegration()
    valid, invalid = nti.validate_ticker_list(["AAPL", "", "BAD$", None, "GOOGL"])
    assert "AAPL" in valid
    assert "GOOGL" in valid
    assert ("" in invalid) or (None in invalid)


@pytest.mark.unit
def test_get_ticker_info_from_manager(mocker):
    # Provide a fake ticker manager
    class FakeManager:
        def get_active_tickers(self):
            return [
                {
                    "ticker": "XYZ",
                    "market_cap": 200e9,
                    "avg_volume": 30e6,
                    "sector": "Technology",
                },
                {
                    "ticker": "ABC",
                    "market_cap": 5e8,
                    "avg_volume": 100000,
                    "sector": "Small",
                },
            ]

    nti = NewsTickerIntegration(ticker_manager=FakeManager())
    pri = nti.get_prioritized_tickers(max_tickers=10)
    assert any(p["ticker"] == "XYZ" for p in pri)
    info = nti.get_ticker_info("XYZ")
    assert info["ticker"] == "XYZ"

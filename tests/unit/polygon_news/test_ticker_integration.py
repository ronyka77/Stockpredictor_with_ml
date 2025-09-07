import pytest

from src.data_collector.polygon_news.ticker_integration import NewsTickerIntegration


@pytest.mark.unit
def test_get_prioritized_tickers_fallback_and_filters():
    nti = NewsTickerIntegration()
    tickers = nti.get_prioritized_tickers(max_tickers=5, include_etfs=False)
    if not isinstance(tickers, list):
        raise AssertionError("Prioritized tickers should be returned as a list")
    if len(tickers) > 5:
        raise AssertionError("Returned more tickers than requested by max_tickers")
    # No ETFs should be present (ETF names like SPY should be skipped if include_etfs=False)
    if not all(not nti._is_etf(t["ticker"]) for t in tickers):
        raise AssertionError("ETF tickers present despite include_etfs=False")


@pytest.mark.unit
def test_validate_ticker_list_filters_invalid():
    nti = NewsTickerIntegration()
    valid, invalid = nti.validate_ticker_list(["AAPL", "", "BAD$", None, "GOOGL"])
    if "AAPL" not in valid:
        raise AssertionError("Valid tickers missing expected 'AAPL'")
    if "GOOGL" not in valid:
        raise AssertionError("Valid tickers missing expected 'GOOGL'")
    if not ("" in invalid or None in invalid):
        raise AssertionError("Invalid ticker list did not include expected empty/None entries")


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
    if not any(p["ticker"] == "XYZ" for p in pri):
        raise AssertionError("Expected prioritized tickers to include 'XYZ'")
    info = nti.get_ticker_info("XYZ")
    if info["ticker"] != "XYZ":
        raise AssertionError("Ticker info lookup returned unexpected ticker")

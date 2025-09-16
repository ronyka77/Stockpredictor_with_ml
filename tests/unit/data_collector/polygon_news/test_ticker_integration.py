import pytest

from src.data_collector.polygon_news.ticker_integration import NewsTickerIntegration


@pytest.mark.unit
def test_get_prioritized_tickers_fallback_and_filters():
    """get_prioritized_tickers filters out ETFs and limits results by max_tickers"""
    nti = NewsTickerIntegration()
    tickers = nti.get_prioritized_tickers(max_tickers=5)
    assert isinstance(tickers, list)
    assert len(tickers) <= 5
    # No ETFs should be present (ETF names like SPY should be skipped if include_etfs=False)
    assert all(not nti._is_etf(t["ticker"]) for t in tickers)


@pytest.mark.unit
def test_validate_ticker_list_filters_invalid():
    """validate_ticker_list separates valid and invalid ticker inputs"""
    nti = NewsTickerIntegration()
    valid, invalid = nti.validate_ticker_list(["AAPL", "", "BAD$", None, "GOOGL"])
    assert "AAPL" in valid
    assert "GOOGL" in valid
    assert ("" in invalid) or (None in invalid)


@pytest.mark.unit
def test_get_ticker_info_from_manager(mocker):
    """Pull ticker info from manager and compute priority list entries"""

    # Patch self.ticker_manager.storage.get_tickers() to return fake ticker data
    class FakeStorage:
        def get_tickers(self):
            """
            Return a list of example ticker records used for testing.
            
            Each entry is a dict with the following keys:
            - "ticker" (str): Ticker symbol.
            - "market_cap" (float): Market capitalization in USD.
            - "avg_volume" (float): Average trading volume.
            - "sector" (str): Sector name.
            
            Returns:
                list[dict]: Two sample ticker records ("XYZ" and "ABC") used by tests.
            """
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

    class FakeManager:
        def __init__(self):
            """
            Initialize the fake manager by creating and assigning a FakeStorage instance to self.storage.
            
            Provides a storage-backed ticker source accessible as the manager's .storage attribute for tests.
            """
            self.storage = FakeStorage()

    nti = NewsTickerIntegration(ticker_manager=FakeManager())
    pri = nti.get_prioritized_tickers(max_tickers=10)
    assert any(p["ticker"] == "XYZ" for p in pri)
    info = nti.get_ticker_info("XYZ")
    assert info["ticker"] == "XYZ"

"""
Integration with existing ticker_manager for news collection prioritization
Provides intelligent ticker selection based on market cap and trading volume
"""

from typing import List, Dict, Optional, Tuple, Any

from src.utils.logger import get_logger
from src.data_collector.ticker_manager import TickerManager

logger = get_logger(__name__, utility="data_collector")


class NewsTickerIntegration:
    """
    Integration layer between news collection and existing ticker management
    Provides prioritized ticker lists for efficient news collection
    """

    def __init__(self, ticker_manager: Optional[TickerManager] = None):
        """
        Initialize ticker integration

        Args:
            ticker_manager: Optional existing TickerManager instance
        """
        self.ticker_manager = ticker_manager
        self.logger = get_logger(self.__class__.__name__, utility="data_collector")

        # Fallback ticker lists if ticker_manager is unavailable
        self.fallback_major_tickers = [
            "AAPL",
            "GOOGL",
            "MSFT",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            "NFLX",
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
            "VTI",
            "VOO",
            "VEA",
            "VWO",
            "JPM",
            "BAC",
            "WFC",
            "GS",
            "MS",
            "C",
            "BRK.B",
            "V",
            "MA",
            "JNJ",
            "PFE",
            "UNH",
            "ABBV",
            "MRK",
            "TMO",
            "DHR",
            "ABT",
            "XOM",
            "CVX",
            "COP",
            "SLB",
            "EOG",
            "PXD",
            "KMI",
            "OKE",
        ]

        self.fallback_growth_tickers = [
            "TSLA",
            "NVDA",
            "AMD",
            "CRM",
            "SNOW",
            "PLTR",
            "ROKU",
            "SQ",
            "SHOP",
            "TWLO",
            "ZM",
            "DOCU",
            "OKTA",
            "CRWD",
            "NET",
            "DDOG",
        ]

        self.fallback_value_tickers = [
            "BRK.B",
            "JPM",
            "BAC",
            "WFC",
            "XOM",
            "CVX",
            "JNJ",
            "PG",
            "KO",
            "PEP",
            "WMT",
            "HD",
            "VZ",
            "T",
            "IBM",
            "INTC",
        ]

    def get_prioritized_tickers(
        self,
        max_tickers: int = 100,
        include_etfs: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Return a prioritized list of tickers for news collection.
        
        If a TickerManager was provided at construction, attempts to retrieve prioritized tickers from it; on any error it falls back to the built-in static lists. If no manager is available, returns the fallback tickers.
        
        Parameters:
            max_tickers (int): Maximum number of tickers to return.
            include_etfs (bool): Whether ETFs should be included in the returned list.
        
        Returns:
            List[Dict[str, Any]]: Ordered list of ticker dictionaries. Each dictionary contains keys such as
            'ticker', 'priority_score', 'market_cap', 'avg_volume', 'sector', 'is_major', and 'category'.
        """
        if self.ticker_manager:
            try:
                return self._get_tickers_from_manager(
                    max_tickers, include_etfs
                )
            except Exception as e:
                self.logger.warning(f"Failed to get tickers from manager: {e}")
                return self._get_fallback_tickers(max_tickers, include_etfs)
        else:
            self.logger.info("Using fallback ticker list")
            return self._get_fallback_tickers(max_tickers, include_etfs)

    def _get_tickers_from_manager(
        self, max_tickers: int, include_etfs: bool
    ) -> List[Dict[str, Any]]:
        """
        Builds and returns a prioritized list of tickers sourced from the configured ticker manager.
        
        Retrieves tickers from self.ticker_manager.storage.get_tickers(), computes a numeric
        priority score for each entry using market cap and average volume, filters out ETFs
        when include_etfs is False, categorizes each ticker, and returns the top results
        sorted by descending priority.
        
        Parameters:
            max_tickers (int): Maximum number of ticker entries to return.
            include_etfs (bool): If False, ETF-like tickers (per _is_etf) are excluded.
        
        Returns:
            List[Dict[str, Any]]: Sorted list (highest priority first) of ticker dictionaries.
            Each dictionary contains:
                - ticker (str)
                - priority_score (float)
                - market_cap (Optional[float])
                - avg_volume (Optional[float])
                - sector (str)
                - is_major (bool)  # whether the ticker is in the fallback major list
                - category (str)   # value from _categorize_ticker
        
        Raises:
            Exception: Propagates exceptions raised while fetching or processing tickers
            from the ticker manager.
        """
        try:
            # Get active tickers from manager
            active_tickers = self.ticker_manager.storage.get_tickers()

            prioritized_tickers = []

            for ticker_info in active_tickers:
                ticker = ticker_info.get("ticker", "")
                market_cap = ticker_info.get("market_cap", 0)
                avg_volume = ticker_info.get("avg_volume", 0)
                sector = ticker_info.get("sector", "")

                # Skip ETFs if not requested
                if not include_etfs and self._is_etf(ticker):
                    continue

                # Calculate priority score
                priority_score = self._calculate_priority_score(
                    market_cap, avg_volume
                )

                prioritized_tickers.append(
                    {
                        "ticker": ticker,
                        "priority_score": priority_score,
                        "market_cap": market_cap,
                        "avg_volume": avg_volume,
                        "sector": sector,
                        "is_major": ticker in self.fallback_major_tickers,
                        "category": self._categorize_ticker(ticker, market_cap),
                    }
                )

            # Sort by priority score (highest first)
            prioritized_tickers.sort(key=lambda x: x["priority_score"], reverse=True)

            return prioritized_tickers[:max_tickers]

        except Exception as e:
            self.logger.error(f"Error getting tickers from manager: {e}")
            raise

    def _get_fallback_tickers(
        self, max_tickers: int, include_etfs: bool
    ) -> List[Dict[str, Any]]:
        """
        Return a stable fallback list of prioritized ticker dictionaries when the ticker manager is unavailable.
        
        The list places predefined major tickers first (highest priority), then appends value tickers until the requested maximum is reached.
        ETFs are excluded when include_etfs is False. Each returned entry is a dict with keys:
        - ticker: ticker symbol (str)
        - priority_score: numeric priority (higher = more important)
        - market_cap: None (fallback entries do not include market cap)
        - avg_volume: None (fallback entries do not include average volume)
        - sector: a best-effort sector string ("Unknown" for major, "Financial" for value)
        - is_major: True for major list members, False for value list members
        - category: "major" or "value"
        
        Parameters:
            max_tickers (int): Maximum number of tickers to return.
            include_etfs (bool): If False, ETF-like tickers are omitted.
        
        Returns:
            List[Dict[str, Any]]: Up to max_tickers fallback ticker dictionaries ordered by priority.
        """
        fallback_tickers = []

        # Major tickers (highest priority)
        for i, ticker in enumerate(self.fallback_major_tickers):
            if not include_etfs and self._is_etf(ticker):
                continue

            fallback_tickers.append(
                {
                    "ticker": ticker,
                    "priority_score": 100 - i,  # Decreasing priority
                    "market_cap": None,
                    "avg_volume": None,
                    "sector": "Unknown",
                    "is_major": True,
                    "category": "major",
                }
            )

        # Add value tickers if we still need more
        if len(fallback_tickers) < max_tickers:
            for i, ticker in enumerate(self.fallback_value_tickers):
                if len(fallback_tickers) >= max_tickers:
                    break

                if ticker not in [t["ticker"] for t in fallback_tickers]:
                    fallback_tickers.append(
                        {
                            "ticker": ticker,
                            "priority_score": 25 - i,
                            "market_cap": None,
                            "avg_volume": None,
                            "sector": "Financial",
                            "is_major": False,
                            "category": "value",
                        }
                    )

        return fallback_tickers[:max_tickers]

    def _calculate_priority_score(
        self, market_cap: float, avg_volume: float
    ) -> float:
        """
        Compute a numeric priority score for a ticker based on market capitalization and average trading volume.
        
        Higher returned values indicate higher priority for news collection. The score is composed of two weighted components:
        - Market cap (0–40 points) determined by descending thresholds.
        - Average daily volume (0–30 points) determined by descending thresholds.
        
        Parameters:
            market_cap (float): Market capitalization in dollars (e.g., 1.5e11 for $150B). If falsy or None, the market-cap component is omitted.
            avg_volume (float): Average daily trading volume in shares. If falsy or None, the volume component is omitted.
        
        Returns:
            float: Combined priority score (higher = higher priority).
        """
        score = 0.0

        # Market cap component (0-40 points)
        if market_cap:
            if market_cap >= 500e9:  # $500B+
                score += 40
            elif market_cap >= 100e9:  # $100B+
                score += 35
            elif market_cap >= 50e9:  # $50B+
                score += 30
            elif market_cap >= 10e9:  # $10B+
                score += 25
            elif market_cap >= 1e9:  # $1B+
                score += 15
            else:
                score += 5

        # Volume component (0-30 points)
        if avg_volume:
            if avg_volume >= 50e6:  # 50M+ shares
                score += 30
            elif avg_volume >= 20e6:  # 20M+ shares
                score += 25
            elif avg_volume >= 10e6:  # 10M+ shares
                score += 20
            elif avg_volume >= 5e6:  # 5M+ shares
                score += 15
            elif avg_volume >= 1e6:  # 1M+ shares
                score += 10
            else:
                score += 5

        return score

    def _categorize_ticker(
        self, ticker: str, market_cap: Optional[float]
    ) -> str:
        """
        Determine the news-collection category for a given ticker.
        
        Returns one of: "major", "value", "etf", "large_cap", "mid_cap", or "small_cap".
        - Tickers present in the instance fallback_major_tickers return "major".
        - Tickers present in fallback_value_tickers return "value".
        - ETF-like tickers (per _is_etf) return "etf".
        - If market_cap (USD) is provided, >= 100e9 → "large_cap"; >= 10e9 → "mid_cap".
        - Otherwise returns "small_cap".
        
        Parameters:
            market_cap (Optional[float]): Market capitalization in USD; may be None.
        """
        if ticker in self.fallback_major_tickers:
            return "major"
        elif ticker in self.fallback_value_tickers:
            return "value"
        elif self._is_etf(ticker):
            return "etf"
        elif market_cap and market_cap >= 100e9:
            return "large_cap"
        elif market_cap and market_cap >= 10e9:
            return "mid_cap"
        else:
            return "small_cap"

    def _is_etf(self, ticker: str) -> bool:
        """Check if ticker is likely an ETF"""
        etf_indicators = [
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
            "VTI",
            "VOO",
            "VEA",
            "VWO",
            "XLF",
            "XLK",
            "XLE",
        ]
        return ticker in etf_indicators or len(ticker) <= 3

    def validate_ticker_list(self, tickers: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate ticker list and return valid/invalid tickers

        Args:
            tickers: List of ticker symbols to validate

        Returns:
            Tuple of (valid_tickers, invalid_tickers)
        """
        valid_tickers = []
        invalid_tickers = []

        for ticker in tickers:
            # Basic validation
            if not ticker or not isinstance(ticker, str):
                invalid_tickers.append(ticker)
                continue

            # Clean ticker symbol
            clean_ticker = ticker.strip().upper()

            # Basic format validation
            if len(clean_ticker) < 1 or len(clean_ticker) > 10:
                invalid_tickers.append(ticker)
                continue

            # Check for invalid characters
            if not clean_ticker.replace(".", "").replace("-", "").isalnum():
                invalid_tickers.append(ticker)
                continue

            valid_tickers.append(clean_ticker)

        return valid_tickers, invalid_tickers

    def get_ticker_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific ticker"""
        prioritized_tickers = self.get_prioritized_tickers(1000)

        for ticker_info in prioritized_tickers:
            if ticker_info["ticker"] == ticker.upper():
                return ticker_info

        # Return basic info if not found in prioritized list
        return {
            "ticker": ticker.upper(),
            "priority_score": 0,
            "market_cap": None,
            "avg_volume": None,
            "sector": "Unknown",
            "is_major": ticker.upper() in self.fallback_major_tickers,
            "category": "unknown",
        }

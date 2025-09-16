"""
Polygon.io API client with comprehensive error handling and retry mechanisms
"""

import time
import requests
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse, parse_qs

from src.utils.logger import get_logger
from src.data_collector.polygon_data.rate_limiter import get_rate_limiter
from src.data_collector.config import config

logger = get_logger(__name__, utility="data_collector")


class PolygonAPIError(Exception):
    """Custom exception for Polygon API errors"""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.response_data = response_data
        super().__init__(self.message)


class PolygonDataClient:
    """Main client for interacting with Polygon.io API"""

    def __init__(self, api_key: Optional[str] = None, requests_per_minute: int = 5):
        """
        Initialize the Polygon.io client

        Args:
            api_key: Polygon.io API key (defaults to config)
            requests_per_minute: Rate limit for API requests
        """
        self.api_key = api_key or config.API_KEY
        self.base_url = config.BASE_URL
        self.rate_limiter = get_rate_limiter(requests_per_minute)
        self.session = requests.Session()

        # Set up session headers
        self.session.headers.update(
            {"User-Agent": "StockPredictor/1.0", "Accept": "application/json"}
        )

        logger.info(
            f"Polygon client initialized with {requests_per_minute} requests/minute limit"
        )

    def _make_request(
        self, endpoint: str, params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make an API request with comprehensive error handling and retries

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            PolygonAPIError: For API-related errors
        """
        # Apply rate limiting (no-op if disabled)
        self.rate_limiter.wait_if_needed()

        # Prepare request
        url = urljoin(self.base_url, endpoint)
        params = params or {}
        params["apikey"] = self.api_key

        # Retry logic with exponential backoff
        for attempt in range(config.MAX_RETRIES):
            try:
                logger.info(
                    f"Making request to {endpoint} (attempt {attempt + 1}/{config.MAX_RETRIES})"
                )

                response = self.session.get(
                    url, params=params, timeout=config.REQUEST_TIMEOUT
                )

                # Handle different response status codes
                if response.status_code == 200:
                    try:
                        # Optional on some limiter implementations
                        self.rate_limiter.handle_successful_request()
                    except Exception as e:
                        logger.error(f"Error handling successful request: {e}")
                        pass
                    # Safely parse JSON and convert parsing errors to PolygonAPIError
                    try:
                        data = response.json()
                    except Exception as e:
                        logger.error(
                            f"Error parsing JSON response from {endpoint}: {e}"
                        )
                        raise PolygonAPIError(
                            "Malformed JSON in response", response.status_code
                        )

                    # Check for API-level errors in response
                    if data.get("status") == "ERROR":
                        error_msg = data.get("error", "Unknown API error")
                        raise PolygonAPIError(
                            f"API Error: {error_msg}", response.status_code, data
                        )

                    logger.info(f"Successful request to {endpoint}")
                    return data

                elif response.status_code == 429:  # Rate limited
                    if getattr(config, "DISABLE_RATE_LIMITING", False):
                        # If plan has no limits, treat as transient and retry immediately without sleeping
                        logger.warning(
                            "Received 429 but rate limiting disabled. Retrying immediately."
                        )
                        continue
                    else:
                        # Backoff path when limiting enabled
                        try:
                            # Not all limiters implement this; guard call
                            self.rate_limiter.handle_rate_limit_error()
                        except Exception as e:
                            logger.error(f"Error handling rate limit error: {e}")
                            pass
                        wait_time = 2**attempt
                        logger.warning(
                            f"Rate limit exceeded. Waiting {wait_time}s before retry"
                        )
                        time.sleep(wait_time)
                        continue

                elif response.status_code == 401:  # Unauthorized
                    raise PolygonAPIError("Invalid API key", response.status_code)

                elif response.status_code == 403:  # Forbidden
                    raise PolygonAPIError(
                        "Access forbidden - check subscription level",
                        response.status_code,
                    )

                elif response.status_code >= 500:  # Server errors
                    wait_time = 2**attempt
                    logger.warning(
                        f"Server error {response.status_code}. Retrying in {wait_time}s"
                    )
                    time.sleep(wait_time)
                    continue

                else:
                    # Other client errors
                    try:
                        error_data = response.json()
                        error_msg = error_data.get(
                            "error", f"HTTP {response.status_code}"
                        )
                    except Exception as e:
                        logger.error(f"Error parsing response: {e}")
                        error_msg = f"HTTP {response.status_code}"

                    raise PolygonAPIError(error_msg, response.status_code)

            except requests.exceptions.Timeout:
                wait_time = 2**attempt
                if getattr(config, "DISABLE_RATE_LIMITING", False):
                    logger.warning(
                        "Request timeout. Retrying without delay (rate limiting disabled)"
                    )
                else:
                    logger.warning(f"Request timeout. Retrying in {wait_time}s")
                    if attempt < config.MAX_RETRIES - 1:
                        time.sleep(wait_time)
                        continue
                raise PolygonAPIError("Request timeout after all retries")

            except requests.exceptions.ConnectionError:
                wait_time = 2**attempt
                if getattr(config, "DISABLE_RATE_LIMITING", False):
                    logger.warning(
                        "Connection error. Retrying without delay (rate limiting disabled)"
                    )
                else:
                    logger.warning(f"Connection error. Retrying in {wait_time}s")
                    if attempt < config.MAX_RETRIES - 1:
                        time.sleep(wait_time)
                        continue
                raise PolygonAPIError("Connection error after all retries")

            except requests.exceptions.RequestException as e:
                if attempt == config.MAX_RETRIES - 1:
                    raise PolygonAPIError(f"Request failed: {str(e)}")

                wait_time = 2**attempt
                if getattr(config, "DISABLE_RATE_LIMITING", False):
                    logger.warning(
                        f"Request failed: {e}. Retrying without delay (rate limiting disabled)"
                    )
                else:
                    logger.warning(f"Request failed: {e}. Retrying in {wait_time}s")
                    time.sleep(wait_time)

        raise PolygonAPIError(
            f"Failed to complete request after {config.MAX_RETRIES} attempts"
        )

    def _fetch_paginated_data(
        self, endpoint: str, params: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Aggregate and return all items from a paginated Polygon API endpoint.
        
        This method repeatedly requests pages from the given endpoint (first using
        `endpoint` and `params`, thereafter following `next_url` values returned by
        the API) until no further pages remain or a page returns no `results`. When a
        subsequent page is requested, the `next_url` is parsed to extract the request
        path and query parameters.
        
        Parameters:
            endpoint (str): Initial API endpoint path (e.g. "/v3/reference/dividends").
            params (Optional[Dict]): Initial query parameters for the first request.
        
        Returns:
            List[Dict]: Concatenated list of result objects from all pages. Each page is
            expected to include a `"results"` list and may include a `"next_url"` string
            used to fetch the next page.
        """
        all_results = []
        next_url = None
        page_count = 0

        while True:
            page_count += 1
            logger.info(f"Fetching page {page_count} from {endpoint}")

            if next_url:
                # Parse next_url to extract parameters
                parsed_url = urlparse(next_url)
                next_params = parse_qs(parsed_url.query)

                # Convert query parameters back to single values
                request_params = {
                    k: v[0] if isinstance(v, list) and len(v) == 1 else v
                    for k, v in next_params.items()
                }

                # Make request to next page
                response = self._make_request(parsed_url.path, request_params)
            else:
                # First page
                response = self._make_request(endpoint, params)

            # Extract results
            if "results" in response and response["results"]:
                results = response["results"]
                all_results.extend(results)
                logger.info(
                    f"Page {page_count}: fetched {len(results)} records. "
                    f"Total: {len(all_results)}"
                )
            else:
                logger.info(f"Page {page_count}: no results found")
                break

            # Check for next page
            if "next_url" not in response or not response["next_url"]:
                logger.info(
                    f"Pagination complete. Total pages: {page_count}, "
                    f"Total records: {len(all_results)}"
                )
                break

            next_url = response["next_url"]

        return all_results

    def get_dividends(
        self,
        ticker: str,
        order: str = "desc",
        limit: int = 1000,
        sort: str = "ex_dividend_date",
    ) -> List[Dict]:
        """
        Retrieve all dividend records for a single ticker from Polygon's dividends endpoint.
        
        The Polygon dividends endpoint accepts only one ticker, so a ticker value is required.
        Parameters:
            order: 'asc' or 'desc' ordering of results.
            limit: Maximum results per page (capped to 1000).
            sort: Field to sort by (default 'ex_dividend_date').
        
        Returns:
            List[Dict]: Aggregated dividend records across all pages.
        """
        if not ticker:
            raise ValueError("ticker is required for get_dividends")

        params = {
            "ticker": ticker,
            "order": order,
            "limit": min(limit, 1000),
            "sort": sort,
        }
        return self._fetch_paginated_data("/v3/reference/dividends", params)

    def get_tickers(
        self, market: str = "stocks", active: bool = True, limit: int = 1000, **kwargs
    ) -> List[Dict]:
        """
        Return a list of tickers from the Polygon.io reference tickers endpoint.
        
        Builds query parameters from the provided arguments (limit is capped at 1000) and fetches all pages, returning an aggregated list of ticker records. Any additional keyword arguments are passed through as extra query parameters to the API.
        
        Parameters:
            market: Market to query (e.g., "stocks", "crypto", "fx").
            active: If True, only include active tickers.
            limit: Maximum number of results per page (will be capped to 1000).
            **kwargs: Additional query parameters forwarded to the API.
        
        Returns:
            List[Dict]: Aggregated list of ticker information dictionaries.
        """
        params = {
            "market": market,
            "active": str(active).lower(),
            "limit": min(limit, 1000),  # API maximum
            "sort": "ticker",
            **kwargs,
        }

        logger.info(f"Fetching {market} tickers (active={active})")
        return self._fetch_paginated_data("/v3/reference/tickers", params)

    def get_aggregates(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        from_date: str,
        to_date: str,
        limit: int = 50000,
    ) -> List[Dict]:
        """
        Get aggregate (OHLCV) data for a ticker

        Args:
            ticker: Stock ticker symbol
            multiplier: Size of the timespan multiplier
            timespan: Size of the time window (minute, hour, day, week, month, quarter, year)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Number of results to return

        Returns:
            List of OHLCV records
        """
        endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"

        params = {"adjusted": "true", "sort": "asc", "limit": min(limit, 50000)}

        logger.info(
            f"Fetching {timespan} aggregates for {ticker} from {from_date} to {to_date}"
        )

        response = self._make_request(endpoint, params)
        return response.get("results", [])

    def get_grouped_daily(self, date: str) -> List[Dict]:
        """Get grouped daily (OHLCV) data for all stocks on a specific date"""
        endpoint = f"/v2/aggs/grouped/locale/us/market/stocks/{date}"

        params = {"adjusted": "true"}

        logger.info(f"Fetching grouped daily data for {date}")

        response = self._make_request(endpoint, params)
        return response.get("results", [])

    def get_ticker_details(self, ticker: str) -> Dict:
        """
        Get detailed information about a ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            Ticker details
        """
        endpoint = f"/v3/reference/tickers/{ticker}"

        logger.info(f"Fetching details for ticker {ticker}")

        response = self._make_request(endpoint)
        return response.get("results", {})

    def health_check(self) -> bool:
        """Perform a health check on the API connection"""
        try:
            # Simple request to check API connectivity
            response = self._make_request("/v3/reference/tickers", {"limit": 1})
            return "results" in response
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup session"""
        self.session.close()

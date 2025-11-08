"""
Polygon.io API client with comprehensive error handling and retry mechanisms
"""

import requests
from typing import Dict, List, Optional, Any, Union, TypedDict, cast, Generator
from urllib.parse import urljoin, urlparse, parse_qs

from src.utils.core.logger import get_logger
from src.utils.core.retry import (
    retry,
    API_RETRY_CONFIG,
    API_CIRCUIT_BREAKER,
    RetryError,
    CircuitBreakerOpenError,
    RetryConfig,
    CircuitBreaker
)
from src.data_collector.polygon_data.rate_limiter import get_rate_limiter
from src.data_collector.config import config

logger = get_logger(__name__, utility="data_collector")


# Type definitions for API responses
class DividendRecord(TypedDict, total=False):
    """Dividend record structure from Polygon API"""
    cash_amount: float
    currency: str
    declaration_date: str
    dividend_type: str
    ex_dividend_date: str
    frequency: int
    pay_date: str
    record_date: str
    ticker: str


class TickerInfo(TypedDict, total=False):
    """Ticker information structure from Polygon API"""
    ticker: str
    name: str
    market: str
    locale: str
    primary_exchange: str
    type: str
    active: bool
    currency_name: str
    cik: str
    composite_figi: str
    share_class_figi: str
    last_updated_utc: str


class OHLCVRecord(TypedDict, total=False):
    """OHLCV record structure from Polygon API"""
    c: float  # close
    h: float  # high
    low: float  # low
    n: int    # transactions
    o: float  # open
    t: int    # timestamp
    v: float  # volume
    vw: float # volume weighted average price


class APIResponse(TypedDict, total=False):
    """Generic API response structure"""
    results: List[Dict[str, Any]]
    status: str
    request_id: str
    next_url: str
    count: int
    error: str


class PolygonAPIError(Exception):
    """Custom exception for Polygon API errors with comprehensive error classification"""

    # Error categories for better error handling and retry decisions
    NETWORK_ERRORS = (ConnectionError, TimeoutError, OSError)
    RETRYABLE_HTTP_ERRORS = (429, 500, 502, 503, 504)  # Rate limits and server errors
    NON_RETRYABLE_HTTP_ERRORS = (400, 401, 403, 404, 422)  # Client errors
    AUTHENTICATION_ERRORS = (401, 403)  # Authentication/authorization issues

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        error_category: Optional[str] = None
    ) -> None:
        self.message: str = message
        self.status_code: Optional[int] = status_code
        self.response_data: Optional[Dict[str, Any]] = response_data
        self.error_category: str = error_category or self._classify_error()
        super().__init__(self.message)

    def _classify_error(self) -> str:
        """Classify the error type based on status code and context"""
        if self.status_code:
            if self.status_code in self.AUTHENTICATION_ERRORS:
                return "authentication"
            elif self.status_code in self.RETRYABLE_HTTP_ERRORS:
                return "retryable"
            elif self.status_code in self.NON_RETRYABLE_HTTP_ERRORS:
                return "client_error"
            elif self.status_code >= 500:
                return "server_error"
        return "unknown"

    def is_retryable(self) -> bool:
        """Determine if this error should trigger a retry"""
        return self.error_category in ("retryable", "server_error", "network")

    def is_authentication_error(self) -> bool:
        """Check if this is an authentication/authorization error"""
        return self.error_category == "authentication"

    def get_retry_config(self) -> RetryConfig:
        """Get appropriate retry configuration based on error type"""
        if self.is_authentication_error():
            # Don't retry authentication errors
            return RetryConfig(max_attempts=1)
        elif self.error_category == "retryable":
            # Use shorter delays for rate limits
            return RetryConfig(
                max_attempts=5,
                base_delay=1.0,
                max_delay=30.0,
                backoff_factor=2.0
            )
        else:
            # Use standard API retry config
            return API_RETRY_CONFIG


class PolygonDataClient:
    """Main client for interacting with Polygon.io API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        requests_per_minute: int = 5,
        circuit_breaker: Optional[CircuitBreaker] = None
    ) -> None:
        """
        Initialize the Polygon.io client with comprehensive error handling and retry mechanisms

        Args:
            api_key: Polygon.io API key (defaults to config)
            requests_per_minute: Rate limit for API requests
            circuit_breaker: Optional circuit breaker for fault tolerance (defaults to shared instance)
        """
        self.api_key: str = api_key or config.API_KEY
        self.base_url: str = config.BASE_URL
        self.rate_limiter: Any = get_rate_limiter(requests_per_minute)
        self.circuit_breaker: CircuitBreaker = circuit_breaker or API_CIRCUIT_BREAKER
        self.session: requests.Session = requests.Session()

        # Set up session headers
        self.session.headers.update(
            {"User-Agent": "StockPredictor/1.0", "Accept": "application/json"}
        )

        # logger.info(
        #     f"Polygon client initialized with {requests_per_minute} requests/minute limit"
        # )

    def _make_single_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a single HTTP request without retry logic (used by the retry decorator)

        Args:
            url: Full URL to request
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            PolygonAPIError: For API-related errors
            requests.RequestException: For network-related errors
        """
        logger.debug(f"Making request to {url}")

        response = self.session.get(url, params=params, timeout=config.REQUEST_TIMEOUT)

        # Handle different response status codes
        if response.status_code == 200:
            try:
                # Optional on some limiter implementations
                self.rate_limiter.handle_successful_request()
            except Exception as e:
                logger.debug(f"Error handling successful request: {e}")

            # Safely parse JSON and convert parsing errors to PolygonAPIError
            try:
                data: Dict[str, Any] = response.json()
            except Exception as e:
                logger.error(f"Error parsing JSON response from {url}: {e}")
                raise PolygonAPIError(
                    "Malformed JSON in response",
                    response.status_code,
                    error_category="client_error"
                )

            # Check for API-level errors in response
            if data.get("status") == "ERROR":
                error_msg = data.get("error", "Unknown API error")
                raise PolygonAPIError(
                    f"API Error: {error_msg}",
                    response.status_code,
                    data,
                    error_category="client_error"
                )

            logger.debug(f"Successful request to {url}")
            return data

        elif response.status_code == 429:  # Rate limited
            try:
                self.rate_limiter.handle_rate_limit_error()
            except Exception as e:
                logger.debug(f"Error handling rate limit error: {e}")

            error_msg = "Rate limit exceeded"
            if not getattr(config, "DISABLE_RATE_LIMITING", False):
                logger.warning(f"{error_msg} - triggering retry with backoff")
            raise PolygonAPIError(
                error_msg,
                response.status_code,
                error_category="retryable"
            )

        elif response.status_code == 401:  # Unauthorized
            raise PolygonAPIError(
                "Invalid API key",
                response.status_code,
                error_category="authentication"
            )

        elif response.status_code == 403:  # Forbidden
            raise PolygonAPIError(
                "Access forbidden - check subscription level",
                response.status_code,
                error_category="authentication"
            )

        elif response.status_code >= 500:  # Server errors
            error_msg = f"Server error {response.status_code}"
            logger.warning(f"{error_msg} - triggering retry")
            raise PolygonAPIError(
                error_msg,
                response.status_code,
                error_category="server_error"
            )

        else:
            # Other client errors
            try:
                error_data = response.json()
                error_msg = error_data.get("error", f"HTTP {response.status_code}")
            except Exception as e:
                logger.debug(f"Error parsing error response: {e}")
                error_msg = f"HTTP {response.status_code}"

            raise PolygonAPIError(
                error_msg,
                response.status_code,
                error_category="client_error"
            )

    def _make_request_with_retry(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make an API request with comprehensive error handling and retry mechanisms

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            PolygonAPIError: For API-related errors
            RetryError: When all retry attempts are exhausted
            CircuitBreakerOpenError: When circuit breaker is open
        """
        # Apply rate limiting (no-op if disabled)
        self.rate_limiter.wait_if_needed()

        # Prepare request
        url: str = urljoin(self.base_url, endpoint)
        params = params or {}
        params["apikey"] = self.api_key

        # Create a retryable function with proper configuration
        @retry(config=API_RETRY_CONFIG, circuit_breaker=self.circuit_breaker)
        def _execute_request() -> Dict[str, Any]:
            return self._make_single_request(url, params)

        try:
            return _execute_request()
        except RetryError as e:
            # Wrap retry exhaustion in a more descriptive error
            raise PolygonAPIError(
                f"Request to {endpoint} failed after {e.attempts} attempts. "
                f"Last error: {e.last_exception}",
                error_category="network"
            ) from e
        except CircuitBreakerOpenError as e:
            # Circuit breaker is open - service is failing
            raise PolygonAPIError(
                f"Circuit breaker is open for {endpoint} - service temporarily unavailable",
                error_category="server_error"
            ) from e

    # Alias for backward compatibility
    _make_request = _make_request_with_retry

    def _fetch_paginated_data(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Fetch all data from a paginated endpoint

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            List of all results from all pages
        """
        all_results: List[Dict[str, Any]] = []
        next_url: Optional[str] = None
        page_count: int = 0

        while True:
            page_count += 1
            if page_count > 1:
                logger.info(f"Fetching page {page_count} from {endpoint}")

            if next_url:
                # Parse next_url to extract parameters
                parsed_url: Any = urlparse(next_url)
                next_params: Dict[str, List[str]] = parse_qs(parsed_url.query)

                # Convert query parameters back to single values
                request_params: Dict[str, Any] = {
                    k: v[0] if isinstance(v, list) and len(v) == 1 else v
                    for k, v in next_params.items()
                }

                # Make request to next page
                response: Dict[str, Any] = self._make_request(parsed_url.path, request_params)
            else:
                # First page
                response = self._make_request(endpoint, params)

            # Extract results
            if "results" in response and response["results"]:
                results: List[Dict[str, Any]] = response["results"]
                all_results.extend(results)
                logger.info(
                    f"Page {page_count}: fetched {len(results)} records. Total: {len(all_results)}"
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

    def _fetch_paginated_data_stream(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Fetch data from a paginated endpoint as a streaming generator

        This method yields chunks of data instead of loading all results into memory,
        significantly reducing memory usage for large datasets.

        Args:
            endpoint: API endpoint path
            params: Query parameters
            chunk_size: Number of records to yield per chunk

        Yields:
            List of results (chunk) from each page until all data is processed
        """
        next_url: Optional[str] = None
        page_count: int = 0
        current_chunk: List[Dict[str, Any]] = []

        while True:
            page_count += 1
            if page_count > 1:
                logger.debug(f"Streaming page {page_count} from {endpoint}")

            if next_url:
                # Parse next_url to extract parameters
                parsed_url: Any = urlparse(next_url)
                next_params: Dict[str, List[str]] = parse_qs(parsed_url.query)

                # Convert query parameters back to single values
                request_params: Dict[str, Any] = {
                    k: v[0] if isinstance(v, list) and len(v) == 1 else v
                    for k, v in next_params.items()
                }

                # Make request to next page
                response: Dict[str, Any] = self._make_request(parsed_url.path, request_params)
            else:
                # First page
                response = self._make_request(endpoint, params)

            # Extract results
            if "results" in response and response["results"]:
                results: List[Dict[str, Any]] = response["results"]
                current_chunk.extend(results)

                # Yield chunks when we reach the chunk size
                while len(current_chunk) >= chunk_size:
                    yield current_chunk[:chunk_size]
                    current_chunk = current_chunk[chunk_size:]

                logger.debug(
                    f"Page {page_count}: processed {len(results)} records. "
                    f"Chunk buffer: {len(current_chunk)}"
                )
            else:
                logger.debug(f"Page {page_count}: no results found")
                break

            # Check for next page
            if "next_url" not in response or not response["next_url"]:
                logger.debug(
                    f"Pagination complete. Total pages: {page_count}"
                )
                break

            next_url = response["next_url"]

        # Yield any remaining records in the final chunk
        if current_chunk:
            yield current_chunk

    def get_dividends(
        self,
        ticker: str,
        order: str = "desc",
        limit: int = 1000,
        sort: str = "ex_dividend_date"
    ) -> List[Union[DividendRecord, Dict[str, Any]]]:
        """
        Get dividends for a single ticker. Polygon's dividends endpoint only supports
        querying one ticker at a time, so this method requires a ticker parameter.

        Args:
            ticker: Stock ticker symbol (required)
            order: 'asc' or 'desc' ordering of results (default 'desc')
            limit: Number of results per page (capped to API limits)
            sort: Field to sort by (default 'ex_dividend_date')

        Returns:
            List of dividend records as returned by Polygon
        """
        if not ticker:
            raise ValueError("ticker is required for get_dividends")

        params: Dict[str, Any] = {"ticker": ticker, "order": order, "limit": min(limit, 1000), "sort": sort}
        return cast(List[Union[DividendRecord, Dict[str, Any]]], self._fetch_paginated_data("/v3/reference/dividends", params))

    def get_tickers(
        self,
        market: str = "stocks",
        active: bool = True,
        limit: int = 1000,
        **kwargs: Any
    ) -> List[Union[TickerInfo, Dict[str, Any]]]:
        """
        Get list of tickers from Polygon.io

        Args:
            market: Market type (stocks, crypto, fx, etc.)
            active: Whether to include only active tickers
            limit: Number of results per page
            **kwargs: Additional query parameters

        Returns:
            List of ticker information
        """
        params: Dict[str, Any] = {
            "market": market,
            "active": str(active).lower(),
            "limit": min(limit, 1000),  # API maximum
            "sort": "ticker",
            **kwargs,
        }

        logger.info(f"Fetching {market} tickers (active={active})")
        return cast(List[Union[TickerInfo, Dict[str, Any]]], self._fetch_paginated_data("/v3/reference/tickers", params))

    def get_aggregates(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        from_date: str,
        to_date: str,
        limit: int = 50000,
    ) -> List[Union[OHLCVRecord, Dict[str, Any]]]:
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
        endpoint: str = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"

        params: Dict[str, Any] = {"adjusted": "true", "sort": "asc", "limit": min(limit, 50000)}

        logger.info(f"Fetching {timespan} aggregates for {ticker} from {from_date} to {to_date}")

        response = self._make_request(endpoint, params)
        return cast(List[Union[OHLCVRecord, Dict[str, Any]]], response.get("results", []))

    def get_grouped_daily(self, date: str) -> List[Union[OHLCVRecord, Dict[str, Any]]]:
        """Get grouped daily (OHLCV) data for all stocks on a specific date"""
        endpoint: str = f"/v2/aggs/grouped/locale/us/market/stocks/{date}"

        params: Dict[str, str] = {"adjusted": "true"}

        logger.info(f"Fetching grouped daily data for {date}")

        response = self._make_request(endpoint, params)
        return cast(List[Union[OHLCVRecord, Dict[str, Any]]], response.get("results", []))

    def get_ticker_details(self, ticker: str) -> Union[TickerInfo, Dict[str, Any]]:
        """
        Get detailed information about a ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            Ticker details
        """
        endpoint: str = f"/v3/reference/tickers/{ticker}"

        logger.info(f"Fetching details for ticker {ticker}")

        response = self._make_request(endpoint)
        return cast(Union[TickerInfo, Dict[str, Any]], response.get("results", {}))

    def get_dividends_stream(
        self,
        ticker: str,
        order: str = "desc",
        limit: int = 1000,
        chunk_size: int = 1000,
        **kwargs: Any
    ) -> Generator[List[Union[DividendRecord, Dict[str, Any]]], None, None]:
        """
        Get dividend data for a ticker as a streaming generator

        This method yields chunks of dividend data instead of loading all results into memory,
        making it suitable for processing large amounts of historical dividend data.

        Args:
            ticker: Stock ticker symbol
            order: Sort order (asc or desc)
            limit: Maximum number of results per page
            chunk_size: Number of records to yield per chunk
            **kwargs: Additional query parameters

        Yields:
            Chunks of dividend records
        """
        if not ticker:
            raise ValueError("ticker is required for get_dividends_stream")

        params: Dict[str, Any] = {"ticker": ticker, "order": order, "limit": min(limit, 1000), **kwargs}

        logger.info(f"Streaming dividends for {ticker} (order={order})")

        for chunk in self._fetch_paginated_data_stream("/v3/reference/dividends", params, chunk_size):
            yield cast(List[Union[DividendRecord, Dict[str, Any]]], chunk)

    def get_tickers_stream(
        self,
        market: str = "stocks",
        active: bool = True,
        limit: int = 1000,
        chunk_size: int = 1000,
        **kwargs: Any
    ) -> Generator[List[Union[TickerInfo, Dict[str, Any]]], None, None]:
        """
        Get list of tickers as a streaming generator

        This method yields chunks of ticker data instead of loading all results into memory,
        making it suitable for processing large ticker lists.

        Args:
            market: Market type (stocks, crypto, fx, etc.)
            active: Whether to include only active tickers
            limit: Number of results per page
            chunk_size: Number of records to yield per chunk
            **kwargs: Additional query parameters

        Yields:
            Chunks of ticker information
        """
        params: Dict[str, Any] = {
            "market": market,
            "active": str(active).lower(),
            "limit": min(limit, 1000),  # API maximum
            "sort": "ticker",
            **kwargs,
        }

        logger.info(f"Streaming {market} tickers (active={active})")

        for chunk in self._fetch_paginated_data_stream("/v3/reference/tickers", params, chunk_size):
            yield cast(List[Union[TickerInfo, Dict[str, Any]]], chunk)

    def process_stream_with_callback(
        self,
        stream_generator: Generator[List[Dict[str, Any]], None, None],
        callback: Any,
        **callback_kwargs: Any
    ) -> int:
        """
        Process a streaming data generator with a callback function

        This utility method allows processing large datasets in chunks without loading
        everything into memory. The callback function is called for each chunk.

        Args:
            stream_generator: Generator yielding data chunks
            callback: Function to call for each chunk (receives chunk and **callback_kwargs)
            **callback_kwargs: Additional keyword arguments to pass to callback

        Returns:
            Total number of records processed
        """
        total_processed = 0

        for chunk in stream_generator:
            if chunk:
                callback(chunk, **callback_kwargs)
                total_processed += len(chunk)
                logger.debug(f"Processed chunk of {len(chunk)} records. Total: {total_processed}")

        logger.info(f"Stream processing complete. Total records processed: {total_processed}")
        return total_processed

    def health_check(self) -> bool:
        """Perform a health check on the API connection"""
        try:
            # Simple request to check API connectivity
            response = self._make_request("/v3/reference/tickers", {"limit": 1})
            return "results" in response
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def __enter__(self) -> "PolygonDataClient":
        """Context manager entry"""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - cleanup session"""
        self.session.close()

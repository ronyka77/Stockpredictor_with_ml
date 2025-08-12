"""
Polygon.io API client with comprehensive error handling and retry mechanisms
"""

import time
import requests
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse, parse_qs

from src.utils.logger import get_polygon_logger
from src.data_collector.polygon_data.rate_limiter import AdaptiveRateLimiter
from src.data_collector.config import config

logger = get_polygon_logger(__name__)


class PolygonAPIError(Exception):
    """Custom exception for Polygon API errors"""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.response_data = response_data
        super().__init__(self.message)


class PolygonDataClient:
    """
    Main client for interacting with Polygon.io API
    
    Provides comprehensive error handling, rate limiting, and retry mechanisms
    for reliable data acquisition from Polygon.io.
    """
    
    def __init__(self, api_key: Optional[str] = None, requests_per_minute: int = 5):
        """
        Initialize the Polygon.io client
        
        Args:
            api_key: Polygon.io API key (defaults to config)
            requests_per_minute: Rate limit for API requests
        """
        self.api_key = api_key or config.API_KEY
        self.base_url = config.BASE_URL
        self.rate_limiter = AdaptiveRateLimiter(requests_per_minute)
        self.session = requests.Session()
        
        # Set up session headers
        self.session.headers.update({
            'User-Agent': 'StockPredictor/1.0',
            'Accept': 'application/json'
        })
        
        logger.info(f"Polygon client initialized with {requests_per_minute} requests/minute limit")
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
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
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Prepare request
        url = urljoin(self.base_url, endpoint)
        params = params or {}
        params['apikey'] = self.api_key
        
        # Retry logic with exponential backoff
        for attempt in range(config.MAX_RETRIES):
            try:
                logger.info(f"Making request to {endpoint} (attempt {attempt + 1}/{config.MAX_RETRIES})")
                
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=config.REQUEST_TIMEOUT
                )
                
                # Handle different response status codes
                if response.status_code == 200:
                    self.rate_limiter.handle_successful_request()
                    data = response.json()
                    
                    # Check for API-level errors in response
                    if data.get('status') == 'ERROR':
                        error_msg = data.get('error', 'Unknown API error')
                        raise PolygonAPIError(f"API Error: {error_msg}", response.status_code, data)
                    
                    logger.info(f"Successful request to {endpoint}")
                    return data
                
                elif response.status_code == 429:  # Rate limit exceeded
                    self.rate_limiter.handle_rate_limit_error()
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit exceeded. Waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code == 401:  # Unauthorized
                    raise PolygonAPIError("Invalid API key", response.status_code)
                
                elif response.status_code == 403:  # Forbidden
                    raise PolygonAPIError("Access forbidden - check subscription level", response.status_code)
                
                elif response.status_code >= 500:  # Server errors
                    wait_time = 2 ** attempt
                    logger.warning(f"Server error {response.status_code}. Retrying in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                
                else:
                    # Other client errors
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('error', f"HTTP {response.status_code}")
                    except Exception as e:
                        logger.error(f"Error parsing response: {e}")
                        error_msg = f"HTTP {response.status_code}"
                    
                    raise PolygonAPIError(error_msg, response.status_code)
                
            except requests.exceptions.Timeout:
                wait_time = 2 ** attempt
                logger.warning(f"Request timeout. Retrying in {wait_time}s")
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(wait_time)
                    continue
                raise PolygonAPIError("Request timeout after all retries")
            
            except requests.exceptions.ConnectionError:
                wait_time = 2 ** attempt
                logger.warning(f"Connection error. Retrying in {wait_time}s")
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(wait_time)
                    continue
                raise PolygonAPIError("Connection error after all retries")
            
            except requests.exceptions.RequestException as e:
                if attempt == config.MAX_RETRIES - 1:
                    raise PolygonAPIError(f"Request failed: {str(e)}")
                
                wait_time = 2 ** attempt
                logger.warning(f"Request failed: {e}. Retrying in {wait_time}s")
                time.sleep(wait_time)
        
        raise PolygonAPIError(f"Failed to complete request after {config.MAX_RETRIES} attempts")
    
    def _fetch_paginated_data(self, endpoint: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Fetch all data from a paginated endpoint
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            List of all results from all pages
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
                request_params = {k: v[0] if isinstance(v, list) and len(v) == 1 else v 
                                for k, v in next_params.items()}
                
                # Make request to next page
                response = self._make_request(parsed_url.path, request_params)
            else:
                # First page
                response = self._make_request(endpoint, params)
            
            # Extract results
            if 'results' in response and response['results']:
                results = response['results']
                all_results.extend(results)
                logger.info(f"Page {page_count}: fetched {len(results)} records. "
                           f"Total: {len(all_results)}")
            else:
                logger.info(f"Page {page_count}: no results found")
                break
            
            # Check for next page
            if 'next_url' not in response or not response['next_url']:
                logger.info(f"Pagination complete. Total pages: {page_count}, "
                           f"Total records: {len(all_results)}")
                break
            
            next_url = response['next_url']
        
        return all_results
    
    def get_tickers(self, market: str = "stocks", active: bool = True, 
                   limit: int = 1000, **kwargs) -> List[Dict]:
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
        params = {
            'market': market,
            'active': str(active).lower(),
            'limit': min(limit, 1000),  # API maximum
            'sort': 'ticker',
            **kwargs
        }
        
        logger.info(f"Fetching {market} tickers (active={active})")
        return self._fetch_paginated_data('/v3/reference/tickers', params)
    
    def get_aggregates(self, ticker: str, multiplier: int, timespan: str,
                        from_date: str, to_date: str, adjusted: bool = True,
                        sort: str = "asc", limit: int = 50000) -> List[Dict]:
        """
        Get aggregate (OHLCV) data for a ticker
        
        Args:
            ticker: Stock ticker symbol
            multiplier: Size of the timespan multiplier
            timespan: Size of the time window (minute, hour, day, week, month, quarter, year)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            adjusted: Whether to return adjusted data
            sort: Sort order (asc or desc)
            limit: Number of results to return
            
        Returns:
            List of OHLCV records
        """
        endpoint = f'/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}'
        
        params = {
            'adjusted': str(adjusted).lower(),
            'sort': sort,
            'limit': min(limit, 50000)  # API maximum
        }
        
        logger.info(f"Fetching {timespan} aggregates for {ticker} from {from_date} to {to_date}")
        
        response = self._make_request(endpoint, params)
        return response.get('results', [])
    
    def get_grouped_daily(self, date: str, adjusted: bool = True) -> List[Dict]:
        """
        Get grouped daily (OHLCV) data for all stocks on a specific date
        
        Args:
            date: Date in YYYY-MM-DD format
            adjusted: Whether to return adjusted data
            
        Returns:
            List of OHLCV records for all stocks
        """
        endpoint = f'/v2/aggs/grouped/locale/us/market/stocks/{date}'
        
        params = {
            'adjusted': str(adjusted).lower()
        }
        
        logger.info(f"Fetching grouped daily data for {date}")
        
        response = self._make_request(endpoint, params)
        return response.get('results', [])
    
    def get_ticker_details(self, ticker: str) -> Dict:
        """
        Get detailed information about a ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Ticker details
        """
        endpoint = f'/v3/reference/tickers/{ticker}'
        
        logger.info(f"Fetching details for ticker {ticker}")
        
        response = self._make_request(endpoint)
        return response.get('results', {})
    
    def health_check(self) -> bool:
        """
        Perform a health check on the API connection
        
        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Simple request to check API connectivity
            response = self._make_request('/v3/reference/tickers', {'limit': 1})
            return 'results' in response
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Get current rate limit status
        
        Returns:
            Dictionary with rate limit information
        """
        return {
            'requests_per_minute': self.rate_limiter.requests_per_minute,
            'remaining_requests': self.rate_limiter.get_remaining_requests(),
            'time_until_reset': self.rate_limiter.get_time_until_reset(),
            'consecutive_errors': getattr(self.rate_limiter, 'consecutive_errors', 0)
        }
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup session"""
        self.session.close() 
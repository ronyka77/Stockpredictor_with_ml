"""
Polygon Fundamentals API Client

This module provides a client for fetching fundamental financial data
from the Polygon API with proper rate limiting and error handling.
"""

import asyncio
import aiohttp
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import json
from pathlib import Path

from src.data_collector.polygon_fundamentals.config import PolygonFundamentalsConfig, polygon_fundamentals_config
from src.data_collector.polygon_fundamentals.data_models import (
    FundamentalDataResponse,
    IncomeStatement,
    BalanceSheet,
    CashFlowStatement,
    FinancialValue
)
from src.utils.logger import get_logger
logger = get_logger(__name__)

class RateLimiter:
    """Simple rate limiter for API requests"""
    
    def __init__(self, requests_per_minute: int = 5):
        self.requests_per_minute = requests_per_minute
        self.requests = []
    
    async def wait_if_needed(self):
        """Wait if we've exceeded the rate limit"""
        now = time.time()
        
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        # If we've hit the limit, wait
        if len(self.requests) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.requests[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, waiting {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        # Record this request
        self.requests.append(now)

class PolygonFundamentalsClient:
    """Client for fetching fundamental data from Polygon API"""
    
    def __init__(self, config: Optional[PolygonFundamentalsConfig] = None):
        self.config = config or polygon_fundamentals_config
        self.rate_limiter = RateLimiter(self.config.REQUESTS_PER_MINUTE)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Setup caching if enabled
        if self.config.CACHE_ENABLED:
            self.cache_dir = Path(self.config.CACHE_DIR)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(
            total=self.config.REQUEST_TIMEOUT,
            connect=self.config.CONNECTION_TIMEOUT
        )
        self.session = aiohttp.ClientSession(
            headers=self.config.headers,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _get_cache_path(self, ticker: str, endpoint: str) -> Path:
        """Get cache file path for a ticker and endpoint"""
        return self.cache_dir / f"{ticker}_{endpoint}_{datetime.now().strftime('%Y%m%d')}.json"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is still valid"""
        if not cache_path.exists():
            return False
        
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < timedelta(hours=self.config.CACHE_DURATION_HOURS)
    
    async def _load_from_cache(self, cache_path: Path) -> Optional[Dict[str, Any]]:
        """Load data from cache file"""
        try:
            if self._is_cache_valid(cache_path):
                with open(cache_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}")
        return None
    
    async def _save_to_cache(self, cache_path: Path, data: Dict[str, Any]):
        """Save data to cache file"""
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache to {cache_path}: {e}")
    
    async def _make_request(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make HTTP request with retry logic"""
        if not self.session:
            raise RuntimeError("Client session not initialized. Use async context manager.")
        
        await self.rate_limiter.wait_if_needed()
        
        for attempt in range(self.config.RETRY_ATTEMPTS):
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    elif response.status == 429:  # Rate limited
                        wait_time = self.config.RETRY_DELAY * (self.config.BACKOFF_FACTOR ** attempt)
                        logger.warning(f"Rate limited, waiting {wait_time} seconds")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"HTTP {response.status}: {await response.text()}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
            
            if attempt < self.config.RETRY_ATTEMPTS - 1:
                wait_time = self.config.RETRY_DELAY * (self.config.BACKOFF_FACTOR ** attempt)
                await asyncio.sleep(wait_time)
        
        return None
    
    async def get_financials(self, ticker: str, **kwargs) -> Optional[FundamentalDataResponse]:
        """
        Get financial statements for a ticker
        
        Args:
            ticker: Stock ticker symbol
            **kwargs: Additional parameters (timeframe, start_date, end_date, etc.)
        
        Returns:
            FundamentalDataResponse object or None if failed
        """
        # Check cache first
        cache_path = self._get_cache_path(ticker, "financials")
        if self.config.CACHE_ENABLED:
            cached_data = await self._load_from_cache(cache_path)
            if cached_data:
                logger.info(f"Loading {ticker} financials from cache")
                return self._parse_financial_response(cached_data, ticker)
        
        # Make API request
        url = self.config.get_financials_url(ticker)
        params = self.config.get_request_params(ticker, **kwargs)
        
        logger.info(f"Fetching financials for {ticker}")
        data = await self._make_request(url, params)
        
        if not data:
            logger.error(f"Failed to fetch financials for {ticker}")
            return None
        
        # Save to cache
        if self.config.CACHE_ENABLED:
            await self._save_to_cache(cache_path, data)
        
        return self._parse_financial_response(data, ticker)
    
    def _parse_financial_response(self, data: Dict[str, Any], ticker: str) -> FundamentalDataResponse:
        """Parse Polygon financial data response"""
        response = FundamentalDataResponse(
            status=data.get('status', 'OK'),
            request_id=data.get('request_id'),
            count=data.get('count'),
            next_url=data.get('next_url')
        )
        
        try:
            # Handle the new nested financials format
            results = data.get('results', [])
            if not results:
                logger.warning(f"No results found in response for {ticker}")
                return response
            
            # Convert new format to legacy format for existing parsing logic
            legacy_data = {}
            
            for result in results:
                financials = result.get('financials', {})
                
                # Parse income statement
                if 'income_statement' in financials:
                    if 'income_statements' not in legacy_data:
                        legacy_data['income_statements'] = []
                    
                    # Map new field names to expected names
                    stmt = self._map_income_statement_fields(financials['income_statement'])
                    stmt.update({
                        'start_date': result.get('start_date'),
                        'end_date': result.get('end_date'),
                        'filing_date': result.get('filing_date'),
                        'timeframe': result.get('timeframe'),
                        'fiscal_period': result.get('fiscal_period'),
                        'fiscal_year': result.get('fiscal_year'),
                        'cik': result.get('cik'),
                        'company_name': result.get('company_name'),
                        'ticker': ticker,
                        'source_filing_url': result.get('source_filing_url'),
                        'source_filing_file_url': result.get('source_filing_file_url')
                    })
                    legacy_data['income_statements'].append(stmt)
                
                # Parse balance sheet
                if 'balance_sheet' in financials:
                    if 'balance_sheets' not in legacy_data:
                        legacy_data['balance_sheets'] = []
                    
                    # Map new field names to expected names
                    stmt = self._map_balance_sheet_fields(financials['balance_sheet'])
                    stmt.update({
                        'start_date': result.get('start_date'),
                        'end_date': result.get('end_date'),
                        'filing_date': result.get('filing_date'),
                        'timeframe': result.get('timeframe'),
                        'fiscal_period': result.get('fiscal_period'),
                        'fiscal_year': result.get('fiscal_year'),
                        'cik': result.get('cik'),
                        'company_name': result.get('company_name'),
                        'ticker': ticker,
                        'source_filing_url': result.get('source_filing_url'),
                        'source_filing_file_url': result.get('source_filing_file_url')
                    })
                    legacy_data['balance_sheets'].append(stmt)
                
                # Parse cash flow statement
                if 'cash_flow_statement' in financials:
                    if 'cash_flow_statements' not in legacy_data:
                        legacy_data['cash_flow_statements'] = []
                    
                    # Map new field names to expected names
                    stmt = self._map_cash_flow_fields(financials['cash_flow_statement'])
                    stmt.update({
                        'start_date': result.get('start_date'),
                        'end_date': result.get('end_date'),
                        'filing_date': result.get('filing_date'),
                        'timeframe': result.get('timeframe'),
                        'fiscal_period': result.get('fiscal_period'),
                        'fiscal_year': result.get('fiscal_year'),
                        'cik': result.get('cik'),
                        'company_name': result.get('company_name'),
                        'ticker': ticker,
                        'source_filing_url': result.get('source_filing_url'),
                        'source_filing_file_url': result.get('source_filing_file_url')
                    })
                    legacy_data['cash_flow_statements'].append(stmt)
            
            return self._parse_direct_list_format(legacy_data, ticker, response)
        
        except Exception as e:
            logger.error(f"Failed to parse financial response for {ticker}: {e}")
            return response

    def _map_income_statement_fields(self, income_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map new API field names to expected field names for income statement"""
        field_mapping = {
            # New API field -> Expected field name
            'basic_average_shares': 'weighted_average_shares_outstanding',
            'diluted_average_shares': 'weighted_average_shares_outstanding_diluted',
            'basic_earnings_per_share': 'earnings_per_share_basic',
            'diluted_earnings_per_share': 'earnings_per_share_diluted',
            'income_loss_from_continuing_operations_before_tax': 'income_loss_before_income_tax_expense_benefit',
            # Keep existing field names that match
            'revenues': 'revenues',
            'cost_of_revenue': 'cost_of_revenue',
            'gross_profit': 'gross_profit',
            'operating_expenses': 'operating_expenses',
            'selling_general_and_administrative_expenses': 'selling_general_and_administrative_expenses',
            'research_and_development': 'research_and_development',
            'operating_income_loss': 'operating_income_loss',
            'interest_expense': 'interest_expense',
            'interest_income': 'interest_income',
            'income_tax_expense_benefit': 'income_tax_expense_benefit',
            'net_income_loss': 'net_income_loss',
            'net_income_loss_attributable_to_parent': 'net_income_loss_attributable_to_parent',
        }
        
        mapped_data = {}
        for api_field, expected_field in field_mapping.items():
            if api_field in income_data:
                mapped_data[expected_field] = income_data[api_field]
        
        # Copy any unmapped fields as-is
        for field, value in income_data.items():
            if field not in field_mapping and field not in mapped_data:
                mapped_data[field] = value
        
        return mapped_data

    def _map_balance_sheet_fields(self, balance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map new API field names to expected field names for balance sheet"""
        field_mapping = {
            # New API field -> Expected field name
            'inventory': 'inventory_net',
            'fixed_assets': 'property_plant_equipment_net',
            'long_term_debt': 'long_term_debt_noncurrent',
            'accounts_payable': 'accounts_payable_current',
            'other_current_assets': 'other_assets_current',
            'other_noncurrent_assets': 'other_assets_noncurrent',
            'other_current_liabilities': 'other_liabilities_current',
            'other_noncurrent_liabilities': 'other_liabilities_noncurrent',
            # Keep existing field names that match
            'assets': 'assets',
            'current_assets': 'current_assets',
            'noncurrent_assets': 'noncurrent_assets',
            'liabilities': 'liabilities',
            'current_liabilities': 'current_liabilities',
            'noncurrent_liabilities': 'noncurrent_liabilities',
            'equity': 'equity',
            'equity_attributable_to_parent': 'equity_attributable_to_parent',
        }
        
        mapped_data = {}
        for api_field, expected_field in field_mapping.items():
            if api_field in balance_data:
                mapped_data[expected_field] = balance_data[api_field]
        
        # Copy any unmapped fields as-is
        for field, value in balance_data.items():
            if field not in field_mapping and field not in mapped_data:
                mapped_data[field] = value
        
        return mapped_data

    def _map_cash_flow_fields(self, cash_flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map new API field names to expected field names for cash flow statement"""
        field_mapping = {
            # Most cash flow fields appear to match existing names
            'net_cash_flow_from_operating_activities': 'net_cash_flow_from_operating_activities',
            'net_cash_flow_from_investing_activities': 'net_cash_flow_from_investing_activities', 
            'net_cash_flow_from_financing_activities': 'net_cash_flow_from_financing_activities',
            'net_cash_flow': 'net_cash_flow',
        }
        
        mapped_data = {}
        for api_field, expected_field in field_mapping.items():
            if api_field in cash_flow_data:
                mapped_data[expected_field] = cash_flow_data[api_field]
        
        # Copy any unmapped fields as-is
        for field, value in cash_flow_data.items():
            if field not in field_mapping and field not in mapped_data:
                mapped_data[field] = value
        
        return mapped_data
    
    def _parse_direct_list_format(self, data: Dict[str, Any], ticker: str, response: FundamentalDataResponse) -> FundamentalDataResponse:
        """Parse the current direct list format from API response"""
        try:
            # Parse income statements
            income_statements = data.get('income_statements', [])
            for stmt_data in income_statements:
                stmt = self._parse_income_statement_direct(stmt_data)
                if stmt:
                    response.income_statements.append(stmt)
            
            # Parse balance sheets
            balance_sheets = data.get('balance_sheets', [])
            for stmt_data in balance_sheets:
                stmt = self._parse_balance_sheet_direct(stmt_data)
                if stmt:
                    response.balance_sheets.append(stmt)
            
            # Parse cash flow statements
            cash_flow_statements = data.get('cash_flow_statements', [])
            for stmt_data in cash_flow_statements:
                stmt = self._parse_cash_flow_statement_direct(stmt_data)
                if stmt:
                    response.cash_flow_statements.append(stmt)
            
            # Set data quality metrics if available
            response.data_quality_score = data.get('data_quality_score')
            response.missing_data_count = data.get('missing_data_count')
            response.total_fields_count = data.get('total_fields_count')
            
            logger.info(f"Parsed direct format: {len(response.income_statements)} income statements, "
                       f"{len(response.balance_sheets)} balance sheets, "
                       f"{len(response.cash_flow_statements)} cash flow statements for {ticker}")
            
        except Exception as e:
            logger.error(f"Failed to parse direct list format for {ticker}: {e}")
        
        return response
    
    def _parse_income_statement_direct(self, stmt_data: Dict[str, Any]) -> Optional[IncomeStatement]:
        """Parse income statement from direct API format"""
        try:
            # Convert all financial fields to FinancialValue objects
            parsed_data = {}
            
            # Copy basic metadata fields
            metadata_fields = [
                'start_date', 'end_date', 'filing_date', 'period_of_report_date',
                'timeframe', 'fiscal_period', 'fiscal_year', 'cik', 'company_name',
                'ticker', 'source_filing_url', 'source_filing_file_url'
            ]
            
            for field in metadata_fields:
                if field in stmt_data:
                    parsed_data[field] = stmt_data[field]
            
            # Parse financial value fields
            financial_fields = [
                'revenues', 'cost_of_revenue', 'gross_profit', 'operating_expenses',
                'selling_general_and_administrative_expenses', 'research_and_development',
                'operating_income_loss', 'interest_expense', 'interest_income',
                'other_income_expense', 'income_loss_before_income_tax_expense_benefit',
                'income_tax_expense_benefit', 'net_income_loss', 'net_income_loss_attributable_to_parent',
                'earnings_per_share_basic', 'earnings_per_share_diluted',
                'weighted_average_shares_outstanding', 'weighted_average_shares_outstanding_diluted',
                'comprehensive_income_loss', 'comprehensive_income_loss_attributable_to_parent'
            ]
            
            for field in financial_fields:
                if field in stmt_data and stmt_data[field] is not None:
                    if isinstance(stmt_data[field], dict):
                        parsed_data[field] = FinancialValue(**stmt_data[field])
                    else:
                        parsed_data[field] = FinancialValue(value=float(stmt_data[field]))
            
            return IncomeStatement(**parsed_data)
            
        except Exception as e:
            logger.error(f"Failed to parse income statement direct format: {e}")
            return None
    
    def _parse_balance_sheet_direct(self, stmt_data: Dict[str, Any]) -> Optional[BalanceSheet]:
        """Parse balance sheet from direct API format"""
        try:
            # Convert all financial fields to FinancialValue objects
            parsed_data = {}
            
            # Copy basic metadata fields
            metadata_fields = [
                'start_date', 'end_date', 'filing_date', 'period_of_report_date',
                'timeframe', 'fiscal_period', 'fiscal_year', 'cik', 'company_name',
                'ticker', 'source_filing_url', 'source_filing_file_url'
            ]
            
            for field in metadata_fields:
                if field in stmt_data:
                    parsed_data[field] = stmt_data[field]
            
            # Parse financial value fields
            financial_fields = [
                'assets', 'current_assets', 'noncurrent_assets',
                'cash_and_cash_equivalents_at_carrying_value', 'short_term_investments',
                'accounts_receivable_net_current', 'inventory_net', 'prepaid_expenses_current',
                'other_assets_current', 'property_plant_equipment_net', 'goodwill',
                'intangible_assets_net_excluding_goodwill', 'long_term_investments',
                'other_assets_noncurrent', 'liabilities', 'current_liabilities',
                'noncurrent_liabilities', 'accounts_payable_current', 'accrued_liabilities_current',
                'short_term_debt', 'other_liabilities_current', 'long_term_debt_noncurrent',
                'other_liabilities_noncurrent', 'equity', 'equity_attributable_to_parent',
                'stockholders_equity', 'common_stock_shares_outstanding',
                'common_stock_par_or_stated_value_per_share', 'additional_paid_in_capital',
                'retained_earnings_accumulated_deficit', 'accumulated_other_comprehensive_income_loss',
                'treasury_stock_value'
            ]
            
            for field in financial_fields:
                if field in stmt_data and stmt_data[field] is not None:
                    if isinstance(stmt_data[field], dict):
                        parsed_data[field] = FinancialValue(**stmt_data[field])
                    else:
                        parsed_data[field] = FinancialValue(value=float(stmt_data[field]))
            
            return BalanceSheet(**parsed_data)
            
        except Exception as e:
            logger.error(f"Failed to parse balance sheet direct format: {e}")
            return None
    
    def _parse_cash_flow_statement_direct(self, stmt_data: Dict[str, Any]) -> Optional[CashFlowStatement]:
        """Parse cash flow statement from direct API format"""
        try:
            # Convert all financial fields to FinancialValue objects
            parsed_data = {}
            
            # Copy basic metadata fields
            metadata_fields = [
                'start_date', 'end_date', 'filing_date', 'period_of_report_date',
                'timeframe', 'fiscal_period', 'fiscal_year', 'cik', 'company_name',
                'ticker', 'source_filing_url', 'source_filing_file_url'
            ]
            
            for field in metadata_fields:
                if field in stmt_data:
                    parsed_data[field] = stmt_data[field]
            
            # Parse financial value fields
            financial_fields = [
                'net_cash_flow_from_operating_activities', 'net_income_loss',
                'depreciation_depletion_and_amortization', 'stock_based_compensation',
                'increase_decrease_in_accounts_receivable', 'increase_decrease_in_inventory',
                'increase_decrease_in_accounts_payable', 'increase_decrease_in_other_operating_assets_liabilities',
                'net_cash_flow_from_investing_activities', 'payments_to_acquire_property_plant_and_equipment',
                'payments_to_acquire_businesses_net_of_cash_acquired', 'payments_to_acquire_investments',
                'proceeds_from_sale_of_investments', 'net_cash_flow_from_financing_activities',
                'proceeds_from_issuance_of_debt', 'repayments_of_debt', 'proceeds_from_issuance_of_common_stock',
                'payments_for_repurchase_of_common_stock', 'payments_of_dividends',
                'cash_and_cash_equivalents_at_beginning_of_period', 'cash_and_cash_equivalents_at_end_of_period',
                'net_cash_flow', 'free_cash_flow'
            ]
            
            for field in financial_fields:
                if field in stmt_data and stmt_data[field] is not None:
                    if isinstance(stmt_data[field], dict):
                        parsed_data[field] = FinancialValue(**stmt_data[field])
                    else:
                        parsed_data[field] = FinancialValue(value=float(stmt_data[field]))
            
            return CashFlowStatement(**parsed_data)
            
        except Exception as e:
            logger.error(f"Failed to parse cash flow statement direct format: {e}")
            return None
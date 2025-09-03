"""
Configuration for Polygon Fundamentals Data Collection

This module contains configuration settings specific to fundamental data collection
from the Polygon API.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
# Load environment variables from a .env file if present
load_dotenv()
@dataclass
class PolygonFundamentalsConfig:
    """Configuration for Polygon fundamentals API client"""
    
    # API Configuration
    API_KEY: str = os.getenv('POLYGON_API_KEY', '')
    BASE_URL: str = "https://api.polygon.io"
    FINANCIALS_ENDPOINT: str = "/vX/reference/financials"
    COMPANY_DETAILS_ENDPOINT: str = "/v3/reference/tickers"
    
    # Rate Limiting
    REQUESTS_PER_MINUTE: int = 5  # Conservative rate limit for fundamentals
    RETRY_ATTEMPTS: int = 3
    RETRY_DELAY: float = 15.0  # seconds
    BACKOFF_FACTOR: float = 2.0
    
    # Data Collection Parameters
    HISTORICAL_YEARS: int = 3
    FILING_TYPES: List[str] = None  # Will default to ['10-K', '10-Q']
    TIMEFRAME: str = "quarterly"  # quarterly or annual
    INCLUDE_SOURCES: bool = True
    
    # Data Quality
    MIN_REQUIRED_FIELDS: int = 10  # Minimum fields required for valid data
    MAX_MISSING_PERCENTAGE: float = 0.3  # 30% max missing data
    
    # Caching
    CACHE_ENABLED: bool = True
    CACHE_DURATION_HOURS: int = 24
    CACHE_DIR: str = "data/cache/fundamentals"
    
    # Timeouts
    REQUEST_TIMEOUT: int = 30  # seconds
    CONNECTION_TIMEOUT: int = 10  # seconds
    
    def __post_init__(self):
        """Initialize default values after dataclass creation"""
        if self.FILING_TYPES is None:
            self.FILING_TYPES = ['10-K', '10-Q']
    
    @property
    def headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests"""
        if not self.API_KEY:
            raise ValueError("POLYGON_API_KEY environment variable is required for API requests")

        return {
            'Authorization': f'Bearer {self.API_KEY}',
            'Content-Type': 'application/json',
            'User-Agent': 'StockPredictor-Fundamentals/1.0'
        }
    
    @property
    def start_date(self) -> str:
        """Get start date for historical data collection"""
        start = datetime(datetime.now().year - self.HISTORICAL_YEARS, 1, 1)
        return start.strftime('%Y-%m-%d')
    
    @property
    def end_date(self) -> str:
        """Get end date for data collection (today)"""
        return datetime.now().strftime('%Y-%m-%d')
    
    def get_financials_url(self, ticker: str) -> str:
        """Get the full URL for financials endpoint"""
        return f"{self.BASE_URL}{self.FINANCIALS_ENDPOINT}"
    
    def get_company_details_url(self, ticker: str) -> str:
        """Get the full URL for company details endpoint"""
        return f"{self.BASE_URL}{self.COMPANY_DETAILS_ENDPOINT}/{ticker}"
    
    def get_request_params(self, ticker: str, **kwargs) -> Dict[str, Any]:
        """Get standard request parameters for API calls"""
        params = {
            'ticker': ticker,
            'timeframe': kwargs.get('timeframe', self.TIMEFRAME),
            'filing_date.gte': kwargs.get('start_date', self.start_date),
            'filing_date.lte': kwargs.get('end_date', self.end_date),
            'include_sources': str(self.INCLUDE_SOURCES).lower(),
            'limit': kwargs.get('limit', 100),
            'sort': 'filing_date',
            'apikey': self.API_KEY
        }
        
        # Add filing types if specified
        if self.FILING_TYPES:
            params['filing_date.filing_type'] = ','.join(self.FILING_TYPES)
        
        return params

# Global configuration instance
polygon_fundamentals_config = PolygonFundamentalsConfig()

# Field mappings for different financial statement types
INCOME_STATEMENT_FIELDS = {
    'revenues': ['revenues'],
    'cost_of_revenue': ['cost_of_revenue'],
    'gross_profit': ['gross_profit'],
    'operating_expenses': ['operating_expenses'],
    'operating_income': ['operating_income_loss'],
    'interest_expense': ['interest_expense', 'interest_expense_operating'],
    'income_before_tax': ['income_loss_from_continuing_operations_before_tax'],
    'income_tax_expense': ['income_tax_expense_benefit'],
    'net_income': ['net_income_loss', 'net_income_loss_attributable_to_parent'],
    'basic_earnings_per_share': ['basic_earnings_per_share'],
    'diluted_earnings_per_share': ['diluted_earnings_per_share'],
    'weighted_average_shares': ['basic_average_shares'],
    'weighted_average_shares_diluted': ['diluted_average_shares'],
    'research_and_development': ['research_and_development'],
    'selling_general_administrative': ['selling_general_and_administrative_expenses']
}

BALANCE_SHEET_FIELDS = {
    'assets': ['assets'],
    'current_assets': ['current_assets'],
    'noncurrent_assets': ['noncurrent_assets'],
    'inventory': ['inventory'],  # Note: API uses 'inventory' not 'inventory_net'
    'property_plant_equipment': ['fixed_assets'],  # Note: API uses 'fixed_assets'
    'liabilities': ['liabilities'],
    'current_liabilities': ['current_liabilities'],
    'noncurrent_liabilities': ['noncurrent_liabilities'],
    'long_term_debt': ['long_term_debt'],  # Note: API uses 'long_term_debt' not 'long_term_debt_noncurrent'
    'equity': ['equity', 'equity_attributable_to_parent'],
    'accounts_payable': ['accounts_payable'],
    'other_assets_current': ['other_current_assets'],
    'other_assets_noncurrent': ['other_noncurrent_assets'],
    'other_liabilities_current': ['other_current_liabilities'],
    'other_liabilities_noncurrent': ['other_noncurrent_liabilities']
}

CASH_FLOW_FIELDS = {
    'operating_cash_flow': ['net_cash_flow_from_operating_activities'],
    'investing_cash_flow': ['net_cash_flow_from_investing_activities'],
    'financing_cash_flow': ['net_cash_flow_from_financing_activities'],
    'net_cash_flow': ['net_cash_flow'],
    'capital_expenditures': ['payments_to_acquire_property_plant_and_equipment'],
    'depreciation_amortization': ['depreciation_depletion_and_amortization']
}

# GICS Sector mapping for sector analysis
GICS_SECTORS = {
    '10': 'Energy',
    '15': 'Materials', 
    '20': 'Industrials',
    '25': 'Consumer Discretionary',
    '30': 'Consumer Staples',
    '35': 'Health Care',
    '40': 'Financials',
    '45': 'Information Technology',
    '50': 'Communication Services',
    '55': 'Utilities',
    '60': 'Real Estate'
}

# Data validation rules
VALIDATION_RULES = {
    'required_fields': {
        'income_statement': ['revenues', 'net_income_loss'],
        'balance_sheet': ['assets', 'liabilities', 'equity'],
        'cash_flow': ['net_cash_flow_from_operating_activities']
    },
    'logical_checks': {
        'assets_equals_liabilities_plus_equity': True,
        'revenues_positive': True,
        'current_ratio_reasonable': (0.1, 10.0),  # Between 0.1 and 10
        'debt_to_equity_reasonable': (0.0, 20.0)  # Between 0 and 20
    },
    'outlier_limits': {
        'pe_ratio': (-100, 100),
        'pb_ratio': (0, 50),
        'debt_to_equity': (0, 10),
        'current_ratio': (0, 20),
        'roe': (-1, 2),  # -100% to 200%
        'roa': (-1, 1)   # -100% to 100%
    }
} 
"""
Optimized Fundamental Data Collector

This module handles fundamental data collection with complete pipeline execution
per ticker and simplified rate limiting.
"""

from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timedelta, date
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.data_collector.polygon_fundamentals.client import PolygonFundamentalsClient
from src.data_collector.polygon_fundamentals.config import PolygonFundamentalsConfig
from src.database.connection import DatabaseConnection
from src.data_collector.polygon_data.data_storage import DataStorage
from src.data_collector.polygon_data.rate_limiter import AdaptiveRateLimiter
from src.utils.logger import get_logger
from src.data_collector.polygon_fundamentals.db_pool import get_connection_pool

logger = get_logger(__name__)

class FundamentalRateLimiter:
    """Wrapper for existing polygon rate limiter"""
    
    def __init__(self):
        self.rate_limiter = AdaptiveRateLimiter()
    
    async def acquire(self):
        """Acquire rate limit using existing framework"""
        # The AdaptiveRateLimiter.wait_if_needed() is synchronous, so we just call it
        self.rate_limiter.wait_if_needed()
    
    def release(self):
        """Release rate limit using existing framework"""
        # No release needed for the AdaptiveRateLimiter - it handles timing internally
        pass

class OptimizedFundamentalCollector:
    """Optimized collector with complete pipeline per ticker"""
    
    def __init__(self, config: Optional[PolygonFundamentalsConfig] = None):
        self.config = config or PolygonFundamentalsConfig()
        self.db_pool = get_connection_pool()
        
        # Create SQLAlchemy engine from connection parameters
        database_url = f"postgresql://{self.db_connection.config['user']}:{self.db_connection.config['password']}@{self.db_connection.config['host']}:{self.db_connection.config['port']}/{self.db_connection.config['database']}"
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Initialize data storage for ticker metadata
        self.data_storage = DataStorage()
        self.ticker_cache = {}  # Cache for ticker_id mapping
        self.existing_data_cache = set()  # Cache for existing ticker-date combinations
        
        # Rate limiting - use existing framework
        self.rate_limiter = FundamentalRateLimiter()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None
        }
    
    def _load_ticker_cache(self) -> Dict[str, int]:
        """Load all ticker metadata into cache"""
        try:
            logger.info("Loading ticker metadata into cache...")
            
            # Get tickers from database using DataStorage
            tickers_data = self.data_storage.get_tickers(
                market='stocks',
                active=True,
                limit=None
            )
            
            if tickers_data:
                ticker_cache = {}
                for ticker_data in tickers_data:
                    ticker = ticker_data.get('ticker')
                    ticker_id = ticker_data.get('id')
                    if ticker and ticker_id is not None:
                        ticker_cache[ticker] = int(ticker_id)
                
                logger.info(f"Loaded {len(ticker_cache)} tickers into cache")
                return ticker_cache
            else:
                logger.warning("No ticker metadata found")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to load ticker cache: {e}")
            return {}
    
    def _load_existing_data_cache(self):
        """Load existing ticker-date combinations to avoid re-collection"""
        try:
            logger.info("Loading existing data cache...")
            
            with self.SessionLocal() as session:
                existing = session.execute(text("""
                    SELECT DISTINCT ticker_id, date 
                    FROM raw_fundamental_data
                    WHERE date >= CURRENT_DATE - INTERVAL '2 years'
                """)).fetchall()
                
                self.existing_data_cache = {(row[0], row[1]) for row in existing}
                logger.info(f"Loaded {len(self.existing_data_cache)} existing data points")
                
        except Exception as e:
            logger.error(f"Failed to load existing data cache: {e}")
            self.existing_data_cache = set()
    
    def _has_recent_data(self, ticker_id: int) -> bool:
        """Check if ticker has recent data (within last 6 months)"""
        cutoff_date = datetime.now() - timedelta(days=180)
        cutoff_date = cutoff_date.date()
        
        for existing_ticker_id, existing_date in self.existing_data_cache:
            if existing_ticker_id == ticker_id and existing_date >= cutoff_date:
                return True
        return False
    
    async def collect_fundamental_data(self, ticker: str) -> bool:
        """
        Collect fundamental data for a single ticker with complete pipeline execution
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize caches if needed
            if not self.ticker_cache:
                self.ticker_cache = self._load_ticker_cache()
            if not self.existing_data_cache:
                self._load_existing_data_cache()
            
            ticker_id = self.ticker_cache.get(ticker)
            if not ticker_id:
                logger.warning(f"Ticker {ticker} not found in cache")
                return False
            
            # Check if we already have recent data for this ticker
            if self._has_recent_data(ticker_id):
                logger.info(f"Skipping {ticker} - recent data exists")
                self.stats['skipped'] += 1
                return True
            
            # Rate limit wait
            await self.rate_limiter.acquire()
            
            try:
                logger.info(f"Fetching fundamental data for {ticker} (ID: {ticker_id})")
                
                async with PolygonFundamentalsClient(self.config) as client:
                    # Fetch financial data
                    response = await client.get_financials(ticker)
                    
                    if not response or not response.income_statements:
                        logger.warning(f"No fundamental data found for {ticker}")
                        return False
                    
                    # Store each statement period immediately
                    stored_count = 0
                    for stmt in response.income_statements:
                        if await self._store_statement_period(ticker_id, stmt, response):
                            stored_count += 1
                    
                    logger.info(f"Stored {stored_count} fundamental records for {ticker}")
                    return stored_count > 0
                    
            finally:
                self.rate_limiter.release()
                
        except Exception as e:
            logger.error(f"Failed to collect fundamental data for {ticker}: {e}")
            return False
    
    async def _store_statement_period(self, ticker_id: int, income_stmt: Any, response: Any) -> bool:
        """Store a single statement period to database"""
        try:
            # Find corresponding balance sheet and cash flow data
            balance_sheet = self._find_matching_statement(
                response.balance_sheets, 
                income_stmt.end_date, 
                income_stmt.fiscal_period, 
                income_stmt.fiscal_year
            )
            
            cash_flow = self._find_matching_statement(
                response.cash_flow_statements,
                income_stmt.end_date,
                income_stmt.fiscal_period,
                income_stmt.fiscal_year
            )
            
            # Prepare data for insertion
            data = self._prepare_raw_data(ticker_id, income_stmt, balance_sheet, cash_flow)
            
            # Insert into database using raw SQL
            with self.SessionLocal() as session:
                stmt = text("""
                    INSERT INTO raw_fundamental_data (
                        ticker_id, date, filing_date, fiscal_period, fiscal_year, timeframe,
                        cik, company_name, source_filing_url, source_filing_file_url,
                        revenues, cost_of_revenue, gross_profit, operating_expenses,
                        selling_general_and_administrative_expenses, research_and_development,
                        operating_income_loss, nonoperating_income_loss,
                        income_loss_from_continuing_operations_before_tax, income_tax_expense_benefit,
                        income_loss_from_continuing_operations_after_tax, net_income_loss,
                        net_income_loss_attributable_to_parent, basic_earnings_per_share,
                        diluted_earnings_per_share, basic_average_shares, diluted_average_shares,
                        assets, current_assets, noncurrent_assets, inventory, other_current_assets,
                        fixed_assets, other_noncurrent_assets, liabilities, current_liabilities,
                        noncurrent_liabilities, accounts_payable, other_current_liabilities,
                        long_term_debt, other_noncurrent_liabilities, equity, equity_attributable_to_parent,
                        net_cash_flow_from_operating_activities, net_cash_flow_from_investing_activities,
                        net_cash_flow_from_financing_activities, net_cash_flow, net_cash_flow_continuing,
                        net_cash_flow_from_operating_activities_continuing,
                        net_cash_flow_from_investing_activities_continuing,
                        net_cash_flow_from_financing_activities_continuing,
                        comprehensive_income_loss, comprehensive_income_loss_attributable_to_parent,
                        other_comprehensive_income_loss, other_comprehensive_income_loss_attributable_to_parent,
                        data_quality_score, missing_data_count
                    ) VALUES (
                        :ticker_id, :date, :filing_date, :fiscal_period, :fiscal_year, :timeframe,
                        :cik, :company_name, :source_filing_url, :source_filing_file_url,
                        :revenues, :cost_of_revenue, :gross_profit, :operating_expenses,
                        :selling_general_and_administrative_expenses, :research_and_development,
                        :operating_income_loss, :nonoperating_income_loss,
                        :income_loss_from_continuing_operations_before_tax, :income_tax_expense_benefit,
                        :income_loss_from_continuing_operations_after_tax, :net_income_loss,
                        :net_income_loss_attributable_to_parent, :basic_earnings_per_share,
                        :diluted_earnings_per_share, :basic_average_shares, :diluted_average_shares,
                        :assets, :current_assets, :noncurrent_assets, :inventory, :other_current_assets,
                        :fixed_assets, :other_noncurrent_assets, :liabilities, :current_liabilities,
                        :noncurrent_liabilities, :accounts_payable, :other_current_liabilities,
                        :long_term_debt, :other_noncurrent_liabilities, :equity, :equity_attributable_to_parent,
                        :net_cash_flow_from_operating_activities, :net_cash_flow_from_investing_activities,
                        :net_cash_flow_from_financing_activities, :net_cash_flow, :net_cash_flow_continuing,
                        :net_cash_flow_from_operating_activities_continuing,
                        :net_cash_flow_from_investing_activities_continuing,
                        :net_cash_flow_from_financing_activities_continuing,
                        :comprehensive_income_loss, :comprehensive_income_loss_attributable_to_parent,
                        :other_comprehensive_income_loss, :other_comprehensive_income_loss_attributable_to_parent,
                        :data_quality_score, :missing_data_count
                    )
                    ON CONFLICT (ticker_id, date) DO UPDATE SET
                        updated_at = CURRENT_TIMESTAMP,
                        filing_date = EXCLUDED.filing_date,
                        fiscal_period = EXCLUDED.fiscal_period,
                        fiscal_year = EXCLUDED.fiscal_year,
                        timeframe = EXCLUDED.timeframe,
                        cik = EXCLUDED.cik,
                        company_name = EXCLUDED.company_name,
                        source_filing_url = EXCLUDED.source_filing_url,
                        source_filing_file_url = EXCLUDED.source_filing_file_url,
                        revenues = EXCLUDED.revenues,
                        cost_of_revenue = EXCLUDED.cost_of_revenue,
                        gross_profit = EXCLUDED.gross_profit,
                        operating_expenses = EXCLUDED.operating_expenses,
                        selling_general_and_administrative_expenses = EXCLUDED.selling_general_and_administrative_expenses,
                        research_and_development = EXCLUDED.research_and_development,
                        operating_income_loss = EXCLUDED.operating_income_loss,
                        nonoperating_income_loss = EXCLUDED.nonoperating_income_loss,
                        income_loss_from_continuing_operations_before_tax = EXCLUDED.income_loss_from_continuing_operations_before_tax,
                        income_tax_expense_benefit = EXCLUDED.income_tax_expense_benefit,
                        income_loss_from_continuing_operations_after_tax = EXCLUDED.income_loss_from_continuing_operations_after_tax,
                        net_income_loss = EXCLUDED.net_income_loss,
                        net_income_loss_attributable_to_parent = EXCLUDED.net_income_loss_attributable_to_parent,
                        basic_earnings_per_share = EXCLUDED.basic_earnings_per_share,
                        diluted_earnings_per_share = EXCLUDED.diluted_earnings_per_share,
                        basic_average_shares = EXCLUDED.basic_average_shares,
                        diluted_average_shares = EXCLUDED.diluted_average_shares,
                        assets = EXCLUDED.assets,
                        current_assets = EXCLUDED.current_assets,
                        noncurrent_assets = EXCLUDED.noncurrent_assets,
                        inventory = EXCLUDED.inventory,
                        other_current_assets = EXCLUDED.other_current_assets,
                        fixed_assets = EXCLUDED.fixed_assets,
                        other_noncurrent_assets = EXCLUDED.other_noncurrent_assets,
                        liabilities = EXCLUDED.liabilities,
                        current_liabilities = EXCLUDED.current_liabilities,
                        noncurrent_liabilities = EXCLUDED.noncurrent_liabilities,
                        accounts_payable = EXCLUDED.accounts_payable,
                        other_current_liabilities = EXCLUDED.other_current_liabilities,
                        long_term_debt = EXCLUDED.long_term_debt,
                        other_noncurrent_liabilities = EXCLUDED.other_noncurrent_liabilities,
                        equity = EXCLUDED.equity,
                        equity_attributable_to_parent = EXCLUDED.equity_attributable_to_parent,
                        net_cash_flow_from_operating_activities = EXCLUDED.net_cash_flow_from_operating_activities,
                        net_cash_flow_from_investing_activities = EXCLUDED.net_cash_flow_from_investing_activities,
                        net_cash_flow_from_financing_activities = EXCLUDED.net_cash_flow_from_financing_activities,
                        net_cash_flow = EXCLUDED.net_cash_flow,
                        net_cash_flow_continuing = EXCLUDED.net_cash_flow_continuing,
                        net_cash_flow_from_operating_activities_continuing = EXCLUDED.net_cash_flow_from_operating_activities_continuing,
                        net_cash_flow_from_investing_activities_continuing = EXCLUDED.net_cash_flow_from_investing_activities_continuing,
                        net_cash_flow_from_financing_activities_continuing = EXCLUDED.net_cash_flow_from_financing_activities_continuing,
                        comprehensive_income_loss = EXCLUDED.comprehensive_income_loss,
                        comprehensive_income_loss_attributable_to_parent = EXCLUDED.comprehensive_income_loss_attributable_to_parent,
                        other_comprehensive_income_loss = EXCLUDED.other_comprehensive_income_loss,
                        other_comprehensive_income_loss_attributable_to_parent = EXCLUDED.other_comprehensive_income_loss_attributable_to_parent,
                        data_quality_score = EXCLUDED.data_quality_score,
                        missing_data_count = EXCLUDED.missing_data_count
                """)
                
                session.execute(stmt, data)
                session.commit()
                
                logger.debug(f"Stored fundamental data for ticker_id {ticker_id}, date {data['date']}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store statement period for ticker_id {ticker_id}: {e}")
            return False
    
    def _find_matching_statement(self, statements: List, end_date: str, fiscal_period: str, fiscal_year: str):
        """Find matching statement by date and fiscal period"""
        for stmt in statements:
            if (stmt.end_date == end_date and 
                stmt.fiscal_period == fiscal_period and 
                stmt.fiscal_year == fiscal_year):
                return stmt
        return None
    
    def _prepare_raw_data(self, ticker_id: int, income_stmt: Any, balance_sheet: Any, cash_flow: Any) -> Dict[str, Any]:
        """Prepare raw data for database insertion"""
        
        # Handle date parsing - check if it's already a date object or string
        def parse_date(date_value):
            if date_value is None:
                return None
            if isinstance(date_value, date):
                return date_value
            elif isinstance(date_value, str):
                return datetime.strptime(date_value, '%Y-%m-%d').date()
            else:
                logger.warning(f"Unexpected date format: {type(date_value)} - {date_value}")
                return None
        
        data = {
            'ticker_id': ticker_id,
            'date': parse_date(income_stmt.end_date),
            'filing_date': parse_date(income_stmt.filing_date),
            'fiscal_period': income_stmt.fiscal_period,
            'fiscal_year': income_stmt.fiscal_year,
            'timeframe': income_stmt.timeframe,
            'cik': income_stmt.cik,
            'company_name': income_stmt.company_name,
            'source_filing_url': income_stmt.source_filing_url,
            'source_filing_file_url': income_stmt.source_filing_file_url,
            'data_quality_score': getattr(income_stmt, 'data_quality_score', None),
            'missing_data_count': getattr(income_stmt, 'missing_data_count', 0)
        }
        
        # Income Statement Fields
        data.update(self._extract_financial_values(income_stmt, [
            'revenues', 'cost_of_revenue', 'gross_profit', 'operating_expenses',
            'selling_general_and_administrative_expenses', 'research_and_development',
            'operating_income_loss', 'nonoperating_income_loss',
            'income_loss_from_continuing_operations_before_tax', 'income_tax_expense_benefit',
            'income_loss_from_continuing_operations_after_tax', 'net_income_loss',
            'net_income_loss_attributable_to_parent', 'basic_earnings_per_share',
            'diluted_earnings_per_share', 'basic_average_shares', 'diluted_average_shares'
        ]))
        
        # Balance Sheet Fields
        if balance_sheet:
            data.update(self._extract_financial_values(balance_sheet, [
                'assets', 'current_assets', 'noncurrent_assets', 'inventory', 'other_current_assets',
                'fixed_assets', 'other_noncurrent_assets', 'liabilities', 'current_liabilities',
                'noncurrent_liabilities', 'accounts_payable', 'other_current_liabilities',
                'long_term_debt', 'other_noncurrent_liabilities', 'equity', 'equity_attributable_to_parent'
            ]))
        
        # Cash Flow Fields
        if cash_flow:
            data.update(self._extract_financial_values(cash_flow, [
                'net_cash_flow_from_operating_activities', 'net_cash_flow_from_investing_activities',
                'net_cash_flow_from_financing_activities', 'net_cash_flow', 'net_cash_flow_continuing',
                'net_cash_flow_from_operating_activities_continuing',
                'net_cash_flow_from_investing_activities_continuing',
                'net_cash_flow_from_financing_activities_continuing'
            ]))
        
        # Comprehensive Income Fields
        data.update(self._extract_financial_values(income_stmt, [
            'comprehensive_income_loss', 'comprehensive_income_loss_attributable_to_parent',
            'other_comprehensive_income_loss', 'other_comprehensive_income_loss_attributable_to_parent'
        ]))
        
        return data
    
    def _extract_financial_values(self, stmt: Any, fields: List[str]) -> Dict[str, Any]:
        """Extract financial values from statement object"""
        result = {}
        for field in fields:
            value = getattr(stmt, field, None)
            if value is not None:
                if hasattr(value, 'value'):
                    result[field] = value.value
                else:
                    result[field] = value
            else:
                result[field] = None
        return result
    
    async def collect_with_progress(self, tickers: List[str]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Collect fundamental data with real-time progress tracking
        
        Args:
            tickers: List of ticker symbols
            
        Yields:
            Progress updates for each ticker
        """
        self.stats['start_time'] = datetime.now()
        self.stats['total_processed'] = 0
        self.stats['successful'] = 0
        self.stats['failed'] = 0
        self.stats['skipped'] = 0
        
        logger.info(f"Starting optimized collection for {len(tickers)} tickers")
        logger.info(f"Rate limit: {self.config.REQUESTS_PER_MINUTE} requests per minute")
        logger.info(f"Estimated time: {len(tickers) * 12 / 60:.1f} minutes")  # 12 seconds per ticker
        
        for i, ticker in enumerate(tickers):
            try:
                success = await self.collect_fundamental_data(ticker)
                
                self.stats['total_processed'] += 1
                if success:
                    self.stats['successful'] += 1
                else:
                    self.stats['failed'] += 1
                
                # Yield progress update
                progress_data = {
                    'current': i + 1,
                    'total': len(tickers),
                    'ticker': ticker,
                    'success': success,
                    'progress': (i + 1) / len(tickers) * 100,
                    'stats': self.stats.copy()
                }
                yield progress_data
                
                # Log progress every 10 tickers
                if (i + 1) % 10 == 0:
                    elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
                    rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
                    logger.info(f"Progress: {i+1}/{len(tickers)} ({((i+1)/len(tickers)*100):.1f}%) - Rate: {rate:.1f} tickers/minute")
                
            except Exception as e:
                logger.error(f"Exception processing {ticker}: {e}")
                self.stats['failed'] += 1
                
                yield {
                    'current': i + 1,
                    'total': len(tickers),
                    'ticker': ticker,
                    'success': False,
                    'error': str(e),
                    'progress': (i + 1) / len(tickers) * 100,
                    'stats': self.stats.copy()
                }
        
        self.stats['end_time'] = datetime.now()
        
        # Final statistics
        elapsed = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        logger.info(f"Collection complete in {elapsed/60:.1f} minutes")
        logger.info(f"Success rate: {self.stats['successful']}/{self.stats['total_processed']} ({self.stats['successful']/self.stats['total_processed']*100:.1f}%)")
        logger.info(f"Skipped: {self.stats['skipped']} tickers")
    
    async def collect_batch(self, tickers: List[str]) -> Dict[str, bool]:
        """
        Collect fundamental data for a list of tickers (legacy method)
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping ticker to success status
        """
        results = {}
        async for progress in self.collect_with_progress(tickers):
            results[progress['ticker']] = progress['success']
        return results 
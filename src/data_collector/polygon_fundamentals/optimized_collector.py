"""
Optimized Fundamental Data Collector

This module handles fundamental data collection with complete pipeline execution
per ticker and simplified rate limiting.
"""

from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.data_collector.polygon_fundamentals.client import PolygonFundamentalsClient
from src.data_collector.polygon_fundamentals.config import PolygonFundamentalsConfig
from src.data_collector.polygon_fundamentals.cache_manager import (
    FundamentalCacheManager,
)
from src.data_collector.polygon_data.data_storage import DataStorage
from src.data_collector.polygon_data.rate_limiter import AdaptiveRateLimiter
from src.data_collector.config import config
from src.utils.logger import get_logger
from src.database.connection import get_global_pool, fetch_all, fetch_one, execute

logger = get_logger(__name__)


def calculate_source_confidence(source: str) -> float:
    mapping = {
        "direct_report": 1.0,
        "intra_report_impute": 0.8,
        "inter_report_derive": 0.6,
    }
    return mapping.get(source, 0.5)


class FundamentalRateLimiter:
    """Wrapper for existing polygon rate limiter"""

    def __init__(self):
        self.rate_limiter = AdaptiveRateLimiter()

    async def acquire(self):
        """Acquire rate limit using existing framework (skip if disabled)"""
        if not getattr(config, "DISABLE_RATE_LIMITING", False):
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
        self.db_pool = get_global_pool()

        # Create SQLAlchemy engine from connection parameters
        # prefer using connection pool config if available
        cfg = getattr(self.db_pool, "config", {})
        database_url = f"postgresql://{cfg.get('user', '')}:{cfg.get('password', '')}@{cfg.get('host', 'localhost')}:{cfg.get('port', 5432)}/{cfg.get('database', '')}"
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

        # Initialize data storage for ticker metadata
        self.data_storage = DataStorage()
        self.ticker_cache = {}  # Cache for ticker_id mapping
        self.existing_data_cache = set()  # Cache for existing ticker-date combinations

        # Initialize cache manager
        self.cache_manager = FundamentalCacheManager()

        # Rate limiting - use existing framework
        self.rate_limiter = FundamentalRateLimiter()

        # Statistics
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "start_time": None,
            "end_time": None,
        }

    def _load_ticker_cache(self) -> Dict[str, int]:
        """
        Build and return an in-memory mapping of ticker symbol to ticker ID.
        
        Queries self.data_storage.get_tickers() and constructs a dict where keys are ticker symbols (str)
        and values are integer ticker IDs. Entries missing a ticker or an ID are skipped. On error or if
        no tickers are found, an empty dict is returned.
        """
        try:
            logger.info("Loading ticker metadata into cache...")

            # Get tickers from database using DataStorage
            tickers_data = self.data_storage.get_tickers()

            if tickers_data:
                ticker_cache = {}
                for ticker_data in tickers_data:
                    ticker = ticker_data.get("ticker")
                    ticker_id = ticker_data.get("id")
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
        """Load existing fundamental data into cache"""
        try:
            logger.info("Loading existing data cache...")

            rows = fetch_all("SELECT ticker_id, date FROM raw_fundamental_data")
            existing_data = set((r["ticker_id"], r["date"]) for r in (rows or []))
            self.existing_data_cache = existing_data
            logger.info(f"Loaded {len(existing_data)} existing data points")

        except Exception as e:
            logger.error(f"Failed to load existing data cache: {e}")
            self.existing_data_cache = set()

    def _has_recent_data(self, ticker_id: int) -> bool:
        """Check if ticker has recent data (within 6 months)"""
        try:
            result = fetch_one(
                """
                    SELECT EXISTS (
                        SELECT 1
                        FROM raw_fundamental_data
                        WHERE ticker_id = %s
                        AND date >= %s::date
                    ) AS has_recent
                """,
                (ticker_id, (datetime.now() - timedelta(days=180)).date()),
            )
            has_recent = result and result.get("has_recent")
            logger.info(f"Ticker ID {ticker_id} recent data: {has_recent}")
            return has_recent

        except Exception as e:
            logger.error(f"Error checking recent data for ticker_id {ticker_id}: {e}")
            return False

    async def _process_cached_data(
        self, ticker: str, cached_data: Dict[str, Any]
    ) -> bool:
        """Process cached data and store to database"""
        try:
            # Get ticker_id from cache
            ticker_id = self.ticker_cache.get(ticker)
            if ticker_id is None:
                logger.warning(f"Ticker {ticker} not found in cache")
                return False

            # Check if we have recent data
            if self._has_recent_data(ticker_id):
                logger.info(f"Skipping {ticker} - recent data exists")
                self.stats["skipped"] += 1
                return True

            # Process cached results
            if not cached_data.get("results"):
                logger.warning(f"No results found in cached data for {ticker}")
                return False

            success_count = 0
            for result in cached_data["results"]:
                if await self._store_statement_period(ticker_id, result, cached_data):
                    success_count += 1

            if success_count > 0:
                logger.info(
                    f"Successfully stored {success_count} statement periods for {ticker} from cache"
                )
                self.stats["successful"] += 1
                self.stats["cache_hits"] += 1
                return True
            else:
                logger.warning(f"No statement periods stored for {ticker} from cache")
                self.stats["failed"] += 1
                return False

        except Exception as e:
            logger.error(f"Error processing cached data for {ticker}: {e}")
            self.stats["failed"] += 1
            return False

    async def collect_fundamental_data(self, ticker: str) -> bool:
        """
        Collect fundamental data for a single ticker with cache-first approach

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize caches if needed
            if not self.ticker_cache:
                self.ticker_cache = self._load_ticker_cache()
                self._load_existing_data_cache()

            # Get ticker_id from cache
            ticker_id = self.ticker_cache.get(ticker)
            if ticker_id is None:
                logger.warning(f"Ticker {ticker} not found in cache")
                return False

            # Check if we have recent data
            if self._has_recent_data(ticker_id):
                logger.info(f"Skipping {ticker} - recent data exists")
                self.stats["skipped"] += 1
                return True

            # Try to get cached data first
            cached_data = self.cache_manager.get_cached_data(ticker)
            if cached_data:
                logger.info(f"Found valid cache for {ticker}, processing cached data")
                return await self._process_cached_data(ticker, cached_data)

            # No valid cache found, proceed with API call
            logger.info(f"No valid cache found for {ticker}, making API call")

            # Rate limiting
            await self.rate_limiter.acquire()

            # Collect data using client
            async with PolygonFundamentalsClient() as client:
                response = await client.get_financials(ticker)
                self.stats["api_calls"] += 1

                if not response or not response.results:
                    logger.warning(f"No financial data found for {ticker}")
                    self.stats["failed"] += 1
                    return False

                # Process each statement period
                success_count = 0
                for result in response.results:
                    if await self._store_statement_period(ticker_id, result, response):
                        success_count += 1

                if success_count > 0:
                    logger.info(
                        f"Successfully stored {success_count} statement periods for {ticker}"
                    )
                    self.stats["successful"] += 1
                    return True
                else:
                    logger.warning(f"No statement periods stored for {ticker}")
                    self.stats["failed"] += 1
                    return False

        except Exception as e:
            logger.error(f"Error collecting fundamental data for {ticker}: {e}")
            self.stats["failed"] += 1
            return False

    async def _store_statement_period(
        self, ticker_id: int, income_stmt: Any, response: Any
    ) -> bool:
        """Store a single statement period to database"""
        try:
            # Handle both dictionary (cached) and object (API response) formats
            if isinstance(response, dict):
                results = response.get("results", [])
            else:
                results = response.results

            # Helper function to get attribute safely
            def get_attr(obj, attr_name, default=None):
                if isinstance(obj, dict):
                    return obj.get(attr_name, default)
                else:
                    return getattr(obj, attr_name, default)

            # Find matching balance sheet and cash flow
            balance_sheet = self._find_matching_statement(
                results,
                get_attr(income_stmt, "end_date"),
                get_attr(income_stmt, "fiscal_period"),
                get_attr(income_stmt, "fiscal_year"),
            )

            cash_flow = self._find_matching_statement(
                results,
                get_attr(income_stmt, "end_date"),
                get_attr(income_stmt, "fiscal_period"),
                get_attr(income_stmt, "fiscal_year"),
            )

            # Prepare raw data
            raw_data = self._prepare_raw_data(
                ticker_id, income_stmt, balance_sheet, cash_flow
            )

            # Store to database using connection pool
            # Use centralized execute helper for single-statement upsert
            execute(
                """
                INSERT INTO raw_fundamental_data (
                    ticker_id, date, filing_date, fiscal_period, fiscal_year, timeframe,
                    revenues, cost_of_revenue, gross_profit, operating_expenses, net_income_loss,
                    assets, current_assets, liabilities, equity, long_term_debt,
                    net_cash_flow_from_operating_activities, net_cash_flow_from_investing_activities,
                    comprehensive_income_loss, other_comprehensive_income_loss,
                    data_quality_score, missing_data_count, source_filing_url, source_filing_file_url,
                    data_source_confidence
                ) VALUES (
                    %(ticker_id)s, %(date)s, %(filing_date)s, %(fiscal_period)s, %(fiscal_year)s, %(timeframe)s,
                    %(revenues)s, %(cost_of_revenue)s, %(gross_profit)s, %(operating_expenses)s, %(net_income_loss)s,
                    %(assets)s, %(current_assets)s, %(liabilities)s, %(equity)s, %(long_term_debt)s,
                    %(net_cash_flow_from_operating_activities)s, %(net_cash_flow_from_investing_activities)s,
                    %(comprehensive_income_loss)s, %(other_comprehensive_income_loss)s,
                    %(data_quality_score)s, %(missing_data_count)s, %(source_filing_url)s, %(source_filing_file_url)s,
                    %(data_source_confidence)s
                ) ON CONFLICT (ticker_id, date) DO UPDATE SET
                    filing_date = EXCLUDED.filing_date,
                    fiscal_period = EXCLUDED.fiscal_period,
                    fiscal_year = EXCLUDED.fiscal_year,
                    timeframe = EXCLUDED.timeframe,
                    revenues = EXCLUDED.revenues,
                    cost_of_revenue = EXCLUDED.cost_of_revenue,
                    gross_profit = EXCLUDED.gross_profit,
                    operating_expenses = EXCLUDED.operating_expenses,
                    net_income_loss = EXCLUDED.net_income_loss,
                    assets = EXCLUDED.assets,
                    current_assets = EXCLUDED.current_assets,
                    liabilities = EXCLUDED.liabilities,
                    equity = EXCLUDED.equity,
                    long_term_debt = EXCLUDED.long_term_debt,
                    net_cash_flow_from_operating_activities = EXCLUDED.net_cash_flow_from_operating_activities,
                    net_cash_flow_from_investing_activities = EXCLUDED.net_cash_flow_from_investing_activities,
                    comprehensive_income_loss = EXCLUDED.comprehensive_income_loss,
                    other_comprehensive_income_loss = EXCLUDED.other_comprehensive_income_loss,
                    data_quality_score = EXCLUDED.data_quality_score,
                    missing_data_count = EXCLUDED.missing_data_count,
                    source_filing_url = EXCLUDED.source_filing_url,
                    source_filing_file_url = EXCLUDED.source_filing_file_url,
                    data_source_confidence = EXCLUDED.data_source_confidence
                """,
                raw_data,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to store statement period: {e}")
            return False

    def _find_matching_statement(
        self, statements: List, end_date: str, fiscal_period: str, fiscal_year: str
    ):
        """Find matching statement by date and fiscal period"""
        for stmt in statements:
            # Handle both dictionary and object formats
            if isinstance(stmt, dict):
                stmt_end_date = stmt.get("end_date")
                stmt_fiscal_period = stmt.get("fiscal_period")
                stmt_fiscal_year = stmt.get("fiscal_year")
            else:
                stmt_end_date = stmt.end_date
                stmt_fiscal_period = stmt.fiscal_period
                stmt_fiscal_year = stmt.fiscal_year

            if (
                stmt_end_date == end_date
                and stmt_fiscal_period == fiscal_period
                and stmt_fiscal_year == fiscal_year
            ):
                return stmt
        return None

    def _prepare_raw_data(
        self, ticker_id: int, income_stmt: Any, balance_sheet: Any, cash_flow: Any
    ) -> Dict[str, Any]:
        """Prepare raw data for database storage"""

        def parse_date(date_value):
            """Parse date string to date object"""
            if isinstance(date_value, str):
                try:
                    return datetime.strptime(date_value, "%Y-%m-%d").date()
                except ValueError:
                    return None
            return date_value

        # Helper function to get attribute safely
        def get_attr(obj, attr_name, default=None):
            if isinstance(obj, dict):
                # Check if this is a cached data structure with nested financials
                if "financials" in obj:
                    # For cached data, metadata fields are at the top level
                    return obj.get(attr_name, default)
                else:
                    return obj.get(attr_name, default)
            else:
                return getattr(obj, attr_name, default)

        # Extract financial values
        income_values = self._extract_financial_values(
            income_stmt,
            [
                "revenues",
                "cost_of_revenue",
                "gross_profit",
                "operating_expenses",
                "net_income_loss",
            ],
        )

        balance_values = (
            self._extract_financial_values(
                balance_sheet,
                ["assets", "current_assets", "liabilities", "equity", "long_term_debt"],
            )
            if balance_sheet
            else {}
        )

        cash_flow_values = (
            self._extract_financial_values(
                cash_flow,
                [
                    "net_cash_flow_from_operating_activities",
                    "net_cash_flow_from_investing_activities",
                ],
            )
            if cash_flow
            else {}
        )

        # Calculate data quality metrics
        all_values = {**income_values, **balance_values, **cash_flow_values}
        non_null_count = sum(1 for v in all_values.values() if v is not None)
        total_fields = len(all_values)
        data_quality_score = non_null_count / total_fields if total_fields > 0 else 0
        missing_data_count = total_fields - non_null_count

        return {
            "ticker_id": ticker_id,
            "date": parse_date(get_attr(income_stmt, "end_date")),
            "filing_date": parse_date(get_attr(income_stmt, "filing_date")),
            "fiscal_period": get_attr(income_stmt, "fiscal_period"),
            "fiscal_year": get_attr(income_stmt, "fiscal_year"),
            "timeframe": get_attr(income_stmt, "timeframe"),
            "revenues": income_values.get("revenues"),
            "cost_of_revenue": income_values.get("cost_of_revenue"),
            "gross_profit": income_values.get("gross_profit"),
            "operating_expenses": income_values.get("operating_expenses"),
            "net_income_loss": income_values.get("net_income_loss"),
            "assets": balance_values.get("assets"),
            "current_assets": balance_values.get("current_assets"),
            "liabilities": balance_values.get("liabilities"),
            "equity": balance_values.get("equity"),
            "long_term_debt": balance_values.get("long_term_debt"),
            "net_cash_flow_from_operating_activities": cash_flow_values.get(
                "net_cash_flow_from_operating_activities"
            ),
            "net_cash_flow_from_investing_activities": cash_flow_values.get(
                "net_cash_flow_from_investing_activities"
            ),
            "comprehensive_income_loss": None,  # Not available in current data
            "other_comprehensive_income_loss": None,  # Not available in current data
            "data_quality_score": data_quality_score,
            "missing_data_count": missing_data_count,
            "source_filing_url": get_attr(income_stmt, "source_filing_url"),
            "source_filing_file_url": get_attr(income_stmt, "source_filing_file_url"),
            "data_source_confidence": income_values.get(
                "data_source_confidence"
            ),  # Add data_source_confidence
        }

    def _extract_financial_values(self, stmt: Any, fields: List[str]) -> Dict[str, Any]:
        """Extract financial values from statement"""
        values = {}
        confidences = []

        # Helper function to get field value safely
        def get_field_value(stmt, field_name):
            if isinstance(stmt, dict):
                return stmt.get(field_name)
            else:
                return getattr(stmt, field_name, None)

        # Helper function to get nested financial value
        def get_nested_financial_value(stmt, field_name):
            if isinstance(stmt, dict):
                # Check if this is a cached data structure with nested financials
                if "financials" in stmt:
                    financials = stmt["financials"]
                    # Check each statement type for the field
                    for stmt_type in [
                        "income_statement",
                        "balance_sheet",
                        "cash_flow_statement",
                    ]:
                        if (
                            stmt_type in financials
                            and field_name in financials[stmt_type]
                        ):
                            return financials[stmt_type][field_name]
                    return None
                else:
                    # Direct field access (legacy format)
                    return stmt.get(field_name)
            else:
                # Object format (from API)
                return getattr(stmt, field_name, None)

        for field in fields:
            value = get_nested_financial_value(stmt, field)
            if value is not None:
                # Handle both dictionary and object formats for financial data
                if isinstance(value, dict):
                    # Dictionary format (from cache)
                    if "value" in value:
                        values[field] = value["value"]
                        # Collect confidence if source is present
                        if "source" in value and value["source"]:
                            confidences.append(
                                calculate_source_confidence(value["source"])
                            )
                    else:
                        values[field] = value
                else:
                    # Object format (from API)
                    if hasattr(value, "value"):
                        values[field] = value.value
                        # Collect confidence if source is present
                        if hasattr(value, "source") and value.source:
                            confidences.append(
                                calculate_source_confidence(value.source)
                            )
                    else:
                        values[field] = value
            else:
                values[field] = None

        # Set average confidence
        values["data_source_confidence"] = (
            sum(confidences) / len(confidences) if confidences else None
        )
        return values

    async def collect_with_progress(
        self, tickers: List[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Collect fundamental data with real-time progress tracking"""
        self.stats["start_time"] = datetime.now()

        for i, ticker in enumerate(tickers):
            logger.info(f"Processing ticker {i + 1}/{len(tickers)}: {ticker}")

            try:
                success = await self.collect_fundamental_data(ticker)
                self.stats["total_processed"] += 1

                yield {
                    "ticker": ticker,
                    "success": success,
                    "progress": i + 1,
                    "total": len(tickers),
                    "stats": self.stats.copy(),
                }

            except Exception as e:
                logger.error(f"Exception processing {ticker}: {e}")
                self.stats["failed"] += 1
                self.stats["total_processed"] += 1

                yield {
                    "ticker": ticker,
                    "success": False,
                    "error": str(e),
                    "progress": i + 1,
                    "total": len(tickers),
                    "stats": self.stats.copy(),
                }

        self.stats["end_time"] = datetime.now()

    def close(self):
        """Close the collector and cleanup resources"""
        try:
            # Close SQLAlchemy engine
            if hasattr(self, "engine"):
                self.engine.dispose()
                logger.info("SQLAlchemy engine disposed")

            # Clear caches
            if hasattr(self, "ticker_cache"):
                self.ticker_cache.clear()
            if hasattr(self, "existing_data_cache"):
                self.existing_data_cache.clear()

            logger.info("OptimizedFundamentalCollector closed")

        except Exception as e:
            logger.error(f"Error during collector cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.close()

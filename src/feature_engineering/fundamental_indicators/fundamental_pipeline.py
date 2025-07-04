"""
Fundamental Analysis Pipeline

This module orchestrates the complete fundamental analysis workflow including:
- Data collection from Polygon API
- Financial statement processing
- Fundamental metrics calculation
- Database storage and caching
- Error handling and monitoring
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from src.feature_engineering.config import FundamentalConfig
from src.data_collector.polygon_fundamentals.client import PolygonFundamentalsClient
from src.data_collector.polygon_fundamentals.data_validator import FundamentalDataValidator
from src.feature_engineering.fundamental_indicators.base import FundamentalCalculatorRegistry
from src.feature_engineering.data_loader import StockDataLoader
from src.database.fundamental_models import (
    FundamentalRatios, FundamentalGrowthMetrics, 
    FundamentalScores, FundamentalSectorAnalysis
)
from src.database.connection import DatabaseConnection
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert

logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    """Result of fundamental pipeline execution"""
    ticker: str
    success: bool
    metrics_calculated: Dict[str, int]
    errors: List[str]
    execution_time: float
    data_quality_score: Optional[float] = None

@dataclass
class PipelineStats:
    """Statistics for pipeline execution"""
    total_tickers: int
    successful: int
    failed: int
    total_metrics: int
    avg_execution_time: float
    start_time: datetime
    end_time: Optional[datetime] = None

class FundamentalPipeline:
    """
    Main pipeline for fundamental analysis processing
    
    Orchestrates the complete workflow from data collection to storage
    """
    
    def __init__(self, config: Optional[FundamentalConfig] = None):
        """
        Initialize the fundamental pipeline
        
        Args:
            config: Configuration object, creates default if None
        """
        self.config = config or FundamentalConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        from src.data_collector.polygon_fundamentals.config import PolygonFundamentalsConfig
        self.api_config = PolygonFundamentalsConfig()
        self.api_client = PolygonFundamentalsClient(self.api_config)
        
        self.data_validator = FundamentalDataValidator()
        self.db_manager = DatabaseConnection()
        
        # Initialize SQLAlchemy for database storage
        from src.feature_engineering.config import config
        self.engine = create_engine(config.database.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Initialize calculators using registry
        self.calculators = {}
        
        # Get all registered calculators from the registry
        available_calculators = FundamentalCalculatorRegistry.get_all_calculators()
        for name, calc_class in available_calculators.items():
            self.calculators[name] = calc_class(self.config)
        
        # Pipeline state
        self.stats = None
        self.results = []
        
        self.logger.info(f"Initialized FundamentalPipeline with {len(self.calculators)} calculators")
        self.logger.debug(f"Available calculators: {list(self.calculators.keys())}")
    
    async def process_ticker(self, ticker: str) -> PipelineResult:
        """
        Process a single ticker through the complete fundamental analysis pipeline
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            PipelineResult with execution details
        """
        start_time = datetime.now()
        result = PipelineResult(
            ticker=ticker,
            success=False,
            metrics_calculated={},
            errors=[],
            execution_time=0.0
        )
        
        try:
            self.logger.info(f"Processing ticker: {ticker}")
            
            # Step 1: Data Collection
            fundamental_data = await self._collect_fundamental_data(ticker)
            if not fundamental_data:
                result.errors.append("Failed to collect fundamental data")
                return result
            
            # Step 2: Data Validation and Quality Calculation
            validation_result = self.data_validator.validate_response(fundamental_data)
            
            # Calculate data quality score on the response object
            data_quality_score = fundamental_data.calculate_data_quality()
            
            # Log detailed validation information
            self._log_data_quality_details(ticker, validation_result, fundamental_data)
            
            if not validation_result.is_valid:
                result.errors.extend(validation_result.errors)
                if validation_result.quality_score < 0.5:  # Default quality threshold
                    result.errors.append(f"Data quality too low: {validation_result.quality_score}")
                    return result
            
            result.data_quality_score = data_quality_score
            
            # Step 3: Calculate Metrics
            calculated_metrics = await self._calculate_all_metrics(fundamental_data)
            result.metrics_calculated = {
                name: len(metrics.values) for name, metrics in calculated_metrics.items()
            }
            
            # Log detailed metrics calculation results
            self._log_metrics_calculation_details(ticker, calculated_metrics)
            
            # Step 4: Store Results
            await self._store_results(ticker, calculated_metrics, fundamental_data)
            self.logger.info(f"Successfully stored metrics to database for {ticker}")
            
            result.success = True
            self.logger.info(f"Successfully processed {ticker}: {sum(result.metrics_calculated.values())} metrics")
            
        except Exception as e:
            self.logger.error(f"Error processing {ticker}: {e}")
            result.errors.append(str(e))
        
        finally:
            result.execution_time = (datetime.now() - start_time).total_seconds()
        
        return result
    
    async def process_batch(self, tickers: List[str]) -> PipelineStats:
        """
        Process a batch of tickers
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            PipelineStats with batch execution statistics
        """
        self.stats = PipelineStats(
            total_tickers=len(tickers),
            successful=0,
            failed=0,
            total_metrics=0,
            avg_execution_time=0.0,
            start_time=datetime.now()
        )
        
        self.results = []
        
        self.logger.info(f"Starting batch processing of {len(tickers)} tickers")
        
        # Process tickers with concurrency control
        semaphore = asyncio.Semaphore(2)  # Reduced concurrency to avoid session conflicts
        
        async def process_with_semaphore(ticker: str) -> PipelineResult:
            async with semaphore:
                return await self.process_ticker(ticker)
        
        # Execute batch processing
        tasks = [process_with_semaphore(ticker) for ticker in tickers]
        self.results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate statistics
        successful_results = [r for r in self.results if isinstance(r, PipelineResult) and r.success]
        failed_results = [r for r in self.results if isinstance(r, PipelineResult) and not r.success]
        exception_results = [r for r in self.results if isinstance(r, Exception)]
        
        self.stats.successful = len(successful_results)
        self.stats.failed = len(failed_results) + len(exception_results)
        self.stats.total_metrics = sum(
            sum(r.metrics_calculated.values()) for r in successful_results
        )
        self.stats.avg_execution_time = (
            sum(r.execution_time for r in successful_results) / len(successful_results)
            if successful_results else 0.0
        )
        self.stats.end_time = datetime.now()
        
        self.logger.info(f"Batch processing completed: {self.stats.successful}/{self.stats.total_tickers} successful")
        
        return self.stats
    
    def _log_data_quality_details(self, ticker: str, validation_result, fundamental_data):
        """Log detailed data quality and validation information"""
        self.logger.info(f"📊 Data Quality Analysis for {ticker}:")
        
        # Basic data availability
        self.logger.info(f"  📈 Income Statements: {len(fundamental_data.income_statements)}")
        self.logger.info(f"  📋 Balance Sheets: {len(fundamental_data.balance_sheets)}")
        self.logger.info(f"  💰 Cash Flow Statements: {len(fundamental_data.cash_flow_statements)}")
        
        # Data quality scores
        self.logger.info(f"  🎯 Overall Data Quality: {fundamental_data.data_quality_score:.2f}")
        self.logger.info(f"  ✅ Validation Quality: {validation_result.quality_score:.2f}")
        self.logger.info(f"  📊 Missing Data Count: {fundamental_data.missing_data_count}")
        self.logger.info(f"  📈 Total Fields Count: {fundamental_data.total_fields_count}")
        
        # Validation issues
        if validation_result.errors:
            self.logger.warning(f"  ❌ Validation Errors ({len(validation_result.errors)}):")
            for error in validation_result.errors[:5]:  # Show first 5 errors
                self.logger.warning(f"    • {error}")
            if len(validation_result.errors) > 5:
                self.logger.warning(f"    ... and {len(validation_result.errors) - 5} more errors")
        
        if validation_result.warnings:
            self.logger.warning(f"  ⚠️  Validation Warnings ({len(validation_result.warnings)}):")
            for warning in validation_result.warnings[:3]:  # Show first 3 warnings
                self.logger.warning(f"    • {warning}")
            if len(validation_result.warnings) > 3:
                self.logger.warning(f"    ... and {len(validation_result.warnings) - 3} more warnings")
        
        if validation_result.missing_fields:
            self.logger.warning(f"  🔍 Missing Fields ({len(validation_result.missing_fields)}):")
            # Group missing fields by statement type
            missing_by_type = {}
            for field in validation_result.missing_fields:
                if '.' in field:
                    stmt_type, field_name = field.split('.', 1)
                    if stmt_type not in missing_by_type:
                        missing_by_type[stmt_type] = []
                    missing_by_type[stmt_type].append(field_name)
                else:
                    if 'other' not in missing_by_type:
                        missing_by_type['other'] = []
                    missing_by_type['other'].append(field)
            
            for stmt_type, fields in missing_by_type.items():
                self.logger.warning(f"    📋 {stmt_type.replace('_', ' ').title()}:")
                for field in fields:
                    self.logger.warning(f"      • {field}")
        
        # Also check for missing fields in each calculator's required fields
        self._log_calculator_missing_fields(ticker, fundamental_data)
        
        if validation_result.outliers:
            self.logger.warning(f"  📊 Data Outliers ({len(validation_result.outliers)}):")
            for outlier in validation_result.outliers[:3]:
                self.logger.warning(f"    • {outlier}")
            if len(validation_result.outliers) > 3:
                self.logger.warning(f"    ... and {len(validation_result.outliers) - 3} more outliers")
        
        # Period coverage analysis
        if fundamental_data.income_statements:
            periods = []
            for stmt in fundamental_data.income_statements:
                if stmt.fiscal_period and stmt.fiscal_year:
                    periods.append(f"{stmt.fiscal_year}-{stmt.fiscal_period}")
            if periods:
                self.logger.info(f"  📅 Available Periods: {', '.join(periods)}")
        
        self.logger.info("  " + "="*50)
    
    def _log_calculator_missing_fields(self, ticker: str, fundamental_data):
        """Log missing fields for each calculator's requirements"""
        self.logger.info("  🔧 Calculator Requirements Analysis:")
        
        for calc_name, calculator in self.calculators.items():
            try:
                required_fields = calculator.get_required_fields()
                missing_for_calc = []
                
                for stmt_type, fields in required_fields.items():
                    if stmt_type == 'income_statement' and fundamental_data.income_statements:
                        latest_stmt = fundamental_data.get_latest_income_statement()
                        if latest_stmt:
                            for field in fields:
                                if not calculator.has_financial_value(latest_stmt, field):
                                    missing_for_calc.append(f"{stmt_type}.{field}")
                    
                    elif stmt_type == 'balance_sheet' and fundamental_data.balance_sheets:
                        latest_stmt = fundamental_data.get_latest_balance_sheet()
                        if latest_stmt:
                            for field in fields:
                                if not calculator.has_financial_value(latest_stmt, field):
                                    missing_for_calc.append(f"{stmt_type}.{field}")
                    
                    elif stmt_type == 'cash_flow' and fundamental_data.cash_flow_statements:
                        latest_stmt = fundamental_data.get_latest_cash_flow()
                        if latest_stmt:
                            for field in fields:
                                if not calculator.has_financial_value(latest_stmt, field):
                                    missing_for_calc.append(f"{stmt_type}.{field}")
                
                if missing_for_calc:
                    self.logger.warning(f"    🧮 {calc_name.replace('_', ' ').title()} - Missing {len(missing_for_calc)} fields:")
                    for missing_field in missing_for_calc:
                        self.logger.warning(f"      • {missing_field}")
                else:
                    self.logger.info(f"    ✅ {calc_name.replace('_', ' ').title()} - All required fields available")
                    
            except Exception as e:
                self.logger.warning(f"    ❌ {calc_name.replace('_', ' ').title()} - Error checking requirements: {e}")
    
    def _log_metrics_calculation_details(self, ticker: str, calculated_metrics: Dict[str, Any]):
        """Log detailed metrics calculation results"""
        self.logger.info(f"🧮 Metrics Calculation Results for {ticker}:")
        
        total_metrics = 0
        for calc_name, result in calculated_metrics.items():
            metrics_count = len(result.values) if hasattr(result, 'values') else 0
            total_metrics += metrics_count
            
            self.logger.info(f"  📊 {calc_name.replace('_', ' ').title()}: {metrics_count} metrics")
            
            if hasattr(result, 'calculation_errors') and result.calculation_errors:
                self.logger.warning(f"    ❌ Errors ({len(result.calculation_errors)}):")
                for error in result.calculation_errors[:3]:
                    self.logger.warning(f"      • {error}")
                if len(result.calculation_errors) > 3:
                    self.logger.warning(f"      ... and {len(result.calculation_errors) - 3} more errors")
            
            if hasattr(result, 'missing_data_count') and result.missing_data_count > 0:
                self.logger.warning(f"    🔍 Missing Data Points: {result.missing_data_count}")
            
            if hasattr(result, 'quality_score'):
                self.logger.info(f"    🎯 Quality Score: {result.quality_score:.1f}")
            
            # Show some sample calculated metrics
            if hasattr(result, 'values') and result.values:
                sample_metrics = list(result.values.items())[:3]
                self.logger.info("    📈 Sample Metrics:")
                for metric_name, value in sample_metrics:
                    if value is not None:
                        if isinstance(value, float):
                            self.logger.info(f"      • {metric_name}: {value:.4f}")
                        else:
                            self.logger.info(f"      • {metric_name}: {value}")
                    else:
                        self.logger.warning(f"      • {metric_name}: NULL")
        
        self.logger.info(f"  🎯 Total Metrics Calculated: {total_metrics}")
        self.logger.info("  " + "="*50)
    
    async def _collect_fundamental_data(self, ticker: str):
        """Collect fundamental data for a ticker"""
        try:
            # Check cache first
            cached_data = await self._get_cached_data(ticker)
            if cached_data and self._is_data_fresh(cached_data):
                self.logger.debug(f"Using cached data for {ticker}")
                return cached_data
            
            # Fetch fresh data using async context manager
            self.logger.debug(f"Fetching fresh data for {ticker}")
            async with self.api_client as client:
                data = await client.get_financials(ticker=ticker)
            
            # Cache the data
            if data:
                await self._cache_data(ticker, data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to collect data for {ticker}: {e}")
            return None
    
    async def _calculate_all_metrics(self, fundamental_data) -> Dict[str, Any]:
        """Calculate all fundamental metrics"""
        calculated_metrics = {}
        
        # Calculate ratios first (needed by scoring systems)
        if 'ratios' in self.calculators:
            try:
                self.logger.debug("Calculating ratios metrics")
                ratios_result = self.calculators['ratios'].calculate(fundamental_data)
                calculated_metrics['ratios'] = ratios_result
            except Exception as e:
                self.logger.error(f"Error calculating ratios: {e}")
        
        # Calculate other metrics, passing ratios to scoring systems
        for calc_name, calculator in self.calculators.items():
            if calc_name == 'ratios':
                continue  # Already calculated above
                
            try:
                self.logger.debug(f"Calculating {calc_name} metrics")
                
                # Pass pre-calculated ratios to scoring systems to avoid recalculation
                if calc_name == 'scoring_systems' and 'ratios' in calculated_metrics:
                    result = calculator.calculate(fundamental_data, pre_calculated_ratios=calculated_metrics['ratios'])
                else:
                    result = calculator.calculate(fundamental_data)
                    
                calculated_metrics[calc_name] = result
                
            except Exception as e:
                self.logger.error(f"Error calculating {calc_name}: {e}")
                # Continue with other calculators
        
        return calculated_metrics
    
    async def _store_results(self, ticker: str, calculated_metrics: Dict[str, Any], fundamental_data):
        """Store calculated metrics to database"""
        try:
            # Use SQLAlchemy session for ORM operations
            session = self.SessionLocal()
            try:
                # Store ratios
                if 'ratios' in calculated_metrics:
                    await self._store_ratios(session, ticker, calculated_metrics['ratios'])
                
                # Store growth metrics
                if 'growth_metrics' in calculated_metrics:
                    await self._store_growth_metrics(session, ticker, calculated_metrics['growth_metrics'])
                
                # Store scores
                if 'scoring_systems' in calculated_metrics:
                    await self._store_scores(session, ticker, calculated_metrics['scoring_systems'])
                
                # Store sector analysis
                if 'sector_analysis' in calculated_metrics:
                    await self._store_sector_analysis(session, ticker, calculated_metrics['sector_analysis'])
                
                # Commit all changes
                session.commit()
                self.logger.debug(f"Database transaction committed for {ticker}")
                
            except Exception as e:
                session.rollback()
                self.logger.error(f"Database transaction rolled back for {ticker}: {e}")
                raise
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Failed to store results for {ticker}: {e}")
            raise
    
    async def _store_ratios(self, session, ticker: str, ratios_result):
        """Store financial ratios to database using upsert"""
        from datetime import datetime
        
        # Convert date string to date object if needed
        if isinstance(ratios_result.date, str):
            result_date = datetime.strptime(ratios_result.date, '%Y-%m-%d').date()
        else:
            result_date = ratios_result.date
        
        # Prepare data for upsert - map all calculated ratios to database fields
        ratios_data = {
            'ticker': ticker,
            'date': result_date,
            
            # Market-based ratios
            'pe_ratio': ratios_result.values.get('pe_ratio'),
            'pb_ratio': ratios_result.values.get('pb_ratio'),
            'ps_ratio': ratios_result.values.get('ps_ratio'),
            'ev_ebitda': ratios_result.values.get('ev_ebitda'),
            'peg_ratio': ratios_result.values.get('peg_ratio'),
            
            # Profitability ratios
            'roe': ratios_result.values.get('roe'),
            'roa': ratios_result.values.get('roa'),
            'roi': ratios_result.values.get('roi'),
            'gross_margin': ratios_result.values.get('gross_margin'),
            'operating_margin': ratios_result.values.get('operating_margin'),
            'net_margin': ratios_result.values.get('net_margin'),
            
            # Liquidity ratios
            'current_ratio': ratios_result.values.get('current_ratio'),
            'quick_ratio': ratios_result.values.get('quick_ratio'),
            'cash_ratio': ratios_result.values.get('cash_ratio'),
            
            # Leverage ratios
            'debt_to_equity': ratios_result.values.get('debt_to_equity'),
            'debt_to_assets': ratios_result.values.get('debt_to_assets'),
            'interest_coverage': ratios_result.values.get('interest_coverage'),
            
            # Metadata
            'data_quality_score': ratios_result.quality_score / 100.0 if ratios_result.quality_score > 1 else ratios_result.quality_score,
            'missing_data_count': ratios_result.missing_data_count or 0,
            'updated_at': datetime.now()
        }
        
        # Perform upsert using PostgreSQL ON CONFLICT DO UPDATE
        stmt = insert(FundamentalRatios).values(**ratios_data)
        
        # Update all fields except primary key fields on conflict
        excluded_fields = {'ticker', 'date', 'id', 'created_at'}
        update_dict = {
            key: stmt.excluded[key] 
            for key in ratios_data.keys() 
            if key not in excluded_fields
        }
        
        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=['ticker', 'date'],
            set_=update_dict
        )
        
        session.execute(upsert_stmt)
    
    async def _store_growth_metrics(self, session, ticker: str, growth_result):
        """Store growth metrics to database using upsert"""
        from datetime import datetime
        
        # Convert date string to date object if needed
        if isinstance(growth_result.date, str):
            result_date = datetime.strptime(growth_result.date, '%Y-%m-%d').date()
        else:
            result_date = growth_result.date
        
        # Prepare data for upsert
        growth_data = {
            'ticker': ticker,
            'date': result_date,
            
            # Revenue growth (map calculated names to DB fields)
            'revenue_growth_1y': growth_result.values.get('revenue_growth_1y'),
            'revenue_growth_3y': growth_result.values.get('revenue_growth_3y'),
            
            # Earnings growth
            'earnings_growth_1y': growth_result.values.get('earnings_growth_1y'),
            
            # Asset growth
            'asset_growth_1y': growth_result.values.get('asset_growth_1y'),
            
            # Efficiency metrics
            'asset_turnover': growth_result.values.get('asset_turnover'),
            
            # Metadata
            'data_quality_score': growth_result.quality_score / 100.0 if growth_result.quality_score > 1 else growth_result.quality_score,
            'missing_data_count': 0,
            'updated_at': datetime.now()
        }
        
        # Perform upsert using PostgreSQL ON CONFLICT DO UPDATE
        stmt = insert(FundamentalGrowthMetrics).values(**growth_data)
        
        # Update all fields except primary key fields on conflict
        excluded_fields = {'ticker', 'date', 'id', 'created_at'}
        update_dict = {
            key: stmt.excluded[key] 
            for key in growth_data.keys() 
            if key not in excluded_fields
        }
        
        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=['ticker', 'date'],
            set_=update_dict
        )
        
        session.execute(upsert_stmt)
    
    async def _store_scores(self, session, ticker: str, scores_result):
        """Store scoring systems results to database using upsert"""
        from datetime import datetime
        
        # Convert date string to date object if needed
        if isinstance(scores_result.date, str):
            result_date = datetime.strptime(scores_result.date, '%Y-%m-%d').date()
        else:
            result_date = scores_result.date
        
        # Prepare data for upsert
        scores_data = {
            'ticker': ticker,
            'date': result_date,
            
            # Bankruptcy Prediction Scores
            'altman_z_score': scores_result.values.get('altman_z_score'),
            'piotroski_f_score': int(scores_result.values.get('piotroski_f_score', 0)) if scores_result.values.get('piotroski_f_score') is not None else None,
            'ohlson_o_score': scores_result.values.get('ohlson_o_score'),
            
            # Financial Health Metrics
            'working_capital_ratio': scores_result.values.get('working_capital_ratio'),
            'financial_leverage': scores_result.values.get('financial_leverage'),

            # Composite Scores
            'financial_health_composite': scores_result.values.get('financial_health_composite'),
            'quality_composite': scores_result.values.get('quality_composite'),
            'bankruptcy_risk_composite': scores_result.values.get('bankruptcy_risk_composite'),
            
            # Individual Piotroski Components (for transparency)
            'piotroski_roa_positive': scores_result.values.get('piotroski_roa_positive'),
            'piotroski_cfo_positive': scores_result.values.get('piotroski_cfo_positive'),
            'piotroski_roa_improved': scores_result.values.get('piotroski_roa_improved'),
            'piotroski_cfo_vs_roa': scores_result.values.get('piotroski_cfo_vs_roa'),
            'piotroski_debt_decreased': scores_result.values.get('piotroski_debt_decreased'),
            'piotroski_current_ratio_improved': scores_result.values.get('piotroski_current_ratio_improved'),
            'piotroski_shares_outstanding': scores_result.values.get('piotroski_shares_outstanding'),
            'piotroski_gross_margin_improved': scores_result.values.get('piotroski_gross_margin_improved'),
            'piotroski_asset_turnover_improved': scores_result.values.get('piotroski_asset_turnover_improved'),
            
            # Metadata
            'data_quality_score': scores_result.quality_score / 100.0 if scores_result.quality_score > 1 else scores_result.quality_score,
            'missing_data_count': scores_result.missing_data_count or 0,
            'updated_at': datetime.now()
        }
        
        # Perform upsert using PostgreSQL ON CONFLICT DO UPDATE
        stmt = insert(FundamentalScores).values(**scores_data)
        
        # Update all fields except primary key fields on conflict
        excluded_fields = {'ticker', 'date', 'id', 'created_at'}
        update_dict = {
            key: stmt.excluded[key] 
            for key in scores_data.keys() 
            if key not in excluded_fields
        }
        
        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=['ticker', 'date'],
            set_=update_dict
        )
        
        session.execute(upsert_stmt)
    
    async def _store_sector_analysis(self, session, ticker: str, sector_result):
        """Store sector analysis results to database using upsert"""
        from datetime import datetime
        
        # Convert date string to date object if needed
        if isinstance(sector_result.date, str):
            result_date = datetime.strptime(sector_result.date, '%Y-%m-%d').date()
        else:
            result_date = sector_result.date
        
        # Prepare data for upsert
        sector_data = {
            'ticker': ticker,
            'date': result_date,
            
            # Sector classification (these fields may not be calculated yet)
            'gics_sector': sector_result.values.get('gics_sector'),
            
            # Note: The sector analysis calculator currently only calculates basic ratios
            # The database model has many more fields for sector comparisons that aren't implemented yet
            
            # Metadata
            'data_quality_score': sector_result.quality_score / 100.0 if sector_result.quality_score > 1 else sector_result.quality_score,
            'missing_data_count': 0,
            'updated_at': datetime.now()
        }
        
        # Perform upsert using PostgreSQL ON CONFLICT DO UPDATE
        stmt = insert(FundamentalSectorAnalysis).values(**sector_data)
        
        # Update all fields except primary key fields on conflict
        excluded_fields = {'ticker', 'date', 'id', 'created_at'}
        update_dict = {
            key: stmt.excluded[key] 
            for key in sector_data.keys() 
            if key not in excluded_fields
        }
        
        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=['ticker', 'date'],
            set_=update_dict
        )
        
        session.execute(upsert_stmt)
    
    async def _get_cached_data(self, ticker: str):
        """Get cached fundamental data"""
        # TODO: Implement caching mechanism (Redis, file cache, etc.)
        return None
    
    async def _cache_data(self, ticker: str, data):
        """Cache fundamental data"""
        # TODO: Implement caching mechanism
        pass
    
    def _is_data_fresh(self, cached_data) -> bool:
        """Check if cached data is still fresh"""
        # TODO: Implement freshness check based on config
        return False
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline execution"""
        if not self.stats:
            return {"status": "not_executed"}
        
        return {
            "status": "completed" if self.stats.end_time else "running",
            "total_tickers": self.stats.total_tickers,
            "successful": self.stats.successful,
            "failed": self.stats.failed,
            "success_rate": self.stats.successful / self.stats.total_tickers if self.stats.total_tickers > 0 else 0,
            "total_metrics": self.stats.total_metrics,
            "avg_execution_time": self.stats.avg_execution_time,
            "total_duration": (
                (self.stats.end_time or datetime.now()) - self.stats.start_time
            ).total_seconds(),
            "calculators_used": list(self.calculators.keys()),
            "registered_calculators": list(FundamentalCalculatorRegistry.get_all_calculators().keys())
        }
    
    def reload_calculators(self):
        """
        Reload calculators from the registry
        
        This method can be used to pick up new calculators that may have been
        registered after pipeline initialization.
        """
        self.logger.info("Reloading calculators from registry...")
        
        # Clear existing calculators
        old_count = len(self.calculators)
        self.calculators.clear()
        
        # Get fresh calculators from registry
        available_calculators = FundamentalCalculatorRegistry.get_all_calculators()
        for name, calc_class in available_calculators.items():
            self.calculators[name] = calc_class(self.config)
        
        new_count = len(self.calculators)
        self.logger.info(f"Reloaded calculators: {old_count} -> {new_count}")
        
        if new_count != old_count:
            self.logger.info(f"Calculator count changed: {list(self.calculators.keys())}")
        
        return new_count

    def get_calculator_info(self) -> Dict[str, Any]:
        """
        Get detailed information about calculators
        
        Returns:
            Dictionary with calculator information
        """
        registered = FundamentalCalculatorRegistry.get_all_calculators()
        active = self.calculators
        
        return {
            "registered_count": len(registered),
            "active_count": len(active),
            "registered_calculators": {
                name: {
                    "class_name": calc_class.__name__,
                    "is_active": name in active,
                    "module": calc_class.__module__
                }
                for name, calc_class in registered.items()
            },
            "inactive_calculators": [
                name for name in registered.keys() if name not in active
            ]
        }

async def main():
    """Main function to run the fundamental pipeline for TSLA"""
    
    logger.info("Starting fundamental pipeline for TSLA")
    
    try:
        # Initialize pipeline with default config
        pipeline = FundamentalPipeline()
        
        # Load tickers using StockDataLoader
        with StockDataLoader() as data_loader:
            tickers = data_loader.get_available_tickers(
                min_data_points=100,  
                active_only=True,
                market='stocks',
                popular_only=True
            )
        
        logger.info(f"Loaded {len(tickers)} tickers for processing")
        
        # Process each ticker one by one
        results = []
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Processing ticker {i}/{len(tickers)}: {ticker}")
            try:
                result = await pipeline.process_ticker(ticker)
                results.append(result)
                
                if result.success:
                    logger.info(f"✓ Successfully processed {ticker} ({i}/{len(tickers)})")
                else:
                    logger.warning(f"✗ Failed to process {ticker} ({i}/{len(tickers)})")
                    
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                # Create error result
                error_result = PipelineResult(
                    ticker=ticker,
                    success=False,
                    metrics_calculated={},
                    errors=[str(e)],
                    execution_time=0.0
                )
                results.append(error_result)
        
        # Calculate overall statistics
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        logger.info(f"Pipeline completed: {len(successful_results)}/{len(tickers)} successful")
        if failed_results:
            logger.warning(f"Failed tickers: {[r.ticker for r in failed_results]}")
        
        result = results[0] if results else None  # Return first result for backward compatibility
        
        # Log results
        if result.success:
            logger.info("Successfully processed TSLA:")
            logger.info(f"  - Execution time: {result.execution_time:.2f} seconds")
            logger.info(f"  - Data quality score: {result.data_quality_score}")
            logger.info(f"  - Metrics calculated: {result.metrics_calculated}")
        else:
            logger.error("Failed to process TSLA:")
            for error in result.errors:
                logger.error(f"  - {error}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

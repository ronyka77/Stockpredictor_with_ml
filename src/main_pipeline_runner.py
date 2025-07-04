#!/usr/bin/env python3
"""
Main Pipeline Runner

This module orchestrates the complete data collection and processing workflow by running
all pipelines in the correct sequence:

1. Data Collection Pipeline (Polygon stock data)
2. News Collection Pipeline (Polygon news data)  
3. Technical Indicators Pipeline (Feature engineering)
4. Fundamental Analysis Pipeline (Fundamental metrics)

The runner provides comprehensive logging, error handling, and monitoring capabilities
for the entire data processing workflow.

Usage:
    python src/main_pipeline_runner.py
"""

import asyncio
import sys
import time
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import traceback

# Core imports
from src.utils.logger import get_logger
from src.database.connection import DatabaseConnection

# Pipeline imports
from src.data_collector.polygon_data.data_pipeline import DataPipeline
from src.data_collector.polygon_news.news_pipeline import PolygonNewsCollector
from src.feature_engineering.fundamental_indicators.fundamental_pipeline import FundamentalPipeline

# Configuration imports
from src.data_collector.config import config as data_config
from src.feature_engineering.config import config as feature_config

logger = get_logger(__name__)

class MainPipelineRunner:
    """
    Main orchestrator for all data collection and processing pipelines
    
    Runs pipelines in sequence with proper error handling, monitoring, and reporting.
    """
    
    def __init__(self):
        """Initialize the main pipeline runner in full mode"""
        self.start_time = None
        self.end_time = None
        
        # Pipeline results storage
        self.results = {
            'data_collection': None,
            'news_collection': None,
            'technical_indicators': None,
            'fundamental_analysis': None
        }
        
        # Pipeline instances
        self.data_pipeline = None
        self.news_collector = None
        self.fundamental_pipeline = None
        
        # Full mode configuration
        self.config = {
            'save_stats': True,
            'validate_data': True,
            'cleanup_on_error': True,
            'data_days_back': 365,  # 1 year of data
            'news_years_back': 1,   # 1 year of news
            'max_tickers': None,    # All available tickers
            'feature_overwrite': False,
            'parallel_workers': feature_config.batch_processing.MAX_WORKERS,
            'batch_size': feature_config.batch_processing.DEFAULT_BATCH_SIZE
        }
        
        logger.info("🚀 Initialized MainPipelineRunner in FULL mode")
        logger.info(f"📋 Configuration: {self.config}")
    
    async def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete pipeline workflow
        
        Returns:
            Dictionary with results from all pipeline stages
        """
        self.start_time = datetime.now()
        logger.info("="*80)
        logger.info("🚀 STARTING COMPLETE PIPELINE WORKFLOW - FULL MODE")
        logger.info("="*80)
        
        try:
            # Step 1: Data Collection Pipeline
            logger.info("📊 STEP 1: DATA COLLECTION PIPELINE")
            logger.info("-" * 50)
            data_result = await self._run_data_collection_pipeline()
            self.results['data_collection'] = data_result
            
            if not self._is_pipeline_successful(data_result):
                logger.error("❌ Data collection failed. Stopping pipeline.")
                return self._generate_final_report("data_collection_failed")
            
            logger.info("✅ Data collection completed successfully")
            
            # Step 2: News Collection Pipeline
            logger.info("\n📰 STEP 2: NEWS COLLECTION PIPELINE")
            logger.info("-" * 50)
            news_result = await self._run_news_collection_pipeline()
            self.results['news_collection'] = news_result
            
            if not self._is_pipeline_successful(news_result):
                logger.warning("⚠️ News collection had issues, but continuing...")
            else:
                logger.info("✅ News collection completed successfully")
            
            # Step 3: Technical Indicators Pipeline
            logger.info("\n🔧 STEP 3: TECHNICAL INDICATORS PIPELINE")
            logger.info("-" * 50)
            indicators_result = await self._run_technical_indicators_pipeline()
            self.results['technical_indicators'] = indicators_result
            
            if not self._is_pipeline_successful(indicators_result):
                logger.warning("⚠️ Technical indicators had issues, but continuing...")
            else:
                logger.info("✅ Technical indicators completed successfully")
            
            # Step 4: Fundamental Analysis Pipeline
            logger.info("\n💰 STEP 4: FUNDAMENTAL ANALYSIS PIPELINE")
            logger.info("-" * 50)
            fundamental_result = await self._run_fundamental_analysis_pipeline()
            self.results['fundamental_analysis'] = fundamental_result
            
            if not self._is_pipeline_successful(fundamental_result):
                logger.warning("⚠️ Fundamental analysis had issues")
            else:
                logger.info("✅ Fundamental analysis completed successfully")
            
            # Generate final report
            self.end_time = datetime.now()
            return self._generate_final_report("completed")
            
        except Exception as e:
            logger.error(f"❌ Pipeline workflow failed: {e}")
            logger.error(traceback.format_exc())
            self.end_time = datetime.now()
            return self._generate_final_report("failed", error=str(e))
        
        finally:
            # Cleanup resources
            await self._cleanup_resources()
    
    async def _run_data_collection_pipeline(self) -> Dict[str, Any]:
        """Run the data collection pipeline"""
        try:
            logger.info("Initializing Polygon data collection pipeline...")
            self.data_pipeline = DataPipeline(
                api_key=data_config.API_KEY,
                requests_per_minute=data_config.REQUESTS_PER_MINUTE
            )
            
            # Calculate date range
            end_date = date.today()
            start_date = end_date - timedelta(days=self.config['data_days_back'])
            
            logger.info(f"📅 Collecting data from {start_date} to {end_date}")
            logger.info("🎯 Processing ALL available tickers")
            
            # Use grouped daily pipeline for comprehensive data collection
            stats = self.data_pipeline.run_grouped_daily_pipeline(
                start_date=start_date,
                end_date=end_date,
                validate_data=self.config['validate_data'],
                save_stats=self.config['save_stats']
            )
            
            return {
                'success': stats.success_rate > 50.0,  # 50% minimum success rate
                'stats': stats,
                'execution_time': stats.duration.total_seconds(),
                'records_processed': stats.total_records_fetched,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Data collection pipeline failed: {e}")
            return {
                'success': False,
                'stats': None,
                'execution_time': 0,
                'records_processed': 0,
                'error': str(e)
            }
    
    async def _run_news_collection_pipeline(self) -> Dict[str, Any]:
        """Run the news collection pipeline"""
        try:
            logger.info("Initializing Polygon news collection pipeline...")
            
            # Create database session for news collector
            db_connection = DatabaseConnection()
            
            from sqlalchemy.orm import sessionmaker
            Session = sessionmaker(bind=db_connection.get_engine())
            session = Session()
            
            try:
                self.news_collector = PolygonNewsCollector(
                    db_session=session,
                    polygon_api_key=data_config.API_KEY,
                    requests_per_minute=data_config.REQUESTS_PER_MINUTE
                )
                
                logger.info("🎯 Processing ALL available tickers")
                
                # Historical collection
                logger.info(f"📅 Collecting {self.config['news_years_back']} years of historical news")
                stats = self.news_collector.collect_historical_news(
                    max_tickers=self.config['max_tickers'] or 50,
                    years_back=self.config['news_years_back'],
                    batch_size_days=30
                )
                
                return {
                    'success': stats['total_articles_stored'] > 0,
                    'stats': stats,
                    'execution_time': (stats['end_time'] - stats['start_time']).total_seconds() if stats['end_time'] and stats['start_time'] else 0,
                    'articles_processed': stats['total_articles_fetched'],
                    'articles_stored': stats['total_articles_stored'],
                    'error': None
                }
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"News collection pipeline failed: {e}")
            return {
                'success': False,
                'stats': None,
                'execution_time': 0,
                'articles_processed': 0,
                'articles_stored': 0,
                'error': str(e)
            }
    
    async def _run_technical_indicators_pipeline(self) -> Dict[str, Any]:
        """Run the technical indicators pipeline"""
        try:
            logger.info("Initializing technical indicators pipeline...")
            
            # Import the production batch function
            from src.feature_engineering.technical_indicators.indicator_pipeline import run_production_batch
            
            logger.info("🎯 Running production batch for technical indicators...")
            
            # Run production batch processing
            start_time = time.time()
            results = run_production_batch()
            execution_time = time.time() - start_time
            
            if results is None:
                return {
                    'success': False,
                    'stats': None,
                    'execution_time': execution_time,
                    'tickers_processed': 0,
                    'features_calculated': 0,
                    'error': 'Production batch returned None'
                }
            
            return {
                'success': results.get('success_rate', 0) > 70.0,  # 70% minimum success rate
                'stats': results,
                'execution_time': execution_time,
                'tickers_processed': results.get('successful', 0),
                'features_calculated': results.get('total_features', 0),
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Technical indicators pipeline failed: {e}")
            return {
                'success': False,
                'stats': None,
                'execution_time': 0,
                'tickers_processed': 0,
                'features_calculated': 0,
                'error': str(e)
            }
    
    async def _run_fundamental_analysis_pipeline(self) -> Dict[str, Any]:
        """Run the fundamental analysis pipeline"""
        try:
            logger.info("Initializing fundamental analysis pipeline...")
            self.fundamental_pipeline = FundamentalPipeline()
            
            # Get tickers for fundamental analysis
            from src.feature_engineering.data_loader import StockDataLoader
            with StockDataLoader() as data_loader:
                available_tickers = data_loader.get_available_tickers(
                    min_data_points=100,
                    active_only=True,
                    market='stocks',
                    popular_only=True
                )
            
            # Use all available tickers (or reasonable limit)
            selected_tickers = available_tickers[:50]  # Reasonable limit for fundamental analysis
            
            logger.info(f"🎯 Processing {len(selected_tickers)} tickers for fundamental analysis")
            
            # Run batch processing
            start_time = time.time()
            stats = await self.fundamental_pipeline.process_batch(selected_tickers)
            execution_time = time.time() - start_time
            
            return {
                'success': stats.successful > 0,
                'stats': stats,
                'execution_time': execution_time,
                'tickers_processed': stats.successful,
                'metrics_calculated': stats.total_metrics,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Fundamental analysis pipeline failed: {e}")
            return {
                'success': False,
                'stats': None,
                'execution_time': 0,
                'tickers_processed': 0,
                'metrics_calculated': 0,
                'error': str(e)
            }
    
    def _is_pipeline_successful(self, result: Dict[str, Any]) -> bool:
        """Check if a pipeline result indicates success"""
        if not result:
            return False
        return result.get('success', False) and result.get('error') is None
    
    def _generate_final_report(self, status: str, error: str = None) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        total_time = (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0
        
        report = {
            'pipeline_status': status,
            'mode': 'full',
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_execution_time': total_time,
            'configuration': self.config,
            'results': self.results,
            'error': error
        }
        
        # Calculate summary statistics
        successful_pipelines = sum(1 for r in self.results.values() if r and r.get('success', False))
        total_pipelines = len([r for r in self.results.values() if r is not None])
        
        report['summary'] = {
            'successful_pipelines': successful_pipelines,
            'total_pipelines': total_pipelines,
            'success_rate': (successful_pipelines / total_pipelines * 100) if total_pipelines > 0 else 0,
            'total_records_processed': sum(r.get('records_processed', 0) for r in self.results.values() if r),
            'total_features_calculated': sum(r.get('features_calculated', 0) for r in self.results.values() if r),
            'total_metrics_calculated': sum(r.get('metrics_calculated', 0) for r in self.results.values() if r)
        }
        
        # Log final report
        self._log_final_report(report)
        
        # Save report to file
        self._save_report(report)
        
        return report
    
    def _log_final_report(self, report: Dict[str, Any]):
        """Log the final pipeline report"""
        logger.info("="*80)
        logger.info("🎉 PIPELINE WORKFLOW COMPLETED")
        logger.info("="*80)
        
        # Status and timing
        logger.info(f"📊 Status: {report['pipeline_status'].upper()}")
        logger.info(f"⏱️  Total Execution Time: {report['total_execution_time']:.2f} seconds")
        logger.info(f"🎯 Mode: FULL")
        
        # Summary statistics
        summary = report['summary']
        logger.info(f"✅ Successful Pipelines: {summary['successful_pipelines']}/{summary['total_pipelines']}")
        logger.info(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        logger.info(f"📊 Total Records Processed: {summary['total_records_processed']:,}")
        logger.info(f"🔧 Total Features Calculated: {summary['total_features_calculated']:,}")
        logger.info(f"💰 Total Metrics Calculated: {summary['total_metrics_calculated']:,}")
        
        # Individual pipeline results
        logger.info("\n📋 INDIVIDUAL PIPELINE RESULTS:")
        logger.info("-" * 50)
        
        for pipeline_name, result in report['results'].items():
            if result:
                status_icon = "✅" if result.get('success', False) else "❌"
                logger.info(f"{status_icon} {pipeline_name.replace('_', ' ').title()}: "
                           f"{result.get('execution_time', 0):.2f}s")
                
                if result.get('error'):
                    logger.info(f"   Error: {result['error']}")
            else:
                logger.info(f"⚪ {pipeline_name.replace('_', ' ').title()}: Not executed")
        
        # Error summary
        if report.get('error'):
            logger.error(f"\n❌ Pipeline Error: {report['error']}")
        
        logger.info("="*80)
    
    def _save_report(self, report: Dict[str, Any]):
        """Save the pipeline report to file"""
        try:
            reports_dir = Path("pipeline_reports")
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = reports_dir / f"pipeline_report_full_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📄 Pipeline report saved: {report_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save pipeline report: {e}")
    
    async def _cleanup_resources(self):
        """Cleanup pipeline resources"""
        try:
            logger.info("🧹 Cleaning up pipeline resources...")
            
            if self.data_pipeline:
                self.data_pipeline.cleanup()
            
            # Close any other resources
            logger.info("✅ Resource cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during resource cleanup: {e}")

async def main():
    """Main entry point"""
    logger.info("🚀 Main Pipeline Runner Starting...")
    
    try:
        # Initialize runner in full mode
        runner = MainPipelineRunner()
        
        # Run the complete pipeline
        report = await runner.run_complete_pipeline()
        
        # Determine exit code based on results
        if report['pipeline_status'] == 'completed' and report['summary']['success_rate'] >= 75.0:
            logger.info("🎉 Pipeline completed successfully!")
            return 0
        elif report['summary']['success_rate'] >= 50.0:
            logger.warning("⚠️ Pipeline completed with some issues")
            return 1
        else:
            logger.error("❌ Pipeline failed or had major issues")
            return 2
            
    except KeyboardInterrupt:
        logger.info("🛑 Pipeline interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    """Entry point for direct execution"""
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 
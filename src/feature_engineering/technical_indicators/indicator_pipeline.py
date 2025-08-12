"""
Batch Feature Engineering Processor

This module provides batch processing capabilities to calculate technical indicators
for all tickers in the database and store results in the technical_features table.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass

from src.feature_engineering.data_loader import StockDataLoader
from src.feature_engineering.technical_indicators.feature_calculator import FeatureCalculator
from src.feature_engineering.technical_indicators.feature_storage import FeatureStorage
from src.feature_engineering.technical_indicators.consolidated_storage import ConsolidatedFeatureStorage, ConsolidatedStorageConfig
from src.utils.logger import get_logger
from src.feature_engineering.config import config

logger = get_logger(__name__, utility='feature_engineering')

@dataclass
class BatchJobConfig:
    """Configuration for batch processing jobs"""
    batch_size: int = config.batch_processing.DEFAULT_BATCH_SIZE
    max_workers: int = config.batch_processing.MAX_WORKERS
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    feature_categories: List[str] = None
    min_data_points: int = config.data_quality.MIN_DATA_POINTS
    save_to_database: bool = config.storage.SAVE_TO_DATABASE
    save_to_parquet: bool = config.storage.SAVE_TO_PARQUET
    use_consolidated_storage: bool = config.storage.USE_CONSOLIDATED_STORAGE
    partitioning_strategy: str = config.storage.PARTITIONING_STRATEGY
    overwrite_existing: bool = config.storage.OVERWRITE_EXISTING
    
    def __post_init__(self):
        if self.feature_categories is None:
            self.feature_categories = config.feature_categories.DEFAULT_CATEGORIES

@dataclass
class ProcessingStats:
    """Statistics for batch processing"""
    total_tickers: int = 0
    processed_tickers: int = 0
    failed_tickers: int = 0
    total_features: int = 0
    total_warnings: int = 0
    start_time: datetime = None
    end_time: datetime = None
    
    @property
    def success_rate(self) -> float:
        if self.total_tickers == 0:
            return 0.0
        return (self.processed_tickers / self.total_tickers) * 100
    
    @property
    def processing_time(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

class BatchFeatureProcessor:
    """
    Batch processor for calculating technical indicators across multiple tickers
    """
    
    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the batch processor
        
        Args:
            db_config: Database configuration dictionary
        """
        self.data_loader = StockDataLoader(db_config)
        self.feature_calculator = FeatureCalculator()
        self.feature_storage = FeatureStorage()
        self.consolidated_storage = ConsolidatedFeatureStorage(
            ConsolidatedStorageConfig(partitioning_strategy="by_date"),
            db_engine=self.data_loader.engine if hasattr(self.data_loader, 'engine') else None
        )
        self.stats = ProcessingStats()
        self._lock = threading.Lock()
        
        logger.info("Initialized BatchFeatureProcessor")
    
    def get_available_tickers(self, min_data_points: int = None, 
                                active_only: bool = None, market: str = None,
                                sp500_only: bool = None, popular_only: bool = None) -> List[str]:
        """
        Get list of tickers with sufficient data for processing
        
        Args:
            min_data_points: Minimum number of data points required
            active_only: Only include active tickers
            market: Market type filter ('stocks', 'crypto', 'forex', 'all')
            sp500_only: Only include S&P 500 tickers
            popular_only: Only include popular tickers
            
        Returns:
            List of ticker symbols
        """
        # Apply config defaults
        min_data_points = min_data_points or config.data_quality.MIN_DATA_POINTS
        active_only = active_only if active_only is not None else config.feature_categories.DEFAULT_ACTIVE_ONLY
        market = market or config.feature_categories.DEFAULT_MARKET
        sp500_only = sp500_only if sp500_only is not None else config.feature_categories.DEFAULT_SP500_ONLY
        popular_only = popular_only if popular_only is not None else config.feature_categories.DEFAULT_POPULAR_ONLY
        
        logger.info(f"Getting tickers with at least {min_data_points} data points")
        logger.info(f"Filters: active_only={active_only}, market={market}, sp500_only={sp500_only}, popular_only={popular_only}")
        
        try:
            tickers = self.data_loader.get_available_tickers(
                min_data_points=min_data_points,
                active_only=active_only,
                market=market,
                sp500_only=sp500_only,
                popular_only=popular_only
            )
            
            logger.info(f"Found {len(tickers)} tickers ready for processing")
            return tickers
        except Exception as e:
            logger.error(f"Error getting available tickers: {str(e)}")
            raise
    
    def process_single_ticker(self, ticker: str, config: BatchJobConfig, 
                            job_id: str) -> Dict[str, Any]:
        """
        Process a single ticker and calculate all features
        
        Args:
            ticker: Stock ticker symbol
            config: Batch job configuration
            job_id: Unique job identifier
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        result = {
            'ticker': ticker,
            'success': False,
            'features_calculated': 0,
            'warnings': 0,
            'error': None,
            'processing_time': 0.0,
            'quality_score': 0.0
        }
        
        try:
            logger.info(f"Processing ticker {ticker}")
            
            # Load stock data
            stock_data = self.data_loader.load_stock_data(
                ticker, 
                config.start_date or '2022-01-01',
                config.end_date or datetime.now().strftime('%Y-%m-%d')
            )
            
            if stock_data.empty:
                result['error'] = 'No data available'
                return result
            
            if len(stock_data) < config.min_data_points:
                result['error'] = f'Insufficient data: {len(stock_data)} < {config.min_data_points}'
                return result
            
            # Calculate features
            feature_result = self.feature_calculator.calculate_all_features(
                stock_data, 
                include_categories=config.feature_categories
            )
            
            result['features_calculated'] = len(feature_result.data.columns)
            result['warnings'] = len(feature_result.warnings)
            result['quality_score'] = feature_result.quality_score
            result['processing_time'] = time.time() - start_time
            
            # Save to storage systems
            if config.save_to_parquet:
                # Save to Parquet (primary storage)
                storage_metadata = {
                    'categories': config.feature_categories,
                    'quality_score': feature_result.quality_score,
                    'warnings': feature_result.warnings,
                    'job_id': job_id
                }
                parquet_metadata = self.feature_storage.save_features(
                    ticker, 
                    feature_result.data, 
                    storage_metadata
                )
                logger.info(f"Saved features to Parquet: {parquet_metadata.file_path}")
            
            if config.save_to_database:
                # Save to database (secondary storage)
                saved_count = self._save_features_to_database(
                    ticker, 
                    feature_result, 
                    job_id,
                    config.overwrite_existing
                )
                logger.info(f"Saved {saved_count} feature records to database for {ticker}")
            
            result['success'] = True
            logger.info(f"Successfully processed {ticker}: {result['features_calculated']} features")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error processing {ticker}: {str(e)}")
        
        return result
    
    def process_batch(self, tickers: List[str], config: BatchJobConfig) -> Dict[str, Any]:
        """
        Process a batch of tickers with parallel processing
        
        Args:
            tickers: List of ticker symbols to process
            config: Batch job configuration
            
        Returns:
            Dictionary with batch processing results
        """
        job_id = str(uuid.uuid4())
        logger.info(f"Starting batch processing job {job_id} for {len(tickers)} tickers")
        
        # Initialize job tracking
        self._create_job_record(job_id, tickers, config)
        
        # Initialize stats
        self.stats = ProcessingStats(
            total_tickers=len(tickers),
            start_time=datetime.now()
        )
        
        results = []
        failed_tickers = []
        
        try:
            # Process tickers in parallel
            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                # Submit all tasks
                future_to_ticker = {
                    executor.submit(self.process_single_ticker, ticker, config, job_id): ticker
                    for ticker in tickers
                }
                
                # Process completed tasks
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    logger.info(f"Processing ticker {ticker}")
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Update stats
                        with self._lock:
                            if result['success']:
                                self.stats.processed_tickers += 1
                                self.stats.total_features += result['features_calculated']
                                self.stats.total_warnings += result['warnings']
                            else:
                                self.stats.failed_tickers += 1
                        
                        # Log progress
                        progress = (self.stats.processed_tickers + self.stats.failed_tickers) / self.stats.total_tickers * 100
                        logger.info(f"Progress: {progress:.1f}% - Processed {ticker} ({'✓' if result['success'] else '✗'})")
                        
                    except Exception as e:
                        logger.error(f"Error processing {ticker}: {str(e)}")
                        failed_tickers.append(ticker)
                        with self._lock:
                            self.stats.failed_tickers += 1
            
            self.stats.end_time = datetime.now()
            
            # Update job record
            self._update_job_record(job_id, self.stats, failed_tickers)
            
            # Prepare summary
            summary = {
                'job_id': job_id,
                'total_tickers': self.stats.total_tickers,
                'successful': self.stats.processed_tickers,
                'failed': self.stats.failed_tickers,
                'success_rate': self.stats.success_rate,
                'total_features': self.stats.total_features,
                'total_warnings': self.stats.total_warnings,
                'processing_time': self.stats.processing_time,
                'failed_tickers': failed_tickers,
                'results': results
            }
            
            logger.info(f"Batch processing completed: {self.stats.processed_tickers}/{self.stats.total_tickers} successful")
            logger.info(f"Total features calculated: {self.stats.total_features}")
            logger.info(f"Processing time: {self.stats.processing_time:.2f} seconds")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            self._update_job_record(job_id, self.stats, failed_tickers, error=str(e))
            raise
    
    def process_all_tickers(self, config: BatchJobConfig) -> Dict[str, Any]:
        """
        Process all available tickers in the database
        
        Args:
            config: Batch job configuration
            
        Returns:
            Dictionary with processing results
        """
        logger.info("Starting full database processing")
        
        # Get all available tickers
        all_tickers = self.get_available_tickers(config.min_data_points)
        
        if not all_tickers:
            logger.warning("No tickers found for processing")
            return {'error': 'No tickers available for processing'}
        
        logger.info(f"Processing {len(all_tickers)} tickers in batches of {config.batch_size}")
        
        # Process in batches
        all_results = []
        total_successful = 0
        total_failed = 0
        total_features = 0
        
        for i in range(0, len(all_tickers), config.batch_size):
            batch_tickers = all_tickers[i:i + config.batch_size]
            batch_num = (i // config.batch_size) + 1
            total_batches = (len(all_tickers) + config.batch_size - 1) // config.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_tickers)} tickers)")
            
            try:
                batch_result = self.process_batch(batch_tickers, config)
                all_results.append(batch_result)
                
                total_successful += batch_result['successful']
                total_failed += batch_result['failed']
                total_features += batch_result['total_features']
                
                logger.info(f"Batch {batch_num} completed: {batch_result['successful']}/{len(batch_tickers)} successful")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {str(e)}")
                total_failed += len(batch_tickers)
        
        # Final summary
        summary = {
            'total_tickers': len(all_tickers),
            'successful': total_successful,
            'failed': total_failed,
            'success_rate': (total_successful / len(all_tickers)) * 100 if all_tickers else 0,
            'total_features': total_features,
            'batch_results': all_results
        }
        
        logger.info(f"Full processing completed: {total_successful}/{len(all_tickers)} tickers successful")
        logger.info(f"Total features calculated: {total_features}")
        
        return summary
    
    def _save_features_to_database(self, ticker: str, feature_result, 
                                    job_id: str, overwrite: bool = False) -> int:
        """
        Save calculated features to the database
        
        Args:
            ticker: Stock ticker symbol
            feature_result: FeatureResult object
            job_id: Job identifier
            overwrite: Whether to overwrite existing features
            
        Returns:
            Number of records saved
        """
        try:
            from sqlalchemy import text
            
            saved_count = 0
            
            with self.data_loader.engine.connect() as conn:
                for date_idx, row in feature_result.data.iterrows():
                    for feature_name, feature_value in row.items():
                        if pd.isna(feature_value) or np.isinf(feature_value):
                            continue
                        
                        # Convert to native Python types and check range
                        feature_value = float(feature_value)
                        
                        # Skip values that are too large for database precision (15,6)
                        # Database can handle values up to 999,999,999.999999
                        if abs(feature_value) >= 1e9:
                            logger.info(f"Skipping {feature_name} value {feature_value} - too large for database precision")
                            continue
                        
                        # Determine feature category
                        category = self._get_feature_category(feature_name)
                        
                        # Check if feature already exists
                        if not overwrite:
                            check_query = text("""
                                SELECT COUNT(*) as count FROM technical_features 
                                WHERE ticker = :ticker AND date = :date 
                                AND feature_category = :category AND feature_name = :name
                            """)
                            
                            result = conn.execute(check_query, {
                                'ticker': ticker,
                                'date': date_idx.date(),
                                'category': category,
                                'name': feature_name
                            }).fetchone()
                            
                            if result and result[0] > 0:
                                continue
                        
                        # Insert or update feature
                        insert_query = text("""
                            INSERT INTO technical_features 
                            (ticker, date, feature_category, feature_name, feature_value, quality_score)
                            VALUES (:ticker, :date, :category, :name, :value, :quality)
                            ON CONFLICT (ticker, date, feature_category, feature_name) 
                            DO UPDATE SET 
                                feature_value = EXCLUDED.feature_value,
                                quality_score = EXCLUDED.quality_score,
                                calculation_timestamp = CURRENT_TIMESTAMP
                        """)
                        
                        conn.execute(insert_query, {
                            'ticker': ticker,
                            'date': date_idx.date(),
                            'category': category,
                            'name': feature_name,
                            'value': feature_value,
                            'quality': float(feature_result.quality_score.item() if hasattr(feature_result.quality_score, 'item') else feature_result.quality_score)
                        })
                        
                        saved_count += 1
                
                conn.commit()
            
            return saved_count
            
        except Exception as e:
            logger.error(f"Error saving features for {ticker}: {str(e)}")
            raise
    
    def _get_feature_category(self, feature_name: str) -> str:
        """Determine feature category from feature name"""
        feature_name_lower = feature_name.lower()
        
        if any(x in feature_name_lower for x in ['sma', 'ema', 'macd', 'ichimoku']):
            return 'trend'
        elif any(x in feature_name_lower for x in ['rsi', 'stoch', 'roc', 'williams']):
            return 'momentum'
        elif any(x in feature_name_lower for x in ['bb', 'bollinger', 'atr', 'volatility']):
            return 'volatility'
        elif any(x in feature_name_lower for x in ['obv', 'vpt', 'ad_line', 'volume', 'mfi']):
            return 'volume'
        else:
            return 'basic'
    
    def _create_job_record(self, job_id: str, tickers: List[str], config: BatchJobConfig):
        """Create a job tracking record in the database"""
        try:
            from sqlalchemy import text
            
            with self.data_loader.engine.connect() as conn:
                query = text("""
                    INSERT INTO feature_calculation_jobs 
                    (job_id, ticker, start_date, end_date, feature_categories, status)
                    VALUES (:job_id, :ticker, :start_date, :end_date, :categories, 'running')
                """)
                
                conn.execute(query, {
                    'job_id': job_id,
                    'ticker': f"{len(tickers)} tickers",
                    'start_date': config.start_date,
                    'end_date': config.end_date,
                    'categories': config.feature_categories
                })
                
                conn.commit()
                
        except Exception as e:
            logger.warning(f"Could not create job record: {str(e)}")
    
    def _update_job_record(self, job_id: str, stats: ProcessingStats, 
                            failed_tickers: List[str], error: str = None):
        """Update job tracking record with results"""
        try:
            from sqlalchemy import text
            
            status = 'failed' if error else ('completed' if stats.failed_tickers == 0 else 'partial_success')
            
            with self.data_loader.engine.connect() as conn:
                query = text("""
                    UPDATE feature_calculation_jobs 
                    SET status = :status,
                        total_features_calculated = :features,
                        total_warnings = :warnings,
                        error_message = :error,
                        completed_at = CURRENT_TIMESTAMP
                    WHERE job_id = :job_id
                """)
                
                conn.execute(query, {
                    'job_id': job_id,
                    'status': status,
                    'features': stats.total_features,
                    'warnings': stats.total_warnings,
                    'error': error
                })
                
                conn.commit()
                
        except Exception as e:
            logger.warning(f"Could not update job record: {str(e)}")
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return {
            'total_tickers': self.stats.total_tickers,
            'processed': self.stats.processed_tickers,
            'failed': self.stats.failed_tickers,
            'success_rate': self.stats.success_rate,
            'total_features': self.stats.total_features,
            'processing_time': self.stats.processing_time,
            'is_running': self.stats.end_time is None
        }
    
    def close(self):
        """Close database connections"""
        if hasattr(self.data_loader, 'close'):
            self.data_loader.close()

def run_production_batch():
    """Run production batch processing for all available tickers"""
    print("🚀 Starting Production Feature Engineering Batch...")
    
    # Production configuration
    job_config = BatchJobConfig(
        batch_size=config.batch_processing.DEFAULT_BATCH_SIZE // 5,  # Conservative batch size
        max_workers=config.batch_processing.MAX_WORKERS // 2,  # Conservative parallel processing
        start_date='2024-01-01',  # 2 years of data
        end_date=datetime.now().strftime('%Y-%m-%d'),
        min_data_points=config.data_quality.MIN_DATA_POINTS // 2,  # Reduced requirement for production
        save_to_parquet=config.storage.SAVE_TO_PARQUET,
        save_to_database=False,  # Skip database for now
        overwrite_existing=config.storage.OVERWRITE_EXISTING
    )
    
    processor = BatchFeatureProcessor()
    
    try:
        # Get all available tickers with sufficient data
        print("📊 Getting all available tickers...")
        
        # Get all tickers that meet minimum data requirements, ordered by data points desc
        all_tickers = processor.get_available_tickers(
            min_data_points=job_config.min_data_points,
            active_only=True,  # Only process active tickers
            market='stocks'    # Focus on stocks market
        )
                
        print(f"   Found {len(all_tickers)} tickers with sufficient data")
        
        # Use all tickers ordered by data points (descending order from get_available_tickers)
        print("📈 Using all tickers ordered by data points (descending)...")
        
        # All tickers are already ordered by data points desc from the query
        selected_tickers = all_tickers
        
        print(f"📈 Processing {len(selected_tickers)} tickers:")
        print(f"   Sample tickers: {', '.join(selected_tickers[:10])}{'...' if len(selected_tickers) > 10 else ''}")
        
        # Run batch processing
        start_time = time.time()
        results = processor.process_batch(selected_tickers, job_config)
        processing_time = time.time() - start_time
        
        # Print results
        print("\n🎉 Batch Processing Completed!")
        print(f"   Total tickers: {results['total_tickers']}")
        print(f"   Successful: {results['successful']}")
        print(f"   Failed: {results['failed']}")
        print(f"   Success rate: {results['success_rate']:.1f}%")
        print(f"   Total features: {results['total_features']:,}")
        print(f"   Processing time: {processing_time:.1f} seconds")
        
        # Check storage stats
        storage = FeatureStorage()
        stats = storage.get_storage_stats()
        print("\n📁 Storage Statistics:")
        print(f"   Total tickers stored: {stats['total_tickers']}")
        print(f"   Total storage size: {stats['total_size_mb']:.2f} MB")
        print(f"   Storage path: {stats['base_path']}")
        
        # Show failed tickers if any
        if results['failed'] > 0:
            failed_tickers = [r['ticker'] for r in results['results'] if not r['success']]
            print(f"\n⚠️  Failed tickers: {', '.join(failed_tickers[:10])}")
        
        # Consolidate into year-based partitions
        if results['successful'] > 0:
            print("\n🗓️ Consolidating features into year-based partitions...")
            try:
                from src.feature_engineering.technical_indicators.consolidated_storage import consolidate_existing_features
                
                consolidation_start = time.time()
                consolidation_result = consolidate_existing_features(strategy='by_year')
                consolidation_time = time.time() - consolidation_start
                
                print(f"✅ Year-based consolidation completed in {consolidation_time:.2f} seconds")
                print(f"   Year-partitioned files: {consolidation_result['files_created']}")
                print(f"   Consolidated size: {consolidation_result['total_size_mb']:.2f} MB")
                print(f"   Compression ratio: {consolidation_result['compression_ratio']:.1f}x")
                
                # Show year breakdown
                print("\n📁 Year-based Files:")
                for file_info in consolidation_result['files']:
                    print(f"   {file_info['file']}: {file_info['rows']:,} rows, Year: {file_info['year']}")
                
                results['consolidation'] = consolidation_result
                
            except Exception as e:
                print(f"⚠️  Consolidation failed: {str(e)}")
                results['consolidation_error'] = str(e)
        
        return results
        
    except Exception as e:
        print(f"❌ Error in production batch: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        processor.close()

def main():
    """Main function for production batch processing"""
    results = run_production_batch()
    if results and results['success_rate'] > 80.0:  # 80% minimum success rate
        print("\n✅ Production batch completed successfully!")
        if 'consolidation' in results:
            print("✅ Year-based consolidation completed successfully!")
            print("\n💡 Ready for ML workflows:")
            print("   ✅ Train on 2024 data, test on 2025")
            print("   ✅ Fast year-specific loading")
            print("   ✅ Memory efficient processing")
    else:
        print("\n⚠️  Production batch completed with issues.")

if __name__ == "__main__":
    main() 
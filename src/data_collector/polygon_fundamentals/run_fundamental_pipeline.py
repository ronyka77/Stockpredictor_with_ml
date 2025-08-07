#!/usr/bin/env python3
"""
Fundamental Data Pipeline Runner

This script runs the complete fundamental data pipeline:
1. Database setup and verification
2. Full fundamental data collection for all active tickers
3. Comprehensive results reporting and statistics
"""

import asyncio
from typing import Dict, Any
from datetime import datetime

from src.data_collector.polygon_fundamentals.setup_database import main as setup_database
from src.data_collector.polygon_fundamentals.optimized_processor import OptimizedFundamentalProcessor
from src.data_collector.polygon_fundamentals.monitor import FundamentalDataMonitor
from src.utils.logger import get_logger

logger = get_logger(__name__)

def print_results(stats: Dict[str, Any], progress: Dict[str, Any], quality: Dict[str, Any]):
    """Print collection results in a formatted way"""
    print("\n" + "="*60)
    print("FUNDAMENTAL DATA COLLECTION RESULTS")
    print("="*60)
    
    # Collection Statistics
    print("\n📊 Collection Statistics:")
    print(f"   Total tickers processed: {stats['total']}")
    print(f"   Successful: {stats['successful']}")
    print(f"   Failed: {stats['failed']}")
    print(f"   Success rate: {stats['success_rate']:.2%}")
    
    if stats['failed_tickers']:
        print(f"   Failed tickers: {', '.join(stats['failed_tickers'][:10])}")
        if len(stats['failed_tickers']) > 10:
            print(f"   ... and {len(stats['failed_tickers']) - 10} more")
    
    # Performance Statistics
    collector_stats = stats['collector_stats']
    if collector_stats['start_time'] and collector_stats['end_time']:
        elapsed = (collector_stats['end_time'] - collector_stats['start_time']).total_seconds()
        print("\n⏱️  Performance:")
        print(f"   Total time: {elapsed/60:.1f} minutes")
        print(f"   Average rate: {collector_stats['total_processed']/elapsed*60:.1f} tickers/minute")
        print(f"   Skipped: {collector_stats['skipped']} tickers")
    
    # Progress Overview
    print("\n📈 Overall Progress:")
    print(f"   Overall progress: {progress.get('overall_progress', 0):.2%}")
    print(f"   Recent data (30 days): {progress.get('recent_data_count', 0)} tickers")
    
    # Data Quality
    print("\n🔍 Data Quality:")
    print(f"   Average quality score: {quality.get('average_quality_score', 0):.2f}")
    
    field_completeness = quality.get('field_completeness', {})
    print("   Field completeness:")
    for field, completeness in field_completeness.items():
        print(f"     {field}: {completeness:.2%}")
    
    print("\n" + "="*60)
    print("Collection complete!")
    print("="*60)

async def run_collection():
    """Run fundamental data collection for all active tickers"""
    processor = OptimizedFundamentalProcessor()
    monitor = FundamentalDataMonitor()
    
    try:
        logger.info("Starting fundamental data collection for all active tickers...")
        
        # Always process all fundamentals
        results = await processor.process_all_fundamentals()
        
        # Get statistics and display results
        stats = processor.get_collection_stats(results)
        progress = monitor.get_collection_progress()
        quality = monitor.get_data_quality_summary()
        
        # Print results
        print_results(stats, progress, quality)
        
        return {
            'results': results,
            'stats': stats,
            'progress': progress,
            'quality': quality,
            'timestamp': datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        return {
            'error': str(e),
            'timestamp': datetime.now()
        }
    finally:
        processor.close()

async def main():
    """Run the complete fundamental data pipeline"""
    print("🚀 Starting Fundamental Data Pipeline")
    print("=" * 50)
    
    # Step 1: Database Setup
    print("\n📊 Step 1: Database Setup")
    print("-" * 30)
    setup_success = await setup_database()
    
    if not setup_success:
        print("❌ Database setup failed. Exiting.")
        return False
    
    print("✅ Database setup completed successfully")
    
    # Step 2: Run Collection
    print("\n📈 Step 2: Fundamental Data Collection")
    print("-" * 40)
    print("Starting collection for all active tickers...")
    print("This may take several hours depending on the number of tickers.")
    
    try:
        results = await run_collection()
        
        if 'error' in results:
            print(f"❌ Collection failed: {results['error']}")
            return False
        else:
            print("✅ Collection completed successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Collection failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🎉 Fundamental Data Pipeline completed successfully!")
    else:
        print("\n💥 Fundamental Data Pipeline failed!")
        exit(1) 
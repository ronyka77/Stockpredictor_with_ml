#!/usr/bin/env python3
"""
Fundamental Data Collection Runner

Main execution script for optimized fundamental data collection.
Supports different collection modes with progress tracking.
"""

import asyncio
import argparse
import logging
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.data_collector.polygon_fundamentals.optimized_processor import OptimizedFundamentalProcessor
from src.data_collector.polygon_fundamentals.monitor import FundamentalDataMonitor
from src.utils.logger import get_logger
from src.data_collector.polygon_fundamentals.db_pool import get_connection_pool, close_connection_pool

logger = get_logger(__name__)

class FundamentalCollectionRunner:
    """Main runner for fundamental data collection"""
    
    def __init__(self):
        self.processor = OptimizedFundamentalProcessor()
        self.monitor = FundamentalDataMonitor()
    
    async def run_collection(self, mode: str, tickers: List[str] = None) -> Dict[str, Any]:
        """
        Run fundamental data collection based on mode
        
        Args:
            mode: Collection mode ('all', 'custom')
            tickers: Custom ticker list (for 'custom' mode)
            
        Returns:
            Collection results and statistics
        """
        actual_mode = mode if mode else 'all'
        logger.info(f"Starting fundamental data collection in {actual_mode} mode")
        
        try:
            if mode == 'all' or mode is None:
                results = await self.processor.process_all_fundamentals()
            elif mode == 'custom':
                if not tickers:
                    raise ValueError("Custom mode requires ticker list")
                results = await self.processor.process_custom_tickers(tickers)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            # Get statistics
            stats = self.processor.get_collection_stats(results)
            
            # Get monitoring data
            progress = self.monitor.get_collection_progress()
            quality = self.monitor.get_data_quality_summary()
            
            return {
                'results': results,
                'stats': stats,
                'progress': progress,
                'quality': quality,
                'mode': actual_mode,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Collection failed: {e}")
            return {
                'error': str(e),
                'mode': mode,
                'timestamp': datetime.now()
            }
    
    def print_results(self, collection_data: Dict[str, Any]):
        """Print collection results in a formatted way"""
        if 'error' in collection_data:
            logger.error(f"Collection failed: {collection_data['error']}")
            return
        
        stats = collection_data['stats']
        progress = collection_data['progress']
        quality = collection_data['quality']
        
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

async def main():
    """Main execution function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Fundamental Data Collection Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Collect all active tickers (default behavior)
            python run_fundamental_collection.py
            
            # Collect all active tickers (explicit)
            python run_fundamental_collection.py --mode all
            
            # Collect custom ticker list
            python run_fundamental_collection.py --mode custom --tickers AAPL MSFT GOOGL
            
            # Monitor only (no collection)
            python run_fundamental_collection.py --monitor-only
    """
    )
    
    parser.add_argument(
        '--mode',
        choices=['all', 'custom'],
        default=None,
        help='Collection mode (default: all - processes all active tickers)'
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='Custom ticker list for custom mode'
    )
    
    parser.add_argument(
        '--monitor-only',
        action='store_true',
        help='Run monitoring only, no collection'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if args.mode == 'custom' and not args.tickers:
        logger.error("Custom mode requires --tickers argument")
        sys.exit(1)
    
    runner = FundamentalCollectionRunner()
    
    if args.monitor_only:
        # Monitor only mode
        logger.info("Running monitoring only...")
        
        progress = runner.monitor.get_collection_progress()
        quality = runner.monitor.get_data_quality_summary()
        recent = runner.monitor.get_recent_activity(days=7)
        
        print("\n" + "="*60)
        print("FUNDAMENTAL DATA COLLECTION MONITOR")
        print("="*60)
        
        print("\n📊 Collection Progress:")
        print(f"   Overall progress: {progress.get('overall_progress', 0):.2%}")
        print(f"   Recent data (30 days): {progress.get('recent_data_count', 0)} tickers")
        
        print("\n🔍 Data Quality:")
        print(f"   Average quality score: {quality.get('average_quality_score', 0):.2f}")
        
        field_completeness = quality.get('field_completeness', {})
        print("   Field completeness:")
        for field, completeness in field_completeness.items():
            print(f"     {field}: {completeness:.2%}")
        
        print("\n📅 Recent Activity (7 days):")
        print(f"   New records: {len(recent)}")
        
        if recent:
            print("   Recent additions:")
            for record in recent[:5]:
                print(f"     {record['ticker']}: {record['fiscal_period']} {record['fiscal_year']}")
        
        print("\n" + "="*60)
        
    else:
        # Collection mode
        logger.info(f"Starting collection in {args.mode} mode...")
        
        if args.mode == 'custom':
            collection_data = await runner.run_collection(args.mode, args.tickers)
        else:
            collection_data = await runner.run_collection(args.mode)
        
        runner.print_results(collection_data)

if __name__ == "__main__":
    asyncio.run(main()) 
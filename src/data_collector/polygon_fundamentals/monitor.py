"""
Fundamental Data Collection Monitor

This script monitors the progress and quality of fundamental data collection.
"""

import asyncio
from typing import Dict, List, Any

from src.database.connection import DatabaseConnectionPool
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FundamentalDataMonitor:
    """Monitor for fundamental data collection progress"""
    
    def __init__(self):
        self.db_pool = DatabaseConnectionPool(
            min_connections=1,
            max_connections=5
        )
    
    def get_collection_progress(self) -> Dict[str, Any]:
        """Get overall collection progress"""
        try:
            with self.db_pool.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Get total tickers
                    cursor.execute("SELECT COUNT(*) FROM tickers WHERE active = true")
                    total_tickers = cursor.fetchone()['count']
                    
                    # Get tickers with fundamental data
                    cursor.execute("""
                        SELECT COUNT(DISTINCT ticker_id) 
                        FROM raw_fundamental_data
                    """)
                    tickers_with_data = cursor.fetchone()['count']
                    
                    # Get recent data (last 30 days)
                    cursor.execute("""
                        SELECT COUNT(DISTINCT ticker_id) 
                        FROM raw_fundamental_data 
                        WHERE date >= CURRENT_DATE - INTERVAL '30 days'
                    """)
                    recent_data = cursor.fetchone()['count']
                    
                    return {
                        'total_tickers': total_tickers,
                        'tickers_with_data': tickers_with_data,
                        'overall_progress': tickers_with_data / total_tickers if total_tickers > 0 else 0,
                        'recent_data_count': recent_data
                    }
                
        except Exception as e:
            logger.error(f"Failed to get collection progress: {e}")
            return {}
    
    def get_data_quality_summary(self) -> Dict[str, Any]:
        """Get data quality summary"""
        try:
            with self.db_pool.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Average data quality score
                    cursor.execute("""
                        SELECT AVG(data_quality_score) 
                        FROM raw_fundamental_data 
                        WHERE data_quality_score IS NOT NULL
                    """)
                    avg_quality = cursor.fetchone()['avg']
                    
                    # Missing data distribution
                    cursor.execute("""
                        SELECT 
                            missing_data_count,
                            COUNT(*) as ticker_count
                        FROM raw_fundamental_data 
                        GROUP BY missing_data_count 
                        ORDER BY missing_data_count
                    """)
                    missing_data_stats = cursor.fetchall()
                    
                    # Completeness by field
                    cursor.execute("""
                        SELECT 
                            COUNT(CASE WHEN revenues IS NOT NULL THEN 1 END) as revenues_count,
                            COUNT(CASE WHEN net_income_loss IS NOT NULL THEN 1 END) as net_income_count,
                            COUNT(CASE WHEN assets IS NOT NULL THEN 1 END) as assets_count,
                            COUNT(CASE WHEN net_cash_flow_from_operating_activities IS NOT NULL THEN 1 END) as cash_flow_count,
                            COUNT(*) as total_records
                        FROM raw_fundamental_data
                    """)
                    field_completeness = cursor.fetchone()
                    
                    return {
                        'average_quality_score': float(avg_quality) if avg_quality else 0,
                        'missing_data_distribution': [
                            {'missing_count': row['missing_data_count'], 'ticker_count': row['ticker_count']} 
                            for row in missing_data_stats
                        ],
                        'field_completeness': {
                            'revenues': field_completeness['revenues_count'] / field_completeness['total_records'] if field_completeness['total_records'] > 0 else 0,
                            'net_income': field_completeness['net_income_count'] / field_completeness['total_records'] if field_completeness['total_records'] > 0 else 0,
                            'assets': field_completeness['assets_count'] / field_completeness['total_records'] if field_completeness['total_records'] > 0 else 0,
                            'cash_flow': field_completeness['cash_flow_count'] / field_completeness['total_records'] if field_completeness['total_records'] > 0 else 0
                        }
                    }
                
        except Exception as e:
            logger.error(f"Failed to get data quality summary: {e}")
            return {}
    
    def get_recent_activity(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent collection activity"""
        try:
            with self.db_pool.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT 
                            t.ticker,
                            t.name as company_name,
                            rfd.date,
                            rfd.fiscal_period,
                            rfd.fiscal_year,
                            rfd.data_quality_score,
                            rfd.missing_data_count,
                            rfd.created_at
                        FROM raw_fundamental_data rfd
                        JOIN tickers t ON rfd.ticker_id = t.id
                        WHERE rfd.created_at >= CURRENT_DATE - INTERVAL %s || ' days'
                        ORDER BY rfd.created_at DESC
                    """, (days,))
                    
                    recent_activity = cursor.fetchall()
                    
                    return [
                        {
                            'ticker': row['ticker'],
                            'company_name': row['company_name'],
                            'date': row['date'],
                            'fiscal_period': row['fiscal_period'],
                            'fiscal_year': row['fiscal_year'],
                            'quality_score': row['data_quality_score'],
                            'missing_count': row['missing_data_count'],
                            'created_at': row['created_at']
                        }
                        for row in recent_activity
                    ]
                
        except Exception as e:
            logger.error(f"Failed to get recent activity: {e}")
            return []
    
    def close(self):
        """Close connection pool when done"""
        if hasattr(self, 'db_pool') and self.db_pool:
            self.db_pool.close()
            logger.debug("Connection pool closed")

async def main():
    """Main monitoring function"""
    monitor = FundamentalDataMonitor()
    try:
        # Get progress
        progress = monitor.get_collection_progress()
        logger.info("=== Collection Progress ===")
        logger.info(f"Overall progress: {progress.get('overall_progress', 0):.2%}")
        logger.info(f"Recent data (30 days): {progress.get('recent_data_count', 0)} tickers")
        
        # Get quality summary
        quality = monitor.get_data_quality_summary()
        logger.info("=== Data Quality ===")
        logger.info(f"Average quality score: {quality.get('average_quality_score', 0):.2f}")
        
        field_completeness = quality.get('field_completeness', {})
        logger.info("Field completeness:")
        for field, completeness in field_completeness.items():
            logger.info(f"  {field}: {completeness:.2%}")
        
        # Get recent activity
        recent = monitor.get_recent_activity(days=7)
        logger.info("=== Recent Activity (7 days) ===")
        logger.info(f"New records: {len(recent)}")
        
        if recent:
            logger.info("Recent additions:")
            for record in recent[:5]:  # Show first 5
                logger.info(f"  {record['ticker']}: {record['fiscal_period']} {record['fiscal_year']}")
    finally:
        # Ensure connection pool is closed
        monitor.close()

if __name__ == "__main__":
    asyncio.run(main()) 
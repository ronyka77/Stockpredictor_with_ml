"""
Fundamental Data Collection Monitor

This script monitors the progress and quality of fundamental data collection.
"""

from typing import Dict, List, Any

from src.data_collector.polygon_fundamentals.db_pool import get_connection_pool
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FundamentalDataMonitor:
    """Monitor for fundamental data collection progress"""
    
    def __init__(self):
        self.db_pool = get_connection_pool()
    
    def get_collection_progress(self) -> Dict[str, Any]:
        """Get overall collection progress"""
        try:
            with self.db_pool.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Get total tickers
                    cursor.execute("SELECT COUNT(*) FROM tickers WHERE active = true and has_financials = true")
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
                            ticker_id,
                            date,
                            filing_date,
                            fiscal_period,
                            fiscal_year,
                            data_quality_score,
                            missing_data_count
                        FROM raw_fundamental_data 
                        WHERE date >= CURRENT_DATE - INTERVAL '%s days'
                        ORDER BY date DESC
                        LIMIT 50
                    """, (days,))
                    
                    recent_activity = []
                    for row in cursor.fetchall():
                        recent_activity.append({
                            'ticker_id': row['ticker_id'],
                            'date': row['date'].isoformat() if row['date'] else None,
                            'filing_date': row['filing_date'].isoformat() if row['filing_date'] else None,
                            'fiscal_period': row['fiscal_period'],
                            'fiscal_year': row['fiscal_year'],
                            'data_quality_score': float(row['data_quality_score']) if row['data_quality_score'] else 0,
                            'missing_data_count': row['missing_data_count']
                        })
                    
                    return recent_activity
                
        except Exception as e:
            logger.error(f"Failed to get recent activity: {e}")
            return []
    
    def close(self):
        """Close the monitor and cleanup resources"""
        # The pool will be closed by the main application when needed
        logger.info("FundamentalDataMonitor closed")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.close()
        except Exception as e:
            logger.error(f"Error in destructor: {e}")
            pass 
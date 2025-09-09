"""
Fundamental Data Collection Monitor

This script monitors the progress and quality of fundamental data collection.
"""

from typing import Dict, List, Any

from src.database.connection import get_global_pool, fetch_one, fetch_all
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FundamentalDataMonitor:
    """Monitor for fundamental data collection progress"""

    def __init__(self):
        self.db_pool = get_global_pool()

    def get_collection_progress(self) -> Dict[str, Any]:
        """Get overall collection progress"""
        try:
            total_row = fetch_one(
                "SELECT COUNT(*) as count FROM tickers WHERE active = true and has_financials = true"
            )
            total_tickers = int(total_row.get("count") if total_row else 0)

            tickers_with_data_row = fetch_one(
                "SELECT COUNT(DISTINCT ticker_id) as count FROM raw_fundamental_data"
            )
            tickers_with_data = int(
                tickers_with_data_row.get("count") if tickers_with_data_row else 0
            )

            recent_row = fetch_one(
                "SELECT COUNT(DISTINCT ticker_id) as count FROM raw_fundamental_data WHERE date >= CURRENT_DATE - INTERVAL '30 days'"
            )
            recent_data = int(recent_row.get("count") if recent_row else 0)

            return {
                "total_tickers": total_tickers,
                "tickers_with_data": tickers_with_data,
                "overall_progress": tickers_with_data / total_tickers
                if total_tickers > 0
                else 0,
                "recent_data_count": recent_data,
            }

        except Exception as e:
            logger.error(f"Failed to get collection progress: {e}")
            return {}

    def get_data_quality_summary(self) -> Dict[str, Any]:
        """Get data quality summary"""
        try:
            avg_row = fetch_one(
                "SELECT AVG(data_quality_score) as avg FROM raw_fundamental_data WHERE data_quality_score IS NOT NULL"
            )
            avg_quality = (
                float(avg_row.get("avg"))
                if avg_row and avg_row.get("avg") is not None
                else 0
            )

            missing_data_stats = fetch_all(
                "SELECT missing_data_count, COUNT(*) as ticker_count FROM raw_fundamental_data GROUP BY missing_data_count ORDER BY missing_data_count"
            )

            field_completeness = fetch_one(
                "SELECT COUNT(CASE WHEN revenues IS NOT NULL THEN 1 END) as revenues_count, COUNT(CASE WHEN net_income_loss IS NOT NULL THEN 1 END) as net_income_count, COUNT(CASE WHEN assets IS NOT NULL THEN 1 END) as assets_count, COUNT(CASE WHEN net_cash_flow_from_operating_activities IS NOT NULL THEN 1 END) as cash_flow_count, COUNT(*) as total_records FROM raw_fundamental_data"
            )

            total_records = (
                field_completeness.get("total_records") if field_completeness else 0
            )

            return {
                "average_quality_score": avg_quality,
                "missing_data_distribution": [
                    {
                        "missing_count": row["missing_data_count"],
                        "ticker_count": row["ticker_count"],
                    }
                    for row in (missing_data_stats or [])
                ],
                "field_completeness": {
                    "revenues": (
                        field_completeness.get("revenues_count", 0) / total_records
                    )
                    if total_records and total_records > 0
                    else 0,
                    "net_income": (
                        field_completeness.get("net_income_count", 0) / total_records
                    )
                    if total_records and total_records > 0
                    else 0,
                    "assets": (
                        field_completeness.get("assets_count", 0) / total_records
                    )
                    if total_records and total_records > 0
                    else 0,
                    "cash_flow": (
                        field_completeness.get("cash_flow_count", 0) / total_records
                    )
                    if total_records and total_records > 0
                    else 0,
                },
            }

        except Exception as e:
            logger.error(f"Failed to get data quality summary: {e}")
            return {}

    def get_recent_activity(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent collection activity"""
        try:
            rows = fetch_all(
                "SELECT ticker_id, date, filing_date, fiscal_period, fiscal_year, data_quality_score, missing_data_count FROM raw_fundamental_data WHERE date >= CURRENT_DATE - INTERVAL %s ORDER BY date DESC LIMIT 50",
                (f"{days} days",),
            )

            recent_activity = []
            for row in rows or []:
                recent_activity.append(
                    {
                        "ticker_id": row["ticker_id"],
                        "date": row["date"].isoformat() if row.get("date") else None,
                        "filing_date": row["filing_date"].isoformat()
                        if row.get("filing_date")
                        else None,
                        "fiscal_period": row.get("fiscal_period"),
                        "fiscal_year": row.get("fiscal_year"),
                        "data_quality_score": float(row.get("data_quality_score"))
                        if row.get("data_quality_score")
                        else 0,
                        "missing_data_count": row.get("missing_data_count"),
                    }
                )

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

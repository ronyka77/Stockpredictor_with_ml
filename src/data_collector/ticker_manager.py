"""
Ticker management and discovery functionality for Polygon.io data
"""

from typing import List, Dict, Optional

from src.utils.core.logger import get_logger
from src.data_collector.polygon_data.client import PolygonDataClient
from src.data_collector.polygon_data.data_validator import DataValidator
from src.data_collector.polygon_data.data_storage import DataStorage

logger = get_logger(__name__, utility="data_collector")


class TickerManager:
    """
    Manages ticker discovery, filtering, and caching for efficient data acquisition
    """

    def __init__(self, client: PolygonDataClient, storage: Optional[DataStorage] = None):
        """
        Initialize the ticker manager

        Args:
            client: Polygon.io API client
            storage: Database storage instance (optional, will create if not provided)
        """
        self.client = client
        self.storage = storage or DataStorage()
        self.validator = DataValidator(strict_mode=False)

    def get_ticker_details(self, ticker: str) -> Optional[Dict]:
        """
        Get detailed information about a specific ticker from database

        Args:
            ticker: Stock ticker symbol

        Returns:
            Ticker details dictionary or None if not found
        """
        try:
            # Get ticker from database using a more efficient query
            with self.storage.engine.connect() as conn:
                from sqlalchemy import text

                result = conn.execute(
                    text("SELECT * FROM tickers WHERE ticker = :ticker LIMIT 1"),
                    {"ticker": ticker.upper()},
                )
                row = result.fetchone()

                if row:
                    # Convert database row to dictionary
                    details = dict(row._mapping)
                    logger.info(f"Retrieved details for {ticker} from database")
                    return details

        except Exception as e:
            logger.error(f"Failed to get ticker details from database: {e}")

        logger.warning(f"No details found for ticker {ticker}")
        return None

    def refresh_ticker_details(self, tickers: List[str], batch_size: int = 50) -> Dict[str, int]:
        """
        Refresh detailed information for specific tickers

        Args:
            tickers: List of ticker symbols to refresh
            batch_size: Number of tickers to process in each batch

        Returns:
            Dictionary with refresh statistics
        """
        logger.info(f"🔍 Refreshing detailed information for {len(tickers)} tickers")

        stats = {"processed": 0, "updated": 0, "errors": 0, "not_found": 0}

        try:
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i : i + batch_size]
                logger.info(f"📋 Processing batch {i // batch_size + 1}: {len(batch)} tickers")

                for ticker in batch:
                    try:
                        stats["processed"] += 1

                        # Fetch details from API
                        details = self.client.get_ticker_details(ticker)

                        if details:
                            # Prepare ticker data for database
                            ticker_data = {
                                "ticker": details.get("ticker", ticker).upper(),
                                "name": details.get("name"),
                                "market": details.get("market", "stocks"),
                                "locale": details.get("locale", "us"),
                                "primary_exchange": details.get("primary_exchange"),
                                "currency_name": details.get("currency_name"),
                                "active": details.get("active", True),
                                "type": details.get("type"),
                                "market_cap": details.get("market_cap"),
                                "weighted_shares_outstanding": details.get(
                                    "weighted_shares_outstanding"
                                ),
                                "round_lot": details.get("round_lot"),
                                "cik": details.get("cik"),
                                "composite_figi": details.get("composite_figi"),
                                "share_class_figi": details.get("share_class_figi"),
                                "sic_code": details.get("sic_code"),
                                "sic_description": details.get("sic_description"),
                                "ticker_root": details.get("ticker_root"),
                                "total_employees": details.get("total_employees"),
                                "list_date": details.get("list_date"),
                            }

                            # Store in database
                            result = self.storage.store_tickers([ticker_data])
                            if result["stored_count"] > 0:
                                stats["updated"] += 1
                                logger.info(f"✅ Updated details for {ticker}")
                        else:
                            stats["not_found"] += 1
                            logger.warning(f"⚠️ No details found for {ticker}")

                    except Exception as e:
                        stats["errors"] += 1
                        logger.error(f"❌ Error refreshing {ticker}: {e}")
                        continue

                # Add small delay between batches to respect rate limits
                import time

                time.sleep(0.1)

            logger.info("🎉 Ticker details refresh completed!")
            logger.info("📊 Details Refresh Summary:")
            logger.info(f"   Processed: {stats['processed']}")
            logger.info(f"   Updated: {stats['updated']}")
            logger.info(f"   Not found: {stats['not_found']}")
            logger.info(f"   Errors: {stats['errors']}")

            return stats

        except Exception as e:
            logger.error(f"❌ Ticker details refresh failed: {e}")
            raise


def main():
    """
    Main function to refresh ticker data in database
    """
    from src.utils.core.logger import get_general_logger

    logger = get_general_logger(__name__)

    try:
        from src.data_collector.polygon_data import PolygonDataClient, DataStorage
        from src.data_collector.config import config

        logger.info("🚀 Starting ticker refresh")

        # Initialize components
        storage = DataStorage()
        client = PolygonDataClient(api_key=config.API_KEY)
        ticker_manager = TickerManager(client, storage)

        ticker_manager.refresh_ticker_details(
            tickers=[t["ticker"] for t in ticker_manager.storage.get_tickers()], batch_size=50
        )

        logger.info("🎉 Ticker refresh completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"❌ Ticker refresh failed: {e}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())

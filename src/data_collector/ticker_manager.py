"""
Ticker management and discovery functionality for Polygon.io data
"""

from typing import List, Dict, Optional

from src.utils.logger import get_logger
from src.data_collector.polygon_data.client import PolygonDataClient
from src.data_collector.polygon_data.data_validator import DataValidator
from src.data_collector.polygon_data.data_storage import DataStorage

logger = get_logger(__name__, utility="data_collector")


class TickerManager:
    """
    Manages ticker discovery, filtering, and caching for efficient data acquisition
    """

    def __init__(
        self, client: PolygonDataClient, storage: Optional[DataStorage] = None
    ):
        """
        Create a TickerManager tied to a PolygonDataClient and optional persistent storage.
        
        If `storage` is not provided a new DataStorage instance will be created. A
        DataValidator is constructed in non-strict mode for incoming ticker data.
        """
        self.client = client
        self.storage = storage or DataStorage()
        self.validator = DataValidator(strict_mode=False)

    def get_ticker_details(self, ticker: str) -> Optional[Dict]:
        """
        Retrieve detailed ticker record from the database.
        
        Ticker symbol is uppercased before querying. Returns a dictionary of column names to values for the first matching row, or None if no record is found or an error occurs (errors are logged).
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

    def resolve_ticker_to_id(self, ticker: str) -> Optional[int]:
        """
        Resolve a stock ticker symbol to its database id.
        
        Looks up the ticker via self.storage.get_tickers and returns the 'id' of the first matching record. If the input ticker is empty, no match is found, or an error occurs during lookup, the function returns None.
        """
        if not ticker:
            logger.warning("resolve_ticker_to_id called with empty ticker")
            return None

        try:
            
            storage_tickers = self.storage.get_tickers(ticker)

            if storage_tickers:
                return storage_tickers[0]["id"]

        except Exception as e:
            logger.error(f"Failed to resolve ticker to id for {ticker}: {e}")

        logger.warning(f"Ticker id not found for {ticker}")
        return None

    def refresh_ticker_details(
        self, tickers: List[str], batch_size: int = 50
    ) -> Dict[str, int]:
        """
        Refresh detailed metadata for a list of tickers by querying the Polygon client and persisting results to storage.
        
        For each ticker the method fetches details from self.client.get_ticker_details, transforms selected fields into a storage-friendly dict (ticker is uppercased when persisted), and calls self.storage.store_tickers in batches. Processing is performed in batches of size `batch_size` with a short pause between batches to help respect rate limits.
        
        Parameters:
            tickers (List[str]): Ticker symbols to refresh.
            batch_size (int): Number of tickers to process per batch (default: 50).
        
        Returns:
            Dict[str, int]: A stats dictionary with counts:
                - "processed": total tickers attempted
                - "updated": number of tickers whose details were stored/updated
                - "not_found": tickers for which no details were returned by the client
                - "errors": tickers that raised exceptions during processing
        """
        logger.info(f"🔍 Refreshing detailed information for {len(tickers)} tickers")

        stats = {"processed": 0, "updated": 0, "errors": 0, "not_found": 0}

        try:
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i : i + batch_size]
                logger.info(
                    f"📋 Processing batch {i // batch_size + 1}: {len(batch)} tickers"
                )

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
    Entry point that refreshes ticker details stored in the database.
    
    This function initializes the data storage and Polygon API client (using config.API_KEY),
    creates a TickerManager, and runs refresh_ticker_details for all tickers returned by storage.
    It logs progress and returns a POSIX-style exit code.
    
    Returns:
        int: 0 on successful completion, 1 on any error.
    """
    from src.utils.logger import get_general_logger

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
            tickers=ticker_manager.storage.get_tickers(),
            batch_size=50,
        )

        logger.info("🎉 Ticker refresh completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"❌ Ticker refresh failed: {e}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())

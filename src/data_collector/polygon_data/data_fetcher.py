"""
Historical data fetching functionality for Polygon.io API
"""

from datetime import datetime, date
from typing import List, Dict, Optional, Any, Union
import time

from src.utils.logger import get_logger
from src.data_collector.polygon_data.client import PolygonDataClient, PolygonAPIError
from src.data_collector.polygon_data.data_validator import DataValidator, OHLCVRecord, DataQualityMetrics
from src.data_collector.config import config

logger = get_logger(__name__, utility="data_collector")


class HistoricalDataFetcher:
    """
    Fetches and processes historical OHLCV data from Polygon.io
    
    Handles data transformation, validation, and quality checks for historical stock data.
    """
    
    def __init__(self, client: PolygonDataClient, validator: Optional[DataValidator] = None):
        """
        Initialize the historical data fetcher
        
        Args:
            client: Polygon.io API client
            validator: Data validator instance (optional)
        """
        self.client = client
        self.validator = validator or DataValidator(strict_mode=False)
        
        logger.info("Historical data fetcher initialized")
    
    def get_historical_data(self, ticker: str, start_date: Union[str, date], 
                            end_date: Union[str, date], timespan: str = "day",
                            multiplier: int = 1,
                            validate_data: bool = True) -> tuple[List[OHLCVRecord], DataQualityMetrics]:
        """
        Fetch historical OHLCV data for a ticker
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD string or date object)
            end_date: End date (YYYY-MM-DD string or date object)
            timespan: Time window (day, week, month, quarter, year)
            multiplier: Size of the timespan multiplier
            validate_data: Whether to validate the data
            
        Returns:
            Tuple of (validated_records, quality_metrics)
        """
        # Convert dates to strings if needed
        start_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, date) else start_date
        end_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, date) else end_date
        
        logger.info(f"Fetching {timespan} data for {ticker} from {start_str} to {end_str}")
        
        try:
            # Fetch raw data from Polygon.io
            raw_data = self.client.get_aggregates(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_date=start_str,
                to_date=end_str,
                limit=config.MAX_RECORDS_PER_REQUEST
            )
            
            if not raw_data:
                logger.warning(f"No data returned for {ticker}")
                return [], DataQualityMetrics()
            
            logger.info(f"Retrieved {len(raw_data)} raw records for {ticker}")
            
            # Transform and validate data if requested
            if validate_data:
                validated_records, metrics = self.validator.validate_ohlcv_batch(
                    raw_data, ticker=ticker
                )
                return validated_records, metrics
            else:
                # Transform without validation
                transformed_records = []
                for record in raw_data:
                    transformed = self._transform_polygon_record(record, ticker)
                    if transformed:
                        try:
                            validated_record = OHLCVRecord(**transformed)
                            transformed_records.append(validated_record)
                        except Exception as e:
                            logger.info(f"Failed to create record: {e}")
                            continue
                
                metrics = DataQualityMetrics()
                metrics.total_records = len(raw_data)
                metrics.valid_records = len(transformed_records)
                
                return transformed_records, metrics
                
        except PolygonAPIError as e:
            logger.error(f"API error fetching data for {ticker}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {ticker}: {e}")
            raise
    
    def get_bulk_historical_data(self, tickers: List[str], start_date: Union[str, date],
                                end_date: Union[str, date], timespan: str = "day",
                                batch_size: int = 10, delay_between_batches: float = 1.0,
                                validate_data: bool = True) -> Dict[str, tuple[List[OHLCVRecord], DataQualityMetrics]]:
        """
        Fetch historical data for multiple tickers in batches
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            timespan: Time window
            batch_size: Number of tickers to process in each batch
            delay_between_batches: Delay between batches in seconds
            validate_data: Whether to validate the data
            
        Returns:
            Dictionary mapping ticker to (records, metrics) tuple
        """
        logger.info(f"Fetching bulk historical data for {len(tickers)} tickers")
        
        results = {}
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(tickers))
            batch_tickers = tickers[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_num + 1}/{total_batches}: "
                        f"tickers {start_idx + 1}-{end_idx}")
            
            for ticker in batch_tickers:
                try:
                    records, metrics = self.get_historical_data(
                        ticker=ticker,
                        start_date=start_date,
                        end_date=end_date,
                        timespan=timespan,
                        validate_data=validate_data
                    )
                    results[ticker] = (records, metrics)
                    
                    logger.info(f"Completed {ticker}: {len(records)} records, "
                                f"{metrics.success_rate:.1f}% success rate")
                    
                except Exception as e:
                    logger.error(f"Failed to fetch data for {ticker}: {e}")
                    results[ticker] = ([], DataQualityMetrics())
            
            # Delay between batches – skip when rate limiting disabled
            if batch_num < total_batches - 1 and delay_between_batches > 0:
                if not getattr(config, 'DISABLE_RATE_LIMITING', False):
                    logger.info(f"Waiting {delay_between_batches}s before next batch")
                    time.sleep(delay_between_batches)
        
        successful_tickers = sum(1 for records, _ in results.values() if records)
        logger.info(f"Bulk fetch complete: {successful_tickers}/{len(tickers)} tickers successful")
        
        return results
    
    def get_grouped_daily_data(self, target_date: Union[str, date],
                                validate_data: bool = True) -> Dict[str, OHLCVRecord]:
        """
        Get grouped daily data for all stocks on a specific date
        
        Args:
            target_date: Date to fetch data for
            validate_data: Whether to validate the data
            
        Returns:
            Dictionary mapping ticker to OHLCV record
        """
        # Convert date to string if needed
        date_str = target_date.strftime('%Y-%m-%d') if isinstance(target_date, date) else target_date
        
        logger.info(f"Fetching grouped daily data for {date_str}")
        
        try:
            raw_data = self.client.get_grouped_daily(date_str)
            
            if not raw_data:
                logger.warning(f"No grouped data returned for {date_str}")
                return {}
            
            logger.info(f"Retrieved {len(raw_data)} grouped records for {date_str}")
            
            results = {}
            
            for record in raw_data:
                ticker = record.get('T')  # Ticker symbol in Polygon format
                if not ticker:
                    continue
                
                try:
                    transformed = self._transform_polygon_record(record, ticker)
                    if transformed:
                        if validate_data:
                            validated_record = self.validator.validate_ohlcv_record(transformed)
                            if validated_record:
                                results[ticker] = validated_record
                        else:
                            results[ticker] = OHLCVRecord(**transformed)
                            
                except Exception as e:
                    logger.info(f"Failed to process grouped record for {ticker}: {e}")
                    continue
            
            logger.info(f"Processed {len(results)} valid grouped records")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching grouped daily data: {e}")
            raise
    
    def _transform_polygon_record(self, polygon_record: Dict[str, Any], ticker: str) -> Optional[Dict[str, Any]]:
        """
        Transform Polygon.io API response format to internal format
        
        Args:
            polygon_record: Raw record from Polygon.io API
            ticker: Ticker symbol to add to record
            
        Returns:
            Transformed record dictionary or None if transformation fails
        """
        try:
            # Handle different timestamp formats
            timestamp = polygon_record.get('t')
            if timestamp:
                # Polygon.io uses milliseconds since epoch
                if isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp / 1000).date()
            else:
                # If no timestamp, try to use current date (for grouped data)
                timestamp = date.today()
            
            # Extract OHLCV data
            transformed = {
                'ticker': ticker,
                'timestamp': timestamp,
                'open': float(polygon_record.get('o', 0)),
                'high': float(polygon_record.get('h', 0)),
                'low': float(polygon_record.get('l', 0)),
                'close': float(polygon_record.get('c', 0)),
                'volume': int(polygon_record.get('v', 0)),
            }
            
            # Add optional fields if available
            if 'vw' in polygon_record and polygon_record['vw'] is not None:
                transformed['vwap'] = float(polygon_record['vw'])
            
            # For adjusted data, use close as adjusted_close
            if polygon_record.get('c') is not None:
                transformed['adjusted_close'] = float(polygon_record['c'])
            
            # Basic validation - ensure all required fields are positive
            required_fields = ['open', 'high', 'low', 'close']
            for field in required_fields:
                if transformed[field] <= 0:
                    logger.info(f"Invalid {field} value: {transformed[field]}")
                    return None
            
            # Ensure volume is non-negative
            if transformed['volume'] < 0:
                logger.info(f"Invalid volume value: {transformed['volume']}")
                return None
            
            return transformed
            
        except (ValueError, TypeError, KeyError) as e:
            logger.info(f"Failed to transform record: {e}")
            return None
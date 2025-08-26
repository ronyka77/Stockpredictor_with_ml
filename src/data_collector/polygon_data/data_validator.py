"""
Data validation schemas and validators for OHLCV stock market data
"""

from datetime import datetime, date
from typing import List, Dict, Optional, Any, Union, Tuple
from pydantic import BaseModel, field_validator, field_serializer, Field, ConfigDict
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger(__name__, utility="data_collector")


class MarketType(str, Enum):
    """Enumeration of supported market types"""
    STOCKS = "stocks"
    CRYPTO = "crypto"
    FOREX = "fx"
    OPTIONS = "options"
    INDICES = "indices"


class OHLCVRecord(BaseModel):
    """
    Validated OHLCV (Open, High, Low, Close, Volume) record
    
    This model ensures data integrity and consistency for stock market data.
    """
    
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    timestamp: Union[datetime, date] = Field(..., description="Date/time of the record")
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="Highest price")
    low: float = Field(..., gt=0, description="Lowest price")
    close: float = Field(..., gt=0, description="Closing price")
    volume: int = Field(..., ge=0, description="Trading volume")
    vwap: Optional[float] = Field(None, gt=0, description="Volume weighted average price")
    adjusted_close: Optional[float] = Field(None, gt=0, description="Adjusted closing price")
    
    # Pydantic V2 configuration
    model_config = ConfigDict(
        validate_assignment=True,
    )

    @field_serializer("timestamp")
    def _serialize_timestamp(self, v, _info):
        if isinstance(v, (datetime, date)):
            return v.isoformat()
        return v
    
    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v):
        """Validate ticker symbol format"""
        if not v or not v.strip():
            raise ValueError('Ticker cannot be empty')
        
        # Clean ticker symbol
        ticker = v.strip().upper()
        
        # Basic ticker validation (alphanumeric + some special chars)
        if not ticker.replace('.', '').replace('-', '').replace('/', '').isalnum():
            raise ValueError(f'Invalid ticker format: {ticker}')
        
        return ticker
    
    @field_validator('high', mode='after')
    @classmethod
    def validate_high_price(cls, v, info):
        values = info.data or {}
        """Validate that high price is the highest among OHLC"""
        if 'low' in values and v < values['low']:
            raise ValueError(f'High price ({v}) must be >= Low price ({values["low"]})')
        
        if 'open' in values and v < values['open']:
            raise ValueError(f'High price ({v}) must be >= Open price ({values["open"]})')
        
        if 'close' in values and v < values['close']:
            raise ValueError(f'High price ({v}) must be >= Close price ({values["close"]})')
        
        return v
    
    @field_validator('low', mode='after')
    @classmethod
    def validate_low_price(cls, v, info):
        values = info.data or {}
        """Validate that low price is the lowest among OHLC"""
        if 'open' in values and v > values['open']:
            raise ValueError(f'Low price ({v}) must be <= Open price ({values["open"]})')
        
        if 'close' in values and v > values['close']:
            raise ValueError(f'Low price ({v}) must be <= Close price ({values["close"]})')
        
        return v
    
    @field_validator('volume')
    @classmethod
    def validate_volume(cls, v):
        """Validate trading volume"""
        if v < 0:
            raise ValueError('Volume cannot be negative')
        
        # Check for unreasonably high volume (potential data error)
        if v > 1e12:  # 1 trillion shares
            raise ValueError(f'Volume appears unreasonably high: {v}')
        
        return v
    
    @field_validator('vwap', mode='after')
    @classmethod
    def validate_vwap(cls, v, info):
        values = info.data or {}
        """Validate volume weighted average price with fallback calculation"""
        if v is None:
            return v
        
        # VWAP should be positive
        if v <= 0:
            # Calculate fallback VWAP using typical price
            if all(key in values for key in ['high', 'low', 'close']):
                volume = values.get('volume', None)
                fallback_vwap = cls._calculate_fallback_vwap(values['high'], values['low'], values['close'], volume)
                logger.warning(f"Invalid VWAP {v}, using fallback calculation: {fallback_vwap}")
                return fallback_vwap
            else:
                raise ValueError('VWAP must be positive')
        
        # VWAP should be reasonable compared to the day's prices (allow some tolerance)
        # VWAP can legitimately be outside the Low-High range due to volume weighting
        if 'high' in values and 'low' in values:
            # Allow VWAP to be up to 20% outside the Low-High range
            tolerance = 0.20
            low_bound = values['low'] * (1 - tolerance)
            high_bound = values['high'] * (1 + tolerance)
            
            if not (low_bound <= v <= high_bound):
                # Calculate fallback VWAP using typical price
                volume = values.get('volume', None)
                fallback_vwap = cls._calculate_fallback_vwap(values['high'], values['low'], values['close'], volume)
                logger.warning(
                    f'API VWAP ({v}) is too far from trading range [{values["low"]}, {values["high"]}], '
                    f'using fallback calculation: {fallback_vwap}'
                )
                return fallback_vwap
        
        return v
    
    @staticmethod
    def _calculate_fallback_vwap(high: float, low: float, close: float, volume: Optional[int] = None) -> float:
        """
        Calculate fallback VWAP using typical price method
        
        For single-record VWAP calculation, we use different approaches:
        1. If volume is available: Use close price weighted by volume (simplified VWAP)
        2. If no volume: Use typical price formula: (High + Low + Close) / 3
        
        This is a reasonable approximation when we don't have intraday volume data.
        
        Args:
            high: High price of the day
            low: Low price of the day
            close: Close price of the day
            volume: Trading volume (optional)
            
        Returns:
            Calculated VWAP approximation
        """
        if volume and volume > 0:
            # Use a weighted approach favoring close price
            # This approximates VWAP by giving more weight to close price
            # Formula: (High + Low + 2*Close) / 4
            vwap = (high + low + 2 * close) / 4
        else:
            # Use typical price when no volume data
            vwap = (high + low + close) / 3
        
        return round(vwap, 4)
    
    @field_validator('adjusted_close')
    @classmethod
    def validate_adjusted_close(cls, v, info):
        # info.data contains other fields if needed
        """Validate adjusted closing price"""
        if v is None:
            return v
        
        # Adjusted close should be positive
        if v <= 0:
            raise ValueError('Adjusted close price must be positive')
        
        return v
    
    @field_validator('timestamp', mode='before')
    @classmethod
    def validate_timestamp(cls, v):
        """Validate timestamp"""
        if isinstance(v, str):
            try:
                # Try to parse ISO format
                if 'T' in v:
                    return datetime.fromisoformat(v.replace('Z', '+00:00'))
                else:
                    return datetime.strptime(v, '%Y-%m-%d').date()
            except ValueError:
                raise ValueError(f'Invalid timestamp format: {v}')
        
        # Check if timestamp is not in the future (with some tolerance)
        if isinstance(v, datetime):
            if v.date() > datetime.now().date():
                raise ValueError(f'Timestamp cannot be in the future: {v}')
        elif isinstance(v, date):
            if v > datetime.now().date():
                raise ValueError(f'Date cannot be in the future: {v}')
        
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        data = self.model_dump()        
        # Convert timestamp to date for database storage
        if isinstance(data['timestamp'], datetime):
            data['date'] = data['timestamp'].date()
        else:
            data['date'] = data['timestamp']
        
        # Remove timestamp field for database
        data.pop('timestamp', None)
        
        return data


class TickerInfo(BaseModel):
    """Validated ticker information"""
    
    ticker: str = Field(..., description="Ticker symbol")
    name: str = Field(..., description="Company/asset name")
    market: MarketType = Field(..., description="Market type")
    locale: str = Field(default="us", description="Market locale")
    primary_exchange: Optional[str] = Field(None, description="Primary exchange")
    currency_name: Optional[str] = Field(None, description="Trading currency")
    active: bool = Field(default=True, description="Whether ticker is active")
    
    @field_validator('ticker')
    @classmethod
    def validate_ticker_format(cls, v):
        """Validate ticker format"""
        return v.strip().upper()

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate company/asset name"""
        if not v or not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()


class DataQualityMetrics(BaseModel):
    """Metrics for data quality assessment"""
    
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    validation_errors: List[str] = []
    data_gaps: List[str] = []
    outliers: List[str] = []
    
    @property
    def success_rate(self) -> float:
        """Calculate validation success rate"""
        if self.total_records == 0:
            return 0.0
        return (self.valid_records / self.total_records) * 100
    
    def add_error(self, error: str) -> None:
        """Add a validation error"""
        self.validation_errors.append(error)
        self.invalid_records += 1
    
    def add_outlier(self, outlier: str) -> None:
        """Add an outlier detection"""
        self.outliers.append(outlier)
    
    def add_gap(self, gap: str) -> None:
        """Add a data gap detection"""
        self.data_gaps.append(gap)


class DataValidator:
    """
    Comprehensive data validator for stock market data
    
    Provides validation, quality checks, and outlier detection for OHLCV data.
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize the data validator
        
        Args:
            strict_mode: Whether to use strict validation rules
        """
        self.strict_mode = strict_mode
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}", utility="data_collector")
    
    def validate_ohlcv_record(self, record: Dict[str, Any], ticker: Optional[str] = None) -> Optional[OHLCVRecord]:
        """
        Validate a single OHLCV record
        
        Args:
            record: Raw OHLCV data dictionary
            ticker: Optional ticker symbol to add if missing
            
        Returns:
            Validated OHLCVRecord or None if validation fails
        """
        try:
            # Transform Polygon.io format if needed
            if 't' in record:  # Polygon.io timestamp format
                transformed_record = self._transform_polygon_record(record, ticker)
            else:
                transformed_record = record
                # Add ticker if provided and not present
                if ticker and 'ticker' not in transformed_record:
                    transformed_record['ticker'] = ticker
            
            validated_record = OHLCVRecord(**transformed_record)
            return validated_record
            
        except Exception as e:
            self.logger.warning(f"Validation failed for record: {e}")
            if self.strict_mode:
                raise
            return None
    
    def validate_ohlcv_batch(self, records: List[Dict[str, Any]], 
                            ticker: Optional[str] = None) -> Tuple[List[OHLCVRecord], DataQualityMetrics]:
        """
        Validate a batch of OHLCV records
        
        Args:
            records: List of raw OHLCV data dictionaries
            ticker: Optional ticker symbol for context
            
        Returns:
            Tuple of (validated_records, quality_metrics)
        """
        validated_records = []
        metrics = DataQualityMetrics()
        metrics.total_records = len(records)
        
        self.logger.info(f"Validating {len(records)} records" + 
                        (f" for {ticker}" if ticker else ""))
        
        for i, record in enumerate(records):
            try:
                # Add ticker to record if provided and not present
                if ticker and 'ticker' not in record:
                    record['ticker'] = ticker
                
                validated_record = self.validate_ohlcv_record(record, ticker)
                
                if validated_record:
                    validated_records.append(validated_record)
                    metrics.valid_records += 1
                else:
                    metrics.add_error(f"Record {i}: Validation failed")
                    
            except Exception as e:
                error_msg = f"Record {i}: {str(e)}"
                metrics.add_error(error_msg)
                self.logger.info(error_msg)
        
        # Perform additional quality checks
        if validated_records:
            self._check_data_quality(validated_records, metrics)
        
        self.logger.info(f"Validation complete: {metrics.valid_records}/{metrics.total_records} "
                        f"records valid ({metrics.success_rate:.1f}%)")
        
        return validated_records, metrics
    
    def _transform_polygon_record(self, polygon_record: Dict[str, Any], ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Transform Polygon.io API response format to internal format
        
        Args:
            polygon_record: Raw record from Polygon.io API
            ticker: Optional ticker symbol to use if not in record
            
        Returns:
            Transformed record dictionary
        """
        # Handle different timestamp formats
        timestamp = polygon_record.get('t')
        if timestamp:
            # Polygon.io uses milliseconds since epoch
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp / 1000)
        
        # Get ticker from record or use provided ticker
        record_ticker = polygon_record.get('T', '')
        if not record_ticker and ticker:
            record_ticker = ticker
        
        transformed = {
            'ticker': record_ticker,
            'timestamp': timestamp,
            'open': polygon_record.get('o', 0),     # Open
            'high': polygon_record.get('h', 0),     # High
            'low': polygon_record.get('l', 0),      # Low
            'close': polygon_record.get('c', 0),    # Close
            'volume': polygon_record.get('v', 0),   # Volume
            'vwap': polygon_record.get('vw'),       # Volume weighted average price
        }
        
        # Remove None values
        return {k: v for k, v in transformed.items() if v is not None}
    
    def _check_data_quality(self, records: List[OHLCVRecord], 
                            metrics: DataQualityMetrics) -> None:
        """
        Perform additional data quality checks
        
        Args:
            records: List of validated records
            metrics: Quality metrics to update
        """
        if not records:
            return
        
        # Sort records by timestamp for gap detection
        sorted_records = sorted(records, key=lambda x: x.timestamp)
        
        # Check for data gaps (missing trading days)
        self._detect_data_gaps(sorted_records, metrics)
        
        # Check for price outliers
        self._detect_price_outliers(sorted_records, metrics)
        
        # Check for volume outliers
        self._detect_volume_outliers(sorted_records, metrics)
    
    def _detect_data_gaps(self, sorted_records: List[OHLCVRecord], 
                            metrics: DataQualityMetrics) -> None:
        """Detect gaps in trading data"""
        if len(sorted_records) < 2:
            return
        
        for i in range(1, len(sorted_records)):
            prev_date = sorted_records[i-1].timestamp
            curr_date = sorted_records[i].timestamp
            
            # Convert to date if datetime
            if isinstance(prev_date, datetime):
                prev_date = prev_date.date()
            if isinstance(curr_date, datetime):
                curr_date = curr_date.date()
            
            # Check for gaps > 3 days (accounting for weekends)
            gap_days = (curr_date - prev_date).days
            if gap_days > 3:
                gap_msg = f"Data gap: {gap_days} days between {prev_date} and {curr_date}"
                metrics.add_gap(gap_msg)
    
    def _detect_price_outliers(self, records: List[OHLCVRecord], 
                                metrics: DataQualityMetrics) -> None:
        """Detect price outliers using statistical methods"""
        if len(records) < 10:  # Need sufficient data for outlier detection
            return
        
        # Calculate price changes
        price_changes = []
        for i in range(1, len(records)):
            prev_close = records[i-1].close
            curr_close = records[i].close
            change_pct = abs((curr_close - prev_close) / prev_close) * 100
            price_changes.append(change_pct)
        
        if not price_changes:
            return
        
        # Detect outliers (changes > 20% in a single day)
        for i, change in enumerate(price_changes):
            if change > 20:  # 20% change threshold
                date = records[i+1].timestamp
                outlier_msg = f"Price outlier: {change:.1f}% change on {date}"
                metrics.add_outlier(outlier_msg)
    
    def _detect_volume_outliers(self, records: List[OHLCVRecord], 
                                metrics: DataQualityMetrics) -> None:
        """Detect volume outliers"""
        if len(records) < 10:
            return
        
        volumes = [r.volume for r in records if r.volume > 0]
        if not volumes:
            return
        
        # Calculate average volume
        avg_volume = sum(volumes) / len(volumes)
        
        # Detect volumes that are 10x average or more
        for record in records:
            if record.volume > avg_volume * 10:
                outlier_msg = f"Volume outlier: {record.volume:,} vs avg {avg_volume:,.0f} on {record.timestamp}"
                metrics.add_outlier(outlier_msg)
    
    def validate_ticker_info(self, ticker_data: Dict[str, Any]) -> Optional[TickerInfo]:
        """
        Validate ticker information
        
        Args:
            ticker_data: Raw ticker data dictionary
            
        Returns:
            Validated TickerInfo or None if validation fails
        """
        try:
            # Transform Polygon.io ticker format if needed
            if 'ticker' in ticker_data:
                transformed_data = {
                    'ticker': ticker_data['ticker'],
                    'name': ticker_data.get('name', ''),
                    'market': ticker_data.get('market', 'stocks'),
                    'locale': ticker_data.get('locale', 'us'),
                    'primary_exchange': ticker_data.get('primary_exchange'),
                    'currency_name': ticker_data.get('currency_name'),
                    'active': ticker_data.get('active', True)
                }
            else:
                transformed_data = ticker_data
            
            return TickerInfo(**transformed_data)
            
        except Exception as e:
            self.logger.warning(f"Ticker validation failed: {e}")
            if self.strict_mode:
                raise
            return None
    
    @staticmethod
    def calculate_batch_vwap(records: List[Dict[str, Any]]) -> float:
        """
        Calculate true VWAP for a batch of intraday records
        
        This method calculates the proper volume-weighted average price
        using the formula: VWAP = Σ(Price × Volume) / Σ(Volume)
        
        Args:
            records: List of OHLCV records with price and volume data
            
        Returns:
            Calculated VWAP
        """
        if not records:
            return 0.0
        
        total_volume = 0
        total_price_volume = 0
        
        for record in records:
            # Use typical price for each record
            high = record.get('high', 0)
            low = record.get('low', 0) 
            close = record.get('close', 0)
            volume = record.get('volume', 0)
            
            if volume > 0 and all(price > 0 for price in [high, low, close]):
                typical_price = (high + low + close) / 3
                total_price_volume += typical_price * volume
                total_volume += volume
        
        if total_volume == 0:
            return 0.0
        
        return round(total_price_volume / total_volume, 4) 
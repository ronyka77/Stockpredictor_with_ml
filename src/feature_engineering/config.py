"""
Feature Engineering Configuration

This module provides centralized configuration for all feature engineering operations,
consolidating parameters that were previously scattered across multiple files.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import dotenv

dotenv.load_dotenv()

@dataclass
class DataQualityConfig:
    """Configuration for data quality and validation parameters"""
    
    # Data Requirements
    MIN_DATA_POINTS: int = int(os.getenv('FE_MIN_DATA_POINTS', '100'))
    MIN_DATA_POINTS_RSI: int = int(os.getenv('FE_MIN_DATA_POINTS_RSI', '100'))  # RSI needs 2x for stability
    
    # Data Quality Thresholds
    MAX_MISSING_PCT: float = float(os.getenv('FE_MAX_MISSING_PCT', '0.05'))  # 5% max missing data
    OUTLIER_THRESHOLD: float = float(os.getenv('FE_OUTLIER_THRESHOLD', '3.0'))  # Z-score threshold
    QUALITY_SCORE_THRESHOLD: float = float(os.getenv('FE_QUALITY_SCORE_THRESHOLD', '50.0'))
    
    # Data Continuity
    DATA_CONTINUITY_MAX_GAP_DAYS: int = int(os.getenv('FE_MAX_GAP_DAYS', '7'))
    
    # Database Precision Limits
    MAX_DB_VALUE: float = float(os.getenv('FE_MAX_DB_VALUE', '1e8'))  # 100 million max for DB compatibility
    
    # Validation Settings
    ENABLE_DATA_VALIDATION: bool = os.getenv('FE_ENABLE_VALIDATION', 'true').lower() == 'true'
    STRICT_VALIDATION: bool = os.getenv('FE_STRICT_VALIDATION', 'false').lower() == 'true'

@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing operations"""
    
    # Batch Processing
    DEFAULT_BATCH_SIZE: int = int(os.getenv('FE_BATCH_SIZE', '50'))
    MAX_WORKERS: int = int(os.getenv('FE_MAX_WORKERS', '4'))
    THREAD_POOL_SIZE: int = int(os.getenv('FE_THREAD_POOL_SIZE', '4'))
    
    # Processing Timeouts
    PROCESSING_TIMEOUT: int = int(os.getenv('FE_PROCESSING_TIMEOUT', '300'))  # seconds
    SINGLE_TICKER_TIMEOUT: int = int(os.getenv('FE_SINGLE_TICKER_TIMEOUT', '60'))  # seconds
    
    # Progress Reporting
    PROGRESS_REPORT_INTERVAL: int = int(os.getenv('FE_PROGRESS_INTERVAL', '10'))  # Log every N tickers
    
    # Error Handling
    MAX_RETRY_ATTEMPTS: int = int(os.getenv('FE_MAX_RETRIES', '3'))
    RETRY_DELAY: float = float(os.getenv('FE_RETRY_DELAY', '1.0'))  # seconds
    CONTINUE_ON_ERROR: bool = os.getenv('FE_CONTINUE_ON_ERROR', 'true').lower() == 'true'

    MIN_SUCCESS_RATE: float = float(os.getenv('FE_MIN_SUCCESS_RATE', '0.5'))

@dataclass
class TechnicalIndicatorConfig:
    """Configuration for technical indicator calculations"""
    
    # Moving Averages
    SMA_PERIODS: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    EMA_PERIODS: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    
    # MACD Parameters
    MACD_FAST: int = int(os.getenv('FE_MACD_FAST', '12'))
    MACD_SLOW: int = int(os.getenv('FE_MACD_SLOW', '26'))
    MACD_SIGNAL: int = int(os.getenv('FE_MACD_SIGNAL', '9'))
    
    @property
    def MACD_PARAMS(self) -> Dict[str, int]:
        return {"fast": self.MACD_FAST, "slow": self.MACD_SLOW, "signal": self.MACD_SIGNAL}
    
    # RSI Parameters
    RSI_PERIODS: List[int] = field(default_factory=lambda: [14, 21])
    RSI_OVERBOUGHT_THRESHOLD: float = float(os.getenv('FE_RSI_OVERBOUGHT', '70.0'))
    RSI_OVERSOLD_THRESHOLD: float = float(os.getenv('FE_RSI_OVERSOLD', '30.0'))
    
    # Bollinger Bands
    BOLLINGER_PERIOD: int = int(os.getenv('FE_BOLLINGER_PERIOD', '20'))
    BOLLINGER_STD: float = float(os.getenv('FE_BOLLINGER_STD', '2.0'))
    
    @property
    def BOLLINGER_PARAMS(self) -> Dict[str, float]:
        return {"period": self.BOLLINGER_PERIOD, "std": self.BOLLINGER_STD}
    
    # Stochastic Oscillator
    STOCHASTIC_K_PERIOD: int = int(os.getenv('FE_STOCH_K_PERIOD', '14'))
    STOCHASTIC_D_PERIOD: int = int(os.getenv('FE_STOCH_D_PERIOD', '3'))
    STOCHASTIC_OVERBOUGHT_THRESHOLD: float = float(os.getenv('FE_STOCH_OVERBOUGHT', '80.0'))
    STOCHASTIC_OVERSOLD_THRESHOLD: float = float(os.getenv('FE_STOCH_OVERSOLD', '20.0'))
    
    @property
    def STOCHASTIC_PARAMS(self) -> Dict[str, int]:
        return {"k_period": self.STOCHASTIC_K_PERIOD, "d_period": self.STOCHASTIC_D_PERIOD}
    
    # ATR (Average True Range)
    ATR_PERIOD: int = int(os.getenv('FE_ATR_PERIOD', '14'))
    
    # Rate of Change
    ROC_PERIODS: List[int] = field(default_factory=lambda: [10, 20, 30])
    
    # Williams %R
    WILLIAMS_R_PERIODS: List[int] = field(default_factory=lambda: [14, 21])
    
    # Ichimoku Cloud
    ICHIMOKU_TENKAN: int = int(os.getenv('FE_ICHIMOKU_TENKAN', '9'))
    ICHIMOKU_KIJUN: int = int(os.getenv('FE_ICHIMOKU_KIJUN', '26'))
    ICHIMOKU_SENKOU_B: int = int(os.getenv('FE_ICHIMOKU_SENKOU_B', '52'))
    ICHIMOKU_DISPLACEMENT: int = int(os.getenv('FE_ICHIMOKU_DISPLACEMENT', '26'))
    
    @property
    def ICHIMOKU_PARAMS(self) -> Dict[str, int]:
        return {
            "tenkan": self.ICHIMOKU_TENKAN,
            "kijun": self.ICHIMOKU_KIJUN,
            "senkou_b": self.ICHIMOKU_SENKOU_B,
            "displacement": self.ICHIMOKU_DISPLACEMENT
        }
    
    # Fibonacci
    FIBONACCI_LOOKBACK: int = int(os.getenv('FE_FIBONACCI_LOOKBACK', '100'))

@dataclass
class DateRangeConfig:
    """Configuration for date ranges and time parameters"""
    
    # Default Date Ranges
    DEFAULT_START_DATE: str = os.getenv('FE_DEFAULT_START_DATE', '2020-01-01')
    DEFAULT_LOOKBACK_YEARS: int = int(os.getenv('FE_DEFAULT_LOOKBACK_YEARS', '3'))
    
    # Date Formats
    DATE_FORMAT: str = os.getenv('FE_DATE_FORMAT', '%Y-%m-%d')
    DATETIME_FORMAT: str = os.getenv('FE_DATETIME_FORMAT', '%Y-%m-%d %H:%M:%S')
    
    # Time Zones
    DEFAULT_TIMEZONE: str = os.getenv('FE_DEFAULT_TIMEZONE', 'UTC')
    MARKET_TIMEZONE: str = os.getenv('FE_MARKET_TIMEZONE', 'America/New_York')

@dataclass
class FeatureCategoryConfig:
    """Configuration for feature categories and selection"""
    
    # Available Categories
    AVAILABLE_CATEGORIES: List[str] = field(default_factory=lambda: ['trend', 'momentum', 'volatility', 'volume'])
    DEFAULT_CATEGORIES: List[str] = field(default_factory=lambda: ['trend', 'momentum', 'volatility', 'volume'])
    
    # Market Filters
    AVAILABLE_MARKETS: List[str] = field(default_factory=lambda: ['stocks', 'crypto', 'forex', 'all'])
    DEFAULT_MARKET: str = os.getenv('FE_DEFAULT_MARKET', 'stocks')
    
    # Ticker Filters
    DEFAULT_ACTIVE_ONLY: bool = os.getenv('FE_DEFAULT_ACTIVE_ONLY', 'true').lower() == 'true'

@dataclass
class StorageConfig:
    """Configuration for feature storage systems"""
    
    # Base Storage Settings
    FEATURES_STORAGE_PATH: str = os.getenv('FE_STORAGE_PATH', 'data/features')
    FEATURE_VERSION: str = os.getenv('FE_FEATURE_VERSION', 'v1.0')
    
    # Parquet Settings
    PARQUET_COMPRESSION: str = os.getenv('FE_PARQUET_COMPRESSION', 'snappy')  # snappy, gzip, brotli
    PARQUET_ENGINE: str = os.getenv('FE_PARQUET_ENGINE', 'pyarrow')  # pyarrow, fastparquet
    PARQUET_ROW_GROUP_SIZE: int = int(os.getenv('FE_PARQUET_ROW_GROUP_SIZE', '50000'))
    
    # Storage Behavior
    SAVE_TO_DATABASE: bool = os.getenv('FE_SAVE_TO_DATABASE', 'true').lower() == 'true'
    SAVE_TO_PARQUET: bool = os.getenv('FE_SAVE_TO_PARQUET', 'true').lower() == 'true'
    USE_CONSOLIDATED_STORAGE: bool = os.getenv('FE_USE_CONSOLIDATED_STORAGE', 'true').lower() == 'true'
    OVERWRITE_EXISTING: bool = os.getenv('FE_OVERWRITE_EXISTING', 'false').lower() == 'true'
    
    # Partitioning
    PARTITIONING_STRATEGY: str = os.getenv('FE_PARTITIONING_STRATEGY', 'by_date')  # Only year-based supported
    
    # File Management
    CLEANUP_OLD_VERSIONS: bool = os.getenv('FE_CLEANUP_OLD_VERSIONS', 'true').lower() == 'true'
    MAX_VERSIONS_TO_KEEP: int = int(os.getenv('FE_MAX_VERSIONS_TO_KEEP', '3'))
    
    # Consolidated Storage
    MAX_ROWS_PER_FILE: int = int(os.getenv('FE_MAX_ROWS_PER_FILE', '5000000'))  # 5M rows
    INCLUDE_METADATA_COLUMNS: bool = os.getenv('FE_INCLUDE_METADATA_COLUMNS', 'true').lower() == 'true'

@dataclass
class MLConfig:
    """Configuration for ML-related operations"""
    
    # Data Splitting
    DEFAULT_TEST_SIZE: float = float(os.getenv('FE_ML_TEST_SIZE', '0.2'))
    DEFAULT_VALIDATION_SIZE: float = float(os.getenv('FE_ML_VALIDATION_SIZE', '0.1'))
    
    # Scaling Methods
    DEFAULT_SCALING_METHOD: str = os.getenv('FE_ML_SCALING_METHOD', 'standard')  # standard, minmax, robust, none
    AVAILABLE_SCALING_METHODS: List[str] = field(default_factory=lambda: ['standard', 'minmax', 'robust', 'none'])
    
    # Missing Value Handling
    DEFAULT_MISSING_STRATEGY: str = os.getenv('FE_ML_MISSING_STRATEGY', 'drop')  # drop, fill_mean, fill_median, fill_zero
    AVAILABLE_MISSING_STRATEGIES: List[str] = field(default_factory=lambda: ['drop', 'fill_mean', 'fill_median', 'fill_zero'])
    
    # Target Variable
    DEFAULT_TARGET_COLUMN: str = os.getenv('FE_ML_TARGET_COLUMN', 'close')
    DEFAULT_PREDICTION_HORIZON: int = int(os.getenv('FE_ML_PREDICTION_HORIZON', '1'))  # days ahead
    
    # Feature Selection
    ENABLE_FEATURE_SELECTION: bool = os.getenv('FE_ML_ENABLE_FEATURE_SELECTION', 'false').lower() == 'true'
    MAX_FEATURES: Optional[int] = int(os.getenv('FE_ML_MAX_FEATURES', '0')) or None

@dataclass
class MonitoringConfig:
    """Configuration for monitoring and logging"""
    
    # Monitoring Intervals
    RECENT_ACTIVITY_DAYS: int = int(os.getenv('FE_RECENT_ACTIVITY_DAYS', '7'))
    STATS_REFRESH_INTERVAL: int = int(os.getenv('FE_STATS_REFRESH_INTERVAL', '300'))  # seconds
    
    # History Limits
    MAX_JOB_HISTORY_RECORDS: int = int(os.getenv('FE_MAX_JOB_HISTORY', '20'))
    MAX_ERROR_LOG_ENTRIES: int = int(os.getenv('FE_MAX_ERROR_LOG_ENTRIES', '100'))
    
    # Performance Monitoring
    MEMORY_USAGE_THRESHOLD: float = float(os.getenv('FE_MEMORY_THRESHOLD', '0.8'))  # 80%
    PROCESSING_TIME_WARNING_THRESHOLD: int = int(os.getenv('FE_PROCESSING_TIME_WARNING', '300'))  # seconds
    
    # Alerting
    ENABLE_ALERTS: bool = os.getenv('FE_ENABLE_ALERTS', 'false').lower() == 'true'
    ALERT_EMAIL: Optional[str] = os.getenv('FE_ALERT_EMAIL')

@dataclass
class DatabaseConfig:
    """Configuration for database operations"""
    
    # Connection Settings
    DB_HOST: str = os.getenv('DB_HOST', 'localhost')
    DB_PORT: int = int(os.getenv('DB_PORT', '5432'))
    DB_NAME: str = os.getenv('DB_NAME', 'stock_data')
    DB_USER: str = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD: str = os.getenv('DB_PASSWORD', '')
    
    # Query Limits
    TICKER_QUERY_LIMIT: int = int(os.getenv('FE_TICKER_QUERY_LIMIT', '1000'))
    FEATURE_HISTORY_LIMIT: int = int(os.getenv('FE_FEATURE_HISTORY_LIMIT', '20'))
    BATCH_INSERT_SIZE: int = int(os.getenv('FE_BATCH_INSERT_SIZE', '1000'))
    
    # Connection Pool
    DB_POOL_SIZE: int = int(os.getenv('FE_DB_POOL_SIZE', '5'))
    DB_MAX_OVERFLOW: int = int(os.getenv('FE_DB_MAX_OVERFLOW', '10'))
    DB_POOL_TIMEOUT: int = int(os.getenv('FE_DB_POOL_TIMEOUT', '30'))
    
    @property
    def database_url(self) -> str:
        """Generate PostgreSQL connection URL"""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

@dataclass
class CommandLineConfig:
    """Default values for command line arguments"""
    
    # Batch Processing Defaults
    CLI_BATCH_SIZE: int = 50
    CLI_MAX_WORKERS: int = 4
    CLI_MIN_DATA_POINTS: int = 100
    
    # Filter Defaults
    CLI_DEFAULT_MARKET: str = 'stocks'
    CLI_DEFAULT_ACTIVE_ONLY: bool = True
    CLI_DEFAULT_CATEGORIES: List[str] = field(default_factory=lambda: ['trend', 'momentum', 'volatility', 'volume'])
    
    # Behavior Defaults
    CLI_DEFAULT_DRY_RUN: bool = False
    CLI_DEFAULT_VERBOSE: bool = False
    CLI_DEFAULT_OVERWRITE: bool = False

def _parse_tuple_float(env_var: str, default: Tuple[float, float]) -> Tuple[float, float]:
    """Parse comma-separated float tuple from environment variable"""
    value = os.getenv(env_var)
    if value:
        try:
            parts = value.split(',')
            if len(parts) == 2:
                return (float(parts[0]), float(parts[1]))
        except (ValueError, IndexError):
            pass
    return default

def _parse_tuple_int(env_var: str, default: Tuple[int, int]) -> Tuple[int, int]:
    """Parse comma-separated int tuple from environment variable"""
    value = os.getenv(env_var)
    if value:
        try:
            parts = value.split(',')
            if len(parts) == 2:
                return (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            pass
    return default

def _parse_list_int(env_var: str, default: List[int]) -> List[int]:
    """Parse comma-separated int list from environment variable"""
    value = os.getenv(env_var)
    if value:
        try:
            return [int(x.strip()) for x in value.split(',') if x.strip()]
        except ValueError:
            pass
    return default

@dataclass
class FundamentalConfig:
    """Configuration for fundamental analysis features"""
    
    # Data Collection
    POLYGON_FUNDAMENTALS_ENDPOINT: str = os.getenv('FE_FUNDAMENTAL_POLYGON_ENDPOINT', 'https://api.polygon.io/vX/reference/financials')
    UPDATE_FREQUENCY: str = os.getenv('FE_FUNDAMENTAL_UPDATE_FREQUENCY', 'daily')
    HISTORICAL_YEARS: int = int(os.getenv('FE_FUNDAMENTAL_HISTORICAL_YEARS', '2'))
    
    # Data Processing
    MISSING_DATA_STRATEGY: str = os.getenv('FE_FUNDAMENTAL_MISSING_DATA_STRATEGY', 'forward_fill')
    OUTLIER_CAPPING: bool = os.getenv('FE_FUNDAMENTAL_OUTLIER_CAPPING', 'true').lower() == 'true'
    SECTOR_CLASSIFICATION: str = os.getenv('FE_FUNDAMENTAL_SECTOR_CLASSIFICATION', 'GICS')
    
    # Calculation Limits
    MAX_RATIO_VALUE: float = float(os.getenv('FE_MAX_RATIO_VALUE', '1e6'))  # Maximum ratio value before flagging as extreme
    
    # Ratio Limits (for outlier capping)
    PE_RATIO_CAP: Tuple[float, float] = field(default_factory=lambda: _parse_tuple_float('FE_PE_RATIO_CAP', (-100, 100)))
    PB_RATIO_CAP: Tuple[float, float] = field(default_factory=lambda: _parse_tuple_float('FE_PB_RATIO_CAP', (0, 50)))
    DEBT_TO_EQUITY_CAP: Tuple[float, float] = field(default_factory=lambda: _parse_tuple_float('FE_DEBT_TO_EQUITY_CAP', (0, 10)))
    CURRENT_RATIO_CAP: Tuple[float, float] = field(default_factory=lambda: _parse_tuple_float('FE_CURRENT_RATIO_CAP', (0, 20)))
    ROE_CAP: Tuple[float, float] = field(default_factory=lambda: _parse_tuple_float('FE_ROE_CAP', (-1, 2)))
    ROA_CAP: Tuple[float, float] = field(default_factory=lambda: _parse_tuple_float('FE_ROA_CAP', (-1, 1)))
    NET_MARGIN_CAP: Tuple[float, float] = field(default_factory=lambda: _parse_tuple_float('FE_NET_MARGIN_CAP', (-2, 2)))
    GROSS_MARGIN_CAP: Tuple[float, float] = field(default_factory=lambda: _parse_tuple_float('FE_GROSS_MARGIN_CAP', (-1, 1)))
    OPERATING_MARGIN_CAP: Tuple[float, float] = field(default_factory=lambda: _parse_tuple_float('FE_OPERATING_MARGIN_CAP', (-2, 2)))
    ASSET_TURNOVER_CAP: Tuple[float, float] = field(default_factory=lambda: _parse_tuple_float('FE_ASSET_TURNOVER_CAP', (0, 10)))
    QUICK_RATIO_CAP: Tuple[float, float] = field(default_factory=lambda: _parse_tuple_float('FE_QUICK_RATIO_CAP', (0, 20)))
    CASH_RATIO_CAP: Tuple[float, float] = field(default_factory=lambda: _parse_tuple_float('FE_CASH_RATIO_CAP', (0, 10)))
    INTEREST_COVERAGE_CAP: Tuple[float, float] = field(default_factory=lambda: _parse_tuple_float('FE_INTEREST_COVERAGE_CAP', (-100, 1000)))
    FINANCIAL_LEVERAGE_CAP: Tuple[float, float] = field(default_factory=lambda: _parse_tuple_float('FE_FINANCIAL_LEVERAGE_CAP', (1, 50)))
    
    # SIC to GICS Sector Mapping Ranges
    SIC_ENERGY_RANGE: Tuple[int, int] = field(default_factory=lambda: _parse_tuple_int('FE_SIC_ENERGY_RANGE', (100, 999)))
    SIC_MATERIALS_RANGE: Tuple[int, int] = field(default_factory=lambda: _parse_tuple_int('FE_SIC_MATERIALS_RANGE', (1000, 1499)))
    SIC_INDUSTRIALS_RANGE_1: Tuple[int, int] = field(default_factory=lambda: _parse_tuple_int('FE_SIC_INDUSTRIALS_RANGE_1', (1500, 1799)))
    SIC_INDUSTRIALS_RANGE_2: Tuple[int, int] = field(default_factory=lambda: _parse_tuple_int('FE_SIC_INDUSTRIALS_RANGE_2', (2000, 3999)))
    SIC_UTILITIES_RANGE: Tuple[int, int] = field(default_factory=lambda: _parse_tuple_int('FE_SIC_UTILITIES_RANGE', (4000, 4999)))
    SIC_CONSUMER_STAPLES_RANGE: Tuple[int, int] = field(default_factory=lambda: _parse_tuple_int('FE_SIC_CONSUMER_STAPLES_RANGE', (5000, 5199)))
    SIC_CONSUMER_DISCRETIONARY_RANGE: Tuple[int, int] = field(default_factory=lambda: _parse_tuple_int('FE_SIC_CONSUMER_DISCRETIONARY_RANGE', (5200, 5999)))
    SIC_FINANCIALS_RANGE: Tuple[int, int] = field(default_factory=lambda: _parse_tuple_int('FE_SIC_FINANCIALS_RANGE', (6000, 6799)))
    SIC_TECHNOLOGY_RANGE: Tuple[int, int] = field(default_factory=lambda: _parse_tuple_int('FE_SIC_TECHNOLOGY_RANGE', (7000, 8999)))
    SIC_DEFAULT_SECTOR: str = os.getenv('FE_SIC_DEFAULT_SECTOR', 'Industrials')
    
    # Quality Thresholds
    MIN_FUNDAMENTAL_DATA_POINTS: int = int(os.getenv('FE_MIN_FUNDAMENTAL_DATA_POINTS', '8'))
    MAX_MISSING_FUNDAMENTAL_PCT: float = float(os.getenv('FE_MAX_MISSING_FUNDAMENTAL_PCT', '0.25'))
    
    # Scoring System Weights
    ALTMAN_Z_SCORE_WEIGHT: float = float(os.getenv('FE_ALTMAN_Z_SCORE_WEIGHT', '0.3'))
    PIOTROSKI_F_SCORE_WEIGHT: float = float(os.getenv('FE_PIOTROSKI_F_SCORE_WEIGHT', '0.3'))
    FINANCIAL_HEALTH_WEIGHT: float = float(os.getenv('FE_FINANCIAL_HEALTH_WEIGHT', '0.4'))
    
    # Sector Analysis
    MIN_SECTOR_COMPANIES: int = int(os.getenv('FE_MIN_SECTOR_COMPANIES', '5'))
    PERCENTILE_CALCULATION_METHOD: str = os.getenv('FE_PERCENTILE_CALCULATION_METHOD', 'linear')
    
    # Growth Calculation
    GROWTH_PERIODS: List[int] = field(default_factory=lambda: _parse_list_int('FE_GROWTH_PERIODS', [1, 3, 5]))
    MIN_PERIODS_FOR_GROWTH: int = int(os.getenv('FE_MIN_PERIODS_FOR_GROWTH', '2'))
    
    # Database Storage
    FUNDAMENTAL_BATCH_SIZE: int = int(os.getenv('FE_FUNDAMENTAL_BATCH_SIZE', '100'))
    FUNDAMENTAL_TIMEOUT: int = int(os.getenv('FE_FUNDAMENTAL_TIMEOUT', '60'))

@dataclass
class FeatureEngineeringConfig:
    """Main configuration class that combines all sub-configurations"""
    
    # Sub-configurations
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    batch_processing: BatchProcessingConfig = field(default_factory=BatchProcessingConfig)
    technical_indicators: TechnicalIndicatorConfig = field(default_factory=TechnicalIndicatorConfig)
    date_range: DateRangeConfig = field(default_factory=DateRangeConfig)
    feature_categories: FeatureCategoryConfig = field(default_factory=FeatureCategoryConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cli: CommandLineConfig = field(default_factory=CommandLineConfig)
    fundamental: FundamentalConfig = field(default_factory=FundamentalConfig)
    
    # Global Settings
    ENABLE_PARALLEL_PROCESSING: bool = os.getenv('FE_ENABLE_PARALLEL', 'true').lower() == 'true'
    DEBUG_MODE: bool = os.getenv('FE_DEBUG_MODE', 'false').lower() == 'true'
    ENVIRONMENT: str = os.getenv('FE_ENVIRONMENT', 'production')  
    
    def validate_config(self) -> List[str]:
        """
        Validate configuration settings and return list of issues
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate batch processing
        if self.batch_processing.DEFAULT_BATCH_SIZE < 1:
            errors.append("Batch size must be at least 1")
        
        if self.batch_processing.MAX_WORKERS < 1:
            errors.append("Max workers must be at least 1")
        
        # Validate data quality
        if not 0 <= self.data_quality.MAX_MISSING_PCT <= 1:
            errors.append("Max missing percentage must be between 0 and 1")
        
        if self.data_quality.MIN_DATA_POINTS < 50:
            errors.append("Minimum data points should be at least 50")
        
        # Validate technical indicators
        if self.technical_indicators.RSI_OVERBOUGHT_THRESHOLD <= self.technical_indicators.RSI_OVERSOLD_THRESHOLD:
            errors.append("RSI overbought threshold must be greater than oversold threshold")
        
        # Validate storage paths
        if not self.storage.FEATURES_STORAGE_PATH:
            errors.append("Features storage path cannot be empty")
        
        # Validate ML config
        if not 0 < self.ml.DEFAULT_TEST_SIZE < 1:
            errors.append("Test size must be between 0 and 1")
        
        if not 0 < self.ml.DEFAULT_VALIDATION_SIZE < 1:
            errors.append("Validation size must be between 0 and 1")
        
        if (self.ml.DEFAULT_TEST_SIZE + self.ml.DEFAULT_VALIDATION_SIZE) >= 1:
            errors.append("Test size + validation size must be less than 1")
        
        return errors
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current configuration
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            'environment': self.ENVIRONMENT,
            'debug_mode': self.DEBUG_MODE,
            'parallel_processing': self.ENABLE_PARALLEL_PROCESSING,
            'batch_size': self.batch_processing.DEFAULT_BATCH_SIZE,
            'max_workers': self.batch_processing.MAX_WORKERS,
            'min_data_points': self.data_quality.MIN_DATA_POINTS,
            'default_categories': self.feature_categories.DEFAULT_CATEGORIES,
            'storage_path': self.storage.FEATURES_STORAGE_PATH,
            'feature_version': self.storage.FEATURE_VERSION,
            'database_url': self.database.database_url.replace(self.database.DB_PASSWORD, '***') if self.database.DB_PASSWORD else self.database.database_url
        }

# Global configuration instance
config = FeatureEngineeringConfig()

# Backward compatibility - expose commonly used values at module level
MIN_DATA_POINTS = config.data_quality.MIN_DATA_POINTS
MAX_MISSING_PCT = config.data_quality.MAX_MISSING_PCT
OUTLIER_THRESHOLD = config.data_quality.OUTLIER_THRESHOLD
SMA_PERIODS = config.technical_indicators.SMA_PERIODS
EMA_PERIODS = config.technical_indicators.EMA_PERIODS
RSI_PERIODS = config.technical_indicators.RSI_PERIODS
MACD_PARAMS = config.technical_indicators.MACD_PARAMS
BOLLINGER_PARAMS = config.technical_indicators.BOLLINGER_PARAMS
STOCHASTIC_PARAMS = config.technical_indicators.STOCHASTIC_PARAMS
ATR_PERIOD = config.technical_indicators.ATR_PERIOD
ICHIMOKU_PARAMS = config.technical_indicators.ICHIMOKU_PARAMS
FIBONACCI_LOOKBACK = config.technical_indicators.FIBONACCI_LOOKBACK
FEATURES_STORAGE_PATH = config.storage.FEATURES_STORAGE_PATH
FEATURE_VERSION = config.storage.FEATURE_VERSION
PARQUET_COMPRESSION = config.storage.PARQUET_COMPRESSION
PARQUET_ENGINE = config.storage.PARQUET_ENGINE
PARQUET_ROW_GROUP_SIZE = config.storage.PARQUET_ROW_GROUP_SIZE
CLEANUP_OLD_VERSIONS = config.storage.CLEANUP_OLD_VERSIONS
MAX_VERSIONS_TO_KEEP = config.storage.MAX_VERSIONS_TO_KEEP
MAX_WORKERS = config.batch_processing.MAX_WORKERS
ENABLE_PARALLEL_PROCESSING = config.ENABLE_PARALLEL_PROCESSING

def validate_configuration() -> None:
    """
    Validate the current configuration and raise exception if invalid
    
    Raises:
        ValueError: If configuration validation fails
    """
    errors = config.validate_config()
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))

def print_config_summary() -> None:
    """Print a summary of the current configuration"""
    summary = config.get_config_summary()
    print("Feature Engineering Configuration Summary:")
    print("=" * 50)
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    # Validate configuration when run directly
    try:
        validate_configuration()
        print("✅ Configuration validation passed")
        print_config_summary()
    except ValueError as e:
        print(f"❌ Configuration validation failed: {e}")
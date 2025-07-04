"""
Configuration settings for Polygon.io data acquisition and news collection
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class PolygonConfig:
    """Configuration class for Polygon.io API settings"""
    
    # API Configuration
    API_KEY: str = os.getenv("POLYGON_API_KEY")
    BASE_URL: str = "https://api.polygon.io"
    
    # Rate Limiting (Free Tier)
    REQUESTS_PER_MINUTE: int = 5
    MAX_RETRIES: int = 3
    REQUEST_TIMEOUT: int = 30
    
    # Data Configuration
    DEFAULT_TIMESPAN: str = "day"
    MAX_RECORDS_PER_REQUEST: int = 50000
    ADJUSTED_DATA: bool = True
    
    # News Collection Configuration
    NEWS_MAX_TICKERS: int = int(os.getenv('NEWS_MAX_TICKERS', '100'))
    NEWS_DAYS_LOOKBACK: int = int(os.getenv('NEWS_DAYS_LOOKBACK', '7'))
    NEWS_RETENTION_YEARS: int = int(os.getenv('NEWS_RETENTION_YEARS', '2'))
    NEWS_BATCH_SIZE: int = int(os.getenv('NEWS_BATCH_SIZE', '100'))
    
    # Database Configuration
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "stock_data")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    
    @property
    def database_url(self) -> str:
        """Generate PostgreSQL connection URL"""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    @classmethod
    def from_env(cls) -> 'PolygonConfig':
        """Create configuration from environment variables"""
        return cls(
            API_KEY=os.getenv("POLYGON_API_KEY", cls.API_KEY),
            DB_HOST=os.getenv("DB_HOST", cls.DB_HOST),
            DB_PORT=int(os.getenv("DB_PORT", str(cls.DB_PORT))),
            DB_NAME=os.getenv("DB_NAME", cls.DB_NAME),
            DB_USER=os.getenv("DB_USER", cls.DB_USER),
            DB_PASSWORD=os.getenv("DB_PASSWORD", cls.DB_PASSWORD),
        )

@dataclass
class FeatureEngineeringConfig:
    """Configuration class for Feature Engineering Pipeline"""
    
    # Technical Indicator Parameters
    RSI_PERIODS: List[int] = field(default_factory=lambda: [14, 21])
    SMA_PERIODS: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    EMA_PERIODS: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    MACD_PARAMS: Dict = field(default_factory=lambda: {"fast": 12, "slow": 26, "signal": 9})
    BOLLINGER_PARAMS: Dict = field(default_factory=lambda: {"period": 20, "std": 2})
    ATR_PERIOD: int = 14
    STOCHASTIC_PARAMS: Dict = field(default_factory=lambda: {"k_period": 14, "d_period": 3})
    
    # Advanced Indicator Parameters
    ICHIMOKU_PARAMS: Dict = field(default_factory=lambda: {
        "tenkan": 9, "kijun": 26, "senkou_b": 52, "displacement": 26
    })
    FIBONACCI_LOOKBACK: int = 100
    
    # Multi-timeframe Settings
    TIMEFRAMES: List[str] = field(default_factory=lambda: ["1D", "1W", "1M"])
    
    # Storage Configuration
    FEATURES_STORAGE_PATH: str = os.getenv("FEATURES_STORAGE_PATH", "data/features")
    FEATURE_VERSION: str = os.getenv("FEATURE_VERSION", "v1.0")
    PARQUET_COMPRESSION: str = "snappy"  # Options: snappy, gzip, brotli
    PARQUET_ENGINE: str = "pyarrow"      # Options: pyarrow, fastparquet
    
    # Normalization Settings
    NORMALIZATION_METHOD: str = "z-score"  # Options: z-score, min-max, robust
    NORMALIZATION_WINDOW: int = 252  # Rolling window for normalization (trading days in a year)
    
    # Validation Settings
    MAX_MISSING_PCT: float = 0.05  # 5% max missing data allowed
    OUTLIER_THRESHOLD: float = 3.0  # Z-score threshold for outliers
    MIN_DATA_POINTS: int = int(os.getenv('MIN_DATA_POINTS', 200))     # Minimum data points required for calculation
    
    # File Management
    CLEANUP_OLD_VERSIONS: bool = True
    MAX_VERSIONS_TO_KEEP: int = 3
    
    # Performance Settings
    PARQUET_ROW_GROUP_SIZE: int = 50000
    ENABLE_PARALLEL_PROCESSING: bool = True
    MAX_WORKERS: int = 4
    
    # Database Configuration (inherits from PolygonConfig)
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "stock_data")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    
    @property
    def database_url(self) -> str:
        """Generate PostgreSQL connection URL"""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

# Global configuration instances
config = PolygonConfig.from_env()
feature_config = FeatureEngineeringConfig()

# Legacy logging config kept for backward compatibility only
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'detailed',
            'class': 'logging.FileHandler',
            'filename': 'src/logs/polygon/polygon_data_acquisition.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
} 
"""
Feature Storage System

This module provides storage capabilities for calculated technical indicators
using Parquet files for both feature data and metadata (optimized for ML workflows).
All data is stored in efficient Parquet format for fast loading and processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import dataclass, asdict

from src.utils.logger import get_logger
from src.feature_engineering.config import config

logger = get_logger(__name__, utility='feature_engineering')

@dataclass
class FeatureMetadata:
    """Metadata for stored features"""
    ticker: str
    feature_version: str
    calculation_date: datetime
    start_date: date
    end_date: date
    feature_categories: List[str]
    total_features: int
    quality_score: float
    file_path: str
    file_size_mb: float
    record_count: int
    warnings: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        data = asdict(self)
        # Convert dates to strings for JSON serialization
        data['calculation_date'] = self.calculation_date.isoformat()
        data['start_date'] = self.start_date.isoformat()
        data['end_date'] = self.end_date.isoformat()
        return data

@dataclass
class StorageConfig:
    """Configuration for feature storage"""
    base_path: str = config.storage.FEATURES_STORAGE_PATH
    version: str = config.storage.FEATURE_VERSION
    compression: str = config.storage.PARQUET_COMPRESSION
    engine: str = config.storage.PARQUET_ENGINE
    row_group_size: int = config.storage.PARQUET_ROW_GROUP_SIZE
    cleanup_old_versions: bool = config.storage.CLEANUP_OLD_VERSIONS
    max_versions: int = config.storage.MAX_VERSIONS_TO_KEEP

class FeatureStorage:
    """
    Feature storage system using Parquet files for both data and metadata
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        """
        Initialize feature storage
        
        Args:
            config: Storage configuration
        """
        self.config = config or StorageConfig()
        
        # Create storage directories
        self.base_path = Path(self.config.base_path)
        self.version_path = self.base_path / self.config.version
        self.version_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized FeatureStorage at {self.version_path}")
    
    def save_features(self, ticker: str, features_data: pd.DataFrame, 
                        metadata: Dict[str, Any]) -> FeatureMetadata:
        """
        Save features and metadata to Parquet files
        
        Args:
            ticker: Stock ticker symbol
            features_data: DataFrame with calculated features
            metadata: Additional metadata about the features
            
        Returns:
            FeatureMetadata object with storage information
        """
        try:
            # Generate file path
            file_path = self._generate_file_path(ticker)
            
            # Prepare data for storage
            storage_data = self._prepare_data_for_storage(features_data)
            
            # Save to Parquet
            self._save_parquet(storage_data, file_path)
            
            # Calculate file statistics
            file_stats = self._get_file_stats(file_path)
            
            # Create metadata
            feature_metadata = FeatureMetadata(
                ticker=ticker,
                feature_version=self.config.version,
                calculation_date=datetime.now(),
                start_date=features_data.index.min().date(),
                end_date=features_data.index.max().date(),
                feature_categories=metadata.get('categories', []),
                total_features=len(features_data.columns),
                quality_score=metadata.get('quality_score', 0.0),
                file_path=str(file_path.relative_to(self.base_path)),
                file_size_mb=file_stats['size_mb'],
                record_count=len(features_data),
                warnings=metadata.get('warnings', [])
            )
            
            # Save metadata to Parquet file alongside features
            self._save_metadata_to_parquet(feature_metadata)
            
            # Cleanup old versions if configured
            if self.config.cleanup_old_versions:
                self._cleanup_old_versions(ticker)
            
            logger.info(f"Saved {len(features_data)} records with {len(features_data.columns)} features for {ticker}")
            logger.info(f"File: {file_path}, Size: {file_stats['size_mb']:.2f} MB")
            
            return feature_metadata
            
        except Exception as e:
            logger.error(f"Error saving features for {ticker}: {str(e)}")
            raise
    
    def load_features(self, ticker: str, start_date: Optional[date] = None, 
                        end_date: Optional[date] = None, 
                        categories: Optional[List[str]] = None,
                        version: Optional[str] = None) -> Tuple[pd.DataFrame, FeatureMetadata]:
        """
        Load features from Parquet file
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date filter
            end_date: End date filter
            categories: Feature categories to load
            version: Specific version to load (default: latest)
            
        Returns:
            Tuple of (features DataFrame, metadata)
        """
        try:
            # Get file path
            file_path = self._get_latest_file_path(ticker, version)
            
            if not file_path.exists():
                raise FileNotFoundError(f"No features found for {ticker}")
            
            # Load from Parquet
            features_data = self._load_parquet(file_path)
            
            # Apply filters
            if start_date:
                features_data = features_data[features_data.index >= pd.Timestamp(start_date)]
            if end_date:
                features_data = features_data[features_data.index <= pd.Timestamp(end_date)]
            
            if categories:
                # Filter columns by category
                category_columns = self._get_columns_by_category(features_data.columns, categories)
                features_data = features_data[category_columns]
            
            # Load metadata
            metadata = self._load_metadata_from_parquet(ticker, version)
            
            logger.info(f"Loaded {len(features_data)} records with {len(features_data.columns)} features for {ticker}")
            
            return features_data, metadata
            
        except Exception as e:
            logger.error(f"Error loading features for {ticker}: {str(e)}")
            raise
    
    def get_available_tickers(self, version: Optional[str] = None) -> List[str]:
        """
        Get list of tickers with stored features
        
        Args:
            version: Specific version to check (default: current)
            
        Returns:
            List of ticker symbols
        """
        version_path = self.version_path if not version else self.base_path / version
        
        if not version_path.exists():
            return []
        
        tickers = []
        for file_path in version_path.glob("*.parquet"):
            ticker = file_path.stem
            tickers.append(ticker)
        
        return sorted(tickers)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics
        
        Returns:
            Dictionary with storage statistics
        """
        stats = {
            'version': self.config.version,
            'base_path': str(self.base_path),
            'total_tickers': 0,
            'total_files': 0,
            'total_size_mb': 0.0,
            'versions': []
        }
        
        # Get stats for all versions
        for version_dir in self.base_path.iterdir():
            if version_dir.is_dir():
                version_stats = self._get_version_stats(version_dir)
                stats['versions'].append(version_stats)
                
                if version_dir.name == self.config.version:
                    stats['total_tickers'] = version_stats['ticker_count']
                    stats['total_files'] = version_stats['file_count']
                    stats['total_size_mb'] = version_stats['total_size_mb']
        
        return stats
    
    def _generate_file_path(self, ticker: str) -> Path:
        """Generate file path for ticker features"""
        filename = f"{ticker}.parquet"
        return self.version_path / filename
    
    def _get_latest_file_path(self, ticker: str, version: Optional[str] = None) -> Path:
        """Get the latest file path for a ticker"""
        version_path = self.version_path if not version else self.base_path / version
        return version_path / f"{ticker}.parquet"
    
    def _prepare_data_for_storage(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Parquet storage"""
        # Reset index to make date a column
        storage_data = data.reset_index()
        
        # Ensure date column is properly typed
        if 'date' in storage_data.columns:
            storage_data['date'] = pd.to_datetime(storage_data['date'])
        
        # Convert any remaining numpy types to native Python types
        for col in storage_data.columns:
            if storage_data[col].dtype == 'object':
                storage_data[col] = storage_data[col].astype('str')
            elif np.issubdtype(storage_data[col].dtype, np.integer):
                storage_data[col] = storage_data[col].astype('int64')
            elif np.issubdtype(storage_data[col].dtype, np.floating):
                storage_data[col] = storage_data[col].astype('float64')
        
        return storage_data
    
    def _save_parquet(self, data: pd.DataFrame, file_path: Path):
        """Save DataFrame to Parquet file"""
        table = pa.Table.from_pandas(data)
        
        pq.write_table(
            table,
            file_path,
            compression=self.config.compression,
            row_group_size=self.config.row_group_size,
            use_dictionary=True,
            write_statistics=True
        )
    
    def _load_parquet(self, file_path: Path) -> pd.DataFrame:
        """Load DataFrame from Parquet file"""
        data = pd.read_parquet(file_path, engine=self.config.engine)
        
        # Set date as index if it exists
        if 'date' in data.columns:
            data = data.set_index('date')
        
        return data
    
    def _get_file_stats(self, file_path: Path) -> Dict[str, Any]:
        """Get file statistics"""
        stat = file_path.stat()
        return {
            'size_mb': stat.st_size / (1024 * 1024),
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime)
        }
    
    def _get_version_stats(self, version_dir: Path) -> Dict[str, Any]:
        """Get statistics for a version directory"""
        files = list(version_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            'version': version_dir.name,
            'ticker_count': len(files),
            'file_count': len(files),
            'total_size_mb': total_size / (1024 * 1024),
            'path': str(version_dir)
        }
    
    def _get_columns_by_category(self, columns: List[str], categories: List[str]) -> List[str]:
        """Filter columns by feature categories"""
        category_columns = []
        
        for col in columns:
            col_lower = col.lower()
            for category in categories:
                if category == 'trend' and any(x in col_lower for x in ['sma', 'ema', 'macd', 'ichimoku']):
                    category_columns.append(col)
                    break
                elif category == 'momentum' and any(x in col_lower for x in ['rsi', 'stoch', 'roc', 'williams']):
                    category_columns.append(col)
                    break
                elif category == 'volatility' and any(x in col_lower for x in ['bb', 'bollinger', 'atr', 'volatility']):
                    category_columns.append(col)
                    break
                elif category == 'volume' and any(x in col_lower for x in ['obv', 'vpt', 'ad_line', 'volume', 'mfi']):
                    category_columns.append(col)
                    break
        
        return category_columns
    
    def _save_metadata_to_parquet(self, metadata: FeatureMetadata):
        """Save metadata to Parquet file"""
        try:
            metadata_path = self._get_metadata_file_path(metadata.ticker)
            metadata_df = pd.DataFrame([metadata.to_dict()])
            
            # Save metadata as Parquet
            metadata_df.to_parquet(
                metadata_path,
                compression=self.config.compression,
                engine=self.config.engine
            )
            
            logger.info(f"Saved metadata for {metadata.ticker} to {metadata_path}")
            
        except Exception as e:
            logger.warning(f"Could not save metadata to Parquet: {str(e)}")
    
    def _load_metadata_from_parquet(self, ticker: str, version: Optional[str] = None) -> Optional[FeatureMetadata]:
        """Load metadata from Parquet file"""
        try:
            metadata_path = self._get_metadata_file_path(ticker, version)
            
            if not metadata_path.exists():
                logger.info(f"No metadata file found for {ticker}")
                return None
            
            metadata_df = pd.read_parquet(metadata_path, engine=self.config.engine)
            
            if metadata_df.empty:
                return None
            
            # Get the first (and should be only) row
            row = metadata_df.iloc[0]
            
            return FeatureMetadata(
                ticker=row['ticker'],
                feature_version=row['feature_version'],
                calculation_date=pd.to_datetime(row['calculation_date']),
                start_date=pd.to_datetime(row['start_date']).date(),
                end_date=pd.to_datetime(row['end_date']).date(),
                feature_categories=row['feature_categories'],
                total_features=int(row['total_features']),
                quality_score=float(row['quality_score']),
                file_path=row['file_path'],
                file_size_mb=float(row['file_size_mb']),
                record_count=int(row['record_count']),
                warnings=row.get('warnings', [])
            )
                
        except Exception as e:
            logger.warning(f"Could not load metadata from Parquet: {str(e)}")
            return None
    
    def _get_metadata_file_path(self, ticker: str, version: Optional[str] = None) -> Path:
        """Get metadata file path for ticker"""
        version_path = self.version_path if not version else self.base_path / version
        filename = f"{ticker}_metadata.parquet"
        return version_path / filename
    
    def _cleanup_old_versions(self, ticker: str):
        """Cleanup old versions of ticker features"""
        try:
            # This is a simple implementation - in practice you might want
            # to keep multiple timestamped versions
            pass
        except Exception as e:
            logger.warning(f"Could not cleanup old versions for {ticker}: {str(e)}")
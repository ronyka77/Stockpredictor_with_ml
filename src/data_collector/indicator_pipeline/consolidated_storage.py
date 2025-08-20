"""
Consolidated Feature Storage System - Year-Based Partitioning Only

This module provides consolidated storage for multiple tickers in year-partitioned 
Parquet files for optimal ML performance and time-series analysis.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import date
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import dataclass

from src.data_collector.indicator_pipeline.feature_storage import FeatureStorage, StorageConfig
from src.data_collector.config import config as fe_config
from src.utils.logger import get_logger

logger = get_logger(__name__, utility='feature_engineering')

@dataclass
class ConsolidatedStorageConfig(StorageConfig):
    """Configuration for consolidated storage - Year-based partitioning only"""
    partitioning_strategy: str = fe_config.storage.PARTITIONING_STRATEGY
    max_rows_per_file: int = fe_config.storage.MAX_ROWS_PER_FILE
    include_metadata_columns: bool = fe_config.storage.INCLUDE_METADATA_COLUMNS

class ConsolidatedFeatureStorage:
    """
    Consolidated storage system for multiple tickers in year-partitioned Parquet files
    """
    
    def __init__(self, config: Optional[ConsolidatedStorageConfig] = None, db_engine=None):
        """
        Initialize consolidated feature storage
        
        Args:
            config: Consolidated storage configuration
            db_engine: Database engine for metadata storage
        """
        self.config = config or ConsolidatedStorageConfig()
        self.db_engine = db_engine
        
        # Create storage directories
        self.base_path = Path(self.config.base_path)
        self.version_path = self.base_path / self.config.version
        self.consolidated_path = self.version_path / "consolidated"
        self.consolidated_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ConsolidatedFeatureStorage at {self.consolidated_path}")
        logger.info(f"Partitioning strategy: {self.config.partitioning_strategy}")
    
    def save_multiple_tickers(self, ticker_data: Dict[str, pd.DataFrame], 
                            metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save features for multiple tickers in year-partitioned format
        
        Args:
            ticker_data: Dictionary mapping ticker -> features DataFrame
            metadata: Metadata about the features
            
        Returns:
            Dictionary with storage information
        """
        logger.info(f"Saving {len(ticker_data)} tickers in consolidated format")
        
        # Combine all ticker data
        combined_data = self._combine_ticker_data(ticker_data)
        
        # Apply year-based partitioning strategy
        if self.config.partitioning_strategy == "by_date":
            files_created = self._save_by_year(combined_data, metadata)
        else:
            raise ValueError(f"Only 'by_date' (year-based) partitioning is supported. Got: {self.config.partitioning_strategy}")
        
        # Calculate storage statistics
        total_size = sum(f['size_mb'] for f in files_created)
        total_rows = sum(f['rows'] for f in files_created)
        
        storage_info = {
            'strategy': self.config.partitioning_strategy,
            'files_created': len(files_created),
            'total_size_mb': total_size,
            'total_rows': total_rows,
            'total_tickers': len(ticker_data),
            'files': files_created,
            'compression_ratio': self._calculate_compression_ratio(combined_data, total_size)
        }
        
        logger.info("Consolidated storage completed:")
        logger.info(f"  Files created: {len(files_created)}")
        logger.info(f"  Total size: {total_size:.2f} MB")
        logger.info(f"  Total rows: {total_rows:,}")
        logger.info(f"  Compression ratio: {storage_info['compression_ratio']:.1f}x")
        
        return storage_info
    
    def load_consolidated_features(self, ticker: Optional[str] = None,
                                    start_date: Optional[date] = None,
                                    end_date: Optional[date] = None,
                                    categories: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load features from consolidated storage with filtering
        
        Args:
            ticker: Single ticker to load (None for ALL tickers)
            start_date: Start date filter
            end_date: End date filter
            categories: Feature categories to load
            
        Returns:
            Combined DataFrame with features
        """
        if ticker is None:
            logger.info("Loading features for ALL tickers from consolidated storage")
        else:
            logger.info(f"Loading features for ticker '{ticker}' from consolidated storage")
        
        # Find relevant files
        parquet_files = list(self.consolidated_path.glob("*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError("No consolidated feature files found")
        
        # Build filters - only single ticker or None
        filters = self._build_parquet_filters(ticker, start_date, end_date)
        
        all_data = []
        for file_path in parquet_files:
            try:
                # Load with filters for efficiency
                if filters:
                    data = pd.read_parquet(file_path, filters=filters)
                else:
                    data = pd.read_parquet(file_path)
                
                if not data.empty:
                    all_data.append(data)
                    
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {str(e)}")
        
        if not all_data:
            logger.warning("No data found matching the filters")
            return pd.DataFrame()
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        # Apply ticker filter if specified (single ticker only)
        if ticker:
            combined_data = combined_data[combined_data['ticker'] == ticker]
            if combined_data.empty:
                logger.warning(f"No data found for ticker '{ticker}'")
                return pd.DataFrame()
        
        # Apply category filter
        if categories:
            category_columns = self._get_columns_by_category(combined_data.columns, categories)
            # Keep metadata columns + category columns
            keep_columns = ['ticker', 'date'] + category_columns
            combined_data = combined_data[keep_columns]
        
        if ticker is None:
            logger.info(f"Loaded {len(combined_data)} records for ALL tickers with {len(combined_data.columns)} columns")
        else:
            logger.info(f"Loaded {len(combined_data)} records for ticker '{ticker}' with {len(combined_data.columns)} columns")
        
        return combined_data
    
    def get_consolidated_stats(self) -> Dict[str, Any]:
        """Get statistics about consolidated storage"""
        parquet_files = list(self.consolidated_path.glob("*.parquet"))
        
        if not parquet_files:
            return {'files': 0, 'total_size_mb': 0, 'total_rows': 0}
        
        total_size = 0
        total_rows = 0
        file_stats = []
        
        for file_path in parquet_files:
            try:
                # Get file size
                size_mb = file_path.stat().st_size / (1024 * 1024)
                total_size += size_mb
                
                # Get row count (read metadata only)
                parquet_file = pq.ParquetFile(file_path)
                rows = parquet_file.metadata.num_rows
                total_rows += rows
                
                file_stats.append({
                    'file': file_path.name,
                    'size_mb': size_mb,
                    'rows': rows,
                    'columns': parquet_file.metadata.num_columns
                })
                
            except Exception as e:
                logger.warning(f"Error reading stats for {file_path}: {str(e)}")
        
        return {
            'files': len(parquet_files),
            'total_size_mb': total_size,
            'total_rows': total_rows,
            'file_stats': file_stats,
            'storage_path': str(self.consolidated_path)
        }
    
    def _combine_ticker_data(self, ticker_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple ticker DataFrames into single DataFrame"""
        combined_data = []
        
        for ticker, features in ticker_data.items():
            # Add ticker column
            features_copy = features.copy()
            features_copy['ticker'] = ticker
            
            # Reset index to make date a column
            if features_copy.index.name == 'date' or isinstance(features_copy.index, pd.DatetimeIndex):
                features_copy = features_copy.reset_index()
            
            # Ensure date column exists and is datetime
            if 'date' in features_copy.columns:
                features_copy['date'] = pd.to_datetime(features_copy['date'])
            
            combined_data.append(features_copy)
        
        if not combined_data:
            return pd.DataFrame()
        
        # Combine all data
        result = pd.concat(combined_data, ignore_index=True)
        
        # Sort by ticker and date for better compression
        result = result.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        logger.info(f"Combined data: {len(result)} rows, {len(result.columns)} columns")
        
        return result

    def _save_by_year(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Save data partitioned by date (year)"""
        files_created = []
        
        # Group by year
        data['year'] = data['date'].dt.year
        
        for year, group_data in data.groupby('year'):
            # Remove the temporary column
            group_data = group_data.drop('year', axis=1)
            
            file_path = self.consolidated_path / f"features_{year}.parquet"
            
            # Save to Parquet
            table = pa.Table.from_pandas(group_data)
            pq.write_table(
                table,
                file_path,
                compression=self.config.compression,
                row_group_size=self.config.row_group_size,
                use_dictionary=True,
                write_statistics=True
            )
            
            # Get file stats
            size_mb = file_path.stat().st_size / (1024 * 1024)
            
            files_created.append({
                'file': file_path.name,
                'path': str(file_path),
                'rows': len(group_data),
                'columns': len(group_data.columns),
                'size_mb': size_mb,
                'year': str(year)
            })
        
        return files_created
    
    def _build_parquet_filters(self, ticker: Optional[str], 
                                start_date: Optional[date], 
                                end_date: Optional[date]) -> Optional[List]:
        """Build Parquet filters for efficient loading - single ticker or all"""
        filters = []
        
        if ticker:
            filters.append(('ticker', '==', ticker))
        
        if start_date:
            filters.append(('date', '>=', pd.Timestamp(start_date)))
        
        if end_date:
            filters.append(('date', '<=', pd.Timestamp(end_date)))
        
        return filters if filters else None
    
    def _get_columns_by_category(self, columns: List[str], categories: List[str]) -> List[str]:
        """Filter columns by feature categories"""
        category_columns = []
        
        for col in columns:
            if col in ['ticker', 'date']:  # Skip metadata columns
                continue
                
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
    
    def _calculate_compression_ratio(self, data: pd.DataFrame, compressed_size_mb: float) -> float:
        """Calculate compression ratio compared to uncompressed CSV"""
        # Estimate uncompressed CSV size (rough approximation)
        avg_chars_per_value = 8  # Average characters per numeric value
        estimated_csv_size_mb = (len(data) * len(data.columns) * avg_chars_per_value) / (1024 * 1024)
        
        if compressed_size_mb > 0:
            return estimated_csv_size_mb / compressed_size_mb
        return 1.0

# Convenience function
def consolidate_existing_features(strategy: str = "by_date") -> Dict[str, Any]:
    """
    Consolidate existing individual Parquet files into year-partitioned format
    
    Args:
        strategy: Consolidation strategy (only "by_date" supported)
        
    Returns:
        Consolidation results
    """
    if strategy != "by_date":
        logger.warning(f"Only 'by_date' strategy supported, using 'by_date' instead of '{strategy}'")
        strategy = "by_date"
    
    logger.info("Consolidating existing feature files")
    
    # Load existing individual files
    storage = FeatureStorage()
    available_tickers = storage.get_available_tickers()
    
    if not available_tickers:
        raise ValueError("No existing feature files found to consolidate")
    
    logger.info(f"Found {len(available_tickers)} existing ticker files")
    
    # Load all ticker data
    ticker_data = {}
    for ticker in available_tickers:
        try:
            features, metadata = storage.load_features(ticker)
            
            # Merge metadata to all rows of features
            if metadata is not None:
                features['ticker'] = ticker
                features['feature_version'] = metadata.feature_version
                features['calculation_date'] = metadata.calculation_date
                features['quality_score'] = metadata.quality_score
                features['total_features'] = metadata.total_features
                features['file_size_mb'] = metadata.file_size_mb
                features['record_count'] = metadata.record_count
                if hasattr(metadata, 'feature_categories') and metadata.feature_categories is not None:
                    features['feature_categories'] = str(metadata.feature_categories)
                if hasattr(metadata, 'warnings') and metadata.warnings is not None:
                    features['warnings'] = str(metadata.warnings)
            
            ticker_data[ticker] = features
        except Exception as e:
            logger.warning(f"Could not load {ticker}: {str(e)}")
    
    # Create consolidated storage
    config = ConsolidatedStorageConfig(partitioning_strategy=strategy)
    consolidated_storage = ConsolidatedFeatureStorage(config)
    
    # Save in consolidated format
    result = consolidated_storage.save_multiple_tickers(
        ticker_data, 
        {'source': 'consolidation', 'original_files': len(available_tickers)}
    )
    
    logger.info(f"Consolidation completed: {result['files_created']} files, {result['total_size_mb']:.2f} MB")
    
    return result 


def main():
    """Main function to run feature consolidation by year"""
    try:
        logger.info("🚀 Starting feature consolidation by year...")
        
        # Run consolidation with year-based strategy
        result = consolidate_existing_features(strategy='by_year')
        
        # log results
        logger.info("\n🎉 Feature Consolidation Completed!")
        logger.info(f"   Files created: {result['files_created']}")
        logger.info(f"   Total size: {result['total_size_mb']:.2f} MB")
        logger.info(f"   Compression ratio: {result['compression_ratio']:.1f}x")
        
        # Show file breakdown
        if 'files' in result:
            logger.info("\n📁 Consolidated Files:")
            for file_info in result['files']:
                logger.info(f"   {file_info['file']}: {file_info['rows']:,} rows, Year: {file_info['year']}")
        
        logger.info("✅ Consolidation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Consolidation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

"""
Cleaned Data Cache Management

This module provides caching functionality for cleaned ML data to avoid
recomputing expensive cleaning operations.
"""

import pandas as pd
from typing import List, Dict, Optional
import hashlib
from pathlib import Path
import json
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__, utility='cleaned_data_cache')

class CleanedDataCache:
    """
    Cache manager for cleaned ML data to avoid recomputing expensive cleaning operations
    """
    
    def __init__(self, cache_dir: str = "data/cleaned_cache"):
        """
        Initialize cleaned data cache
        
        Args:
            cache_dir: Directory to store cached cleaned data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # logger.info(f"CleanedDataCache initialized with cache directory: {self.cache_dir}")
    
    def _generate_cache_key(self, **kwargs) -> str:
        """
        Generate a unique cache key based on function parameters
        
        Args:
            **kwargs: Parameters to hash for cache key
            
        Returns:
            Unique cache key string
        """
        # Create a string representation of all parameters
        param_str = str(sorted(kwargs.items()))
        # Generate hash
        cache_key = hashlib.md5(param_str.encode()).hexdigest()
        return cache_key
    
    def _get_cache_paths(self, cache_key: str, data_type: str = "training") -> Dict[str, Path]:
        """
        Get file paths for cached data components
        
        Args:
            cache_key: Unique cache identifier
            data_type: Type of data ('training' or 'prediction')
            
        Returns:
            Dictionary of file paths for different data components
        """
        base_path = self.cache_dir / f"{data_type}_{cache_key}"
        
        paths = {
            'X_train': base_path.with_suffix('.X_train.parquet'),
            'X_test': base_path.with_suffix('.X_test.parquet'),
            'y_train': base_path.with_suffix('.y_train.parquet'),
            'y_test': base_path.with_suffix('.y_test.parquet'),
            'metadata': base_path.with_suffix('.metadata.parquet'),
            'info': base_path.with_suffix('.info.json')
        }
        
        # For prediction data, we only need X_test and y_test
        if data_type == "prediction":
            paths = {
                'X_test': base_path.with_suffix('.X_test.parquet'),
                'y_test': base_path.with_suffix('.y_test.parquet'),
                'metadata': base_path.with_suffix('.metadata.parquet'),
                'info': base_path.with_suffix('.info.json')
            }
        
        return paths
    
    def cache_exists(self, cache_key: str, data_type: str = "training") -> bool:
        """
        Check if cached data exists for given parameters
        
        Args:
            cache_key: Unique cache identifier
            data_type: Type of data ('training' or 'prediction')
            
        Returns:
            True if cache exists and is complete
        """
        paths = self._get_cache_paths(cache_key, data_type)
        
        # Check if all required files exist
        required_files = ['X_test', 'y_test', 'metadata', 'info']
        if data_type == "training":
            required_files.extend(['X_train', 'y_train'])
        
        return all(paths[file].exists() for file in required_files)
    
    def save_cleaned_data(self, data_result: Dict, cache_key: str, data_type: str = "training") -> None:
        """
        Save cleaned data to cache
        
        Args:
            data_result: Dictionary containing cleaned data
            cache_key: Unique cache identifier
            data_type: Type of data ('training' or 'prediction')
        """
        paths = self._get_cache_paths(cache_key, data_type)
        
        try:
            # Save DataFrames to parquet
            if data_type == "training":
                data_result['X_train'].to_parquet(paths['X_train'])
                data_result['y_train'].to_frame().to_parquet(paths['y_train'])
                logger.info(f"   Saved training data: {len(data_result['X_train'])} samples")
            
            data_result['X_test'].to_parquet(paths['X_test'])
            data_result['y_test'].to_frame().to_parquet(paths['y_test'])
            
            # Save metadata
            metadata = {
                'target_column': data_result.get('target_column', ''),
                'feature_count': data_result.get('feature_count', 0),
                'prediction_horizon': data_result.get('prediction_horizon', 10),
                'train_samples': data_result.get('train_samples', 0),
                'test_samples': data_result.get('test_samples', 0),
                'train_date_range': data_result.get('train_date_range', ''),
                'test_date_range': data_result.get('test_date_range', ''),
                'split_date': data_result.get('split_date', ''),
                'removed_features': data_result.get('removed_features', {}),
                'diversity_analysis': data_result.get('diversity_analysis', {})
            }
            
            # Fix Parquet serialization issues with empty structures
            # Convert empty dictionaries to strings to avoid Parquet struct type issues
            if isinstance(metadata['removed_features'], dict):
                if not metadata['removed_features'] or all(not v for v in metadata['removed_features'].values()):
                    metadata['removed_features'] = '{}'
                else:
                    # Convert to JSON string to preserve structure
                    metadata['removed_features'] = json.dumps(metadata['removed_features'])
            
            if isinstance(metadata['diversity_analysis'], dict):
                if not metadata['diversity_analysis']:
                    metadata['diversity_analysis'] = '{}'
                else:
                    # Convert to JSON string to preserve structure
                    metadata['diversity_analysis'] = json.dumps(metadata['diversity_analysis'])
            
            metadata_df = pd.DataFrame([metadata])
            metadata_df.to_parquet(paths['metadata'])
            
            # Save info file with cache details
            info = {
                'cache_key': cache_key,
                'data_type': data_type,
                'created_at': datetime.now().isoformat(),
                'file_paths': {k: str(v) for k, v in paths.items()}
            }
            
            with open(paths['info'], 'w') as f:
                json.dump(info, f, indent=2)
            
            logger.info(f"✅ Cached cleaned {data_type} data with key: {cache_key}")
            
        except Exception as e:
            logger.error(f"❌ Error saving cleaned data to cache: {str(e)}")
            # Clean up partial files with better error handling
            for path in paths.values():
                if path.exists():
                    try:
                        path.unlink()
                        logger.info(f"   Cleaned up partial file: {path}")
                    except PermissionError as pe:
                        logger.warning(f"   Could not delete {path}: {pe}")
                    except Exception as cleanup_error:
                        logger.warning(f"   Error cleaning up {path}: {cleanup_error}")
            raise
    
    def load_cleaned_data(self, cache_key: str, data_type: str = "training") -> Dict:
        """
        Load cleaned data from cache
        
        Args:
            cache_key: Unique cache identifier
            data_type: Type of data ('training' or 'prediction')
            
        Returns:
            Dictionary containing loaded cleaned data
        """
        if not self.cache_exists(cache_key, data_type):
            raise FileNotFoundError(f"Cache not found for key: {cache_key}, type: {data_type}")
        
        paths = self._get_cache_paths(cache_key, data_type)
        
        try:
            # Load DataFrames
            result = {}
            
            if data_type == "training":
                result['X_train'] = pd.read_parquet(paths['X_train'])
                result['y_train'] = pd.read_parquet(paths['y_train']).squeeze()
                logger.info(f"   Loaded training data: {len(result['X_train'])} samples")
            
            result['X_test'] = pd.read_parquet(paths['X_test'])
            result['y_test'] = pd.read_parquet(paths['y_test']).squeeze()
            
            # Load metadata
            metadata_df = pd.read_parquet(paths['metadata'])
            metadata = metadata_df.iloc[0].to_dict()
            
            # Convert JSON strings back to dictionaries for removed_features and diversity_analysis
            if 'removed_features' in metadata:
                if isinstance(metadata['removed_features'], str):
                    try:
                        metadata['removed_features'] = json.loads(metadata['removed_features'])
                    except json.JSONDecodeError:
                        # If it's just '{}', convert to empty dict
                        metadata['removed_features'] = {}
            
            if 'diversity_analysis' in metadata:
                if isinstance(metadata['diversity_analysis'], str):
                    try:
                        metadata['diversity_analysis'] = json.loads(metadata['diversity_analysis'])
                    except json.JSONDecodeError:
                        # If it's just '{}', convert to empty dict
                        metadata['diversity_analysis'] = {}
            
            # Add metadata to result
            result.update(metadata)
            
            logger.info(f"✅ Loaded cached cleaned {data_type} data with key: {cache_key}")
            logger.info(f"   Test data: {len(result['X_test'])} samples, {len(result['X_test'].columns)} features")
            # logger.info(f"   Test data features: {result['X_test'].columns.to_list()}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error loading cleaned data from cache: {str(e)}")
            raise
    
    def clear_cache(self, cache_key: Optional[str] = None, data_type: Optional[str] = None) -> None:
        """
        Clear cached data
        
        Args:
            cache_key: Specific cache key to clear (None for all)
            data_type: Specific data type to clear (None for all)
        """
        if cache_key and data_type:
            # Clear specific cache
            paths = self._get_cache_paths(cache_key, data_type)
            for path in paths.values():
                if path.exists():
                    path.unlink()
            logger.info(f"Cleared cache for key: {cache_key}, type: {data_type}")
        else:
            # Clear all cache files
            for file in self.cache_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            logger.info("Cleared all cached data")
    
    def list_cached_data(self) -> List[Dict]:
        """
        List all cached data entries
        
        Returns:
            List of cache information dictionaries
        """
        cached_entries = []
        
        for info_file in self.cache_dir.glob("*.info.json"):
            try:
                with open(info_file, 'r') as f:
                    info = json.load(f)
                cached_entries.append(info)
            except Exception as e:
                logger.warning(f"Could not read cache info file {info_file}: {str(e)}")
        
        return cached_entries 

    # Compatibility convenience methods
    def set(self, cache_key: str, df: pd.DataFrame) -> None:
        """Compatibility wrapper: save a single DataFrame as prediction X_test cache.

        Creates a minimal data_result and stores it as a 'prediction' cache entry.
        """
        data_result = {
            'X_test': df,
            'y_test': pd.Series([]),
            'target_column': '',
            'feature_count': df.shape[1] if hasattr(df, 'shape') else 0,
            'prediction_horizon': 0,
            'train_samples': 0,
            'test_samples': len(df) if hasattr(df, '__len__') else 0,
            'removed_features': {},
            'diversity_analysis': {}
        }
        # Use existing save helper with data_type='prediction'
        self.save_cleaned_data(data_result, cache_key=cache_key, data_type='prediction')

    def get(self, cache_key: str) -> pd.DataFrame:
        """Compatibility wrapper: load cached prediction X_test for given key."""
        result = self.load_cleaned_data(cache_key=cache_key, data_type='prediction')
        # Expect X_test in result
        if 'X_test' not in result:
            raise KeyError(f"Cached X_test not found for key: {cache_key}")
        return result['X_test']
    
# Start Generation Here
    def get_cache_age_hours(self, cache_key: str, data_type: str = "training") -> Optional[float]:
        """
        Get the age of cached data in hours
        
        Args:
            cache_key: Unique cache identifier
            data_type: Type of data ('training' or 'prediction')
            
        Returns:
            Age of cache in hours, or None if cache doesn't exist
        """
        paths = self._get_cache_paths(cache_key, data_type)
        
        # Check if cache exists
        if not self.cache_exists(cache_key, data_type):
            return None
        
        # Get the modification time of any cache file (use info.json as reference)
        info_file = paths['info']
        if not info_file.exists():
            return None
            
        try:
            # Get file modification time
            mtime = info_file.stat().st_mtime
            cache_datetime = datetime.fromtimestamp(mtime)
            current_datetime = datetime.now()
            
            # Calculate age in hours
            age_hours = (current_datetime - cache_datetime).total_seconds() / 3600
            return age_hours
            
        except Exception as e:
            logger.error(f"Error getting cache age: {str(e)}")
            return None

def main():
    """
    Main function to clear the entire cache
    """
    cache = CleanedDataCache()
    cache.clear_cache()
    print("🧹 Entire cache cleared successfully!")

if __name__ == "__main__":
    main()

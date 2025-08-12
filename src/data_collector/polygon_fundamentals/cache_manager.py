"""
Fundamental Cache Manager

This module handles reading cached fundamental data from JSON files
to reduce API calls and improve performance.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.utils.logger import get_logger

logger = get_logger(__name__)

class FundamentalCacheManager:
    """Manages cached fundamental data from JSON files"""
    
    def __init__(self, cache_dir: str = "data/cache/fundamentals"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _parse_date_from_filename(self, filename: str) -> Optional[datetime]:
        """Extract date from filename like 'AAPL_financials_20250805.json'"""
        try:
            # Extract date part (last part before .json)
            date_part = filename.replace('.json', '').split('_')[-1]
            return datetime.strptime(date_part, '%Y%m%d')
        except (ValueError, IndexError):
            return None
    
    def _is_cache_valid(self, file_date: datetime) -> bool:
        """Check if cache file is valid (yesterday or newer)"""
        valid_date = datetime.now() - timedelta(days=7)
        return file_date >= valid_date
    
    def get_cached_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get cached fundamental data for a ticker if available and valid
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Cached data dict if valid cache exists, None otherwise
        """
        try:
            # Look for cache file with pattern: TICKER_financials_YYYYMMDD.json
            pattern = f"{ticker}_financials_*.json"
            cache_files = list(self.cache_dir.glob(pattern))
            
            if not cache_files:
                logger.info(f"No cache files found for {ticker}")
                return None
            
            # Find the most recent valid cache file
            valid_caches = []
            for cache_file in cache_files:
                file_date = self._parse_date_from_filename(cache_file.name)
                if file_date and self._is_cache_valid(file_date):
                    valid_caches.append((cache_file, file_date))
            
            if not valid_caches:
                logger.info(f"No valid cache files found for {ticker} from {len(cache_files)} files")
                return None
            
            # Get the most recent cache file
            most_recent_cache = max(valid_caches, key=lambda x: x[1])
            cache_file, file_date = most_recent_cache
            
            logger.info(f"Using cached data for {ticker} from {file_date.strftime('%Y-%m-%d')}")
            
            # Read and parse JSON file
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            return cached_data
            
        except Exception as e:
            logger.error(f"Error reading cache for {ticker}: {e}")
            return None
    
    def list_available_caches(self) -> Dict[str, List[str]]:
        """
        List all available cache files grouped by ticker
        
        Returns:
            Dict mapping ticker to list of available cache dates
        """
        try:
            cache_files = list(self.cache_dir.glob("*_financials_*.json"))
            ticker_caches = {}
            
            for cache_file in cache_files:
                ticker = cache_file.name.split('_')[0]
                file_date = self._parse_date_from_filename(cache_file.name)
                
                if ticker not in ticker_caches:
                    ticker_caches[ticker] = []
                
                if file_date:
                    ticker_caches[ticker].append(file_date.strftime('%Y-%m-%d'))
            
            return ticker_caches
            
        except Exception as e:
            logger.error(f"Error listing cache files: {e}")
            return {}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cache usage
        
        Returns:
            Dict with cache statistics
        """
        try:
            cache_files = list(self.cache_dir.glob("*_financials_*.json"))
            total_files = len(cache_files)
            
            valid_caches = 0
            expired_caches = 0
            
            for cache_file in cache_files:
                file_date = self._parse_date_from_filename(cache_file.name)
                if file_date:
                    if self._is_cache_valid(file_date):
                        valid_caches += 1
                    else:
                        expired_caches += 1
            
            return {
                'total_cache_files': total_files,
                'valid_caches': valid_caches,
                'expired_caches': expired_caches,
                'cache_hit_rate': valid_caches / total_files if total_files > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def clear_expired_caches(self) -> int:
        """
        Remove expired cache files
        
        Returns:
            Number of files removed
        """
        try:
            cache_files = list(self.cache_dir.glob("*_financials_*.json"))
            removed_count = 0
            
            for cache_file in cache_files:
                file_date = self._parse_date_from_filename(cache_file.name)
                if file_date and not self._is_cache_valid(file_date):
                    cache_file.unlink()
                    removed_count += 1
                    logger.info(f"Removed expired cache: {cache_file.name}")
            
            logger.info(f"Removed {removed_count} expired cache files")
            return removed_count
            
        except Exception as e:
            logger.error(f"Error clearing expired caches: {e}")
            return 0 

    def save_cache(
        self,
        ticker: str,
        data: Dict[str, Any],
        overwrite: bool = True,
    ) -> Optional[Path]:
        """
        Save fundamental data to cache as JSON.

        The file name follows the pattern: TICKER_financials_YYYYMMDD.json

        Args:
            ticker: Stock ticker symbol
            data: Data to be cached (JSON-serializable)
            as_of: Date to use in the filename; defaults to now if not provided
            overwrite: If False, will not overwrite an existing file for the same date

        Returns:
            Path to the saved cache file if successful, None otherwise
        """
        try:
            normalized_ticker = (ticker or "").strip().upper()
            if not normalized_ticker:
                logger.error("Ticker is required to save cache")
                return None

            cache_date = datetime.now()
            filename = f"{normalized_ticker}_financials_{cache_date.strftime('%Y%m%d')}.json"
            file_path = self.cache_dir / filename

            if file_path.exists() and not overwrite:
                logger.info(
                    f"Cache file already exists and overwrite is False: {file_path.name}"
                )
                return None

            # Write JSON with UTF-8 encoding and readable formatting
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    data,
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=self._json_fallback_serializer,
                )

            logger.info(
                f"Saved cache for {normalized_ticker} at {file_path.name}"
            )
            return file_path
        except Exception as e:
            logger.error(f"Error saving cache for {ticker}: {e}")
            return None

    def _json_fallback_serializer(self, obj: Any) -> str:
        """Fallback serializer for objects not natively JSON serializable."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        # Final fallback to string representation
        try:
            return str(obj)
        except Exception:  # pragma: no cover - extremely rare
            return repr(obj)

if __name__ == "__main__":
    cache_manager = FundamentalCacheManager()
    cache_manager.clear_expired_caches()
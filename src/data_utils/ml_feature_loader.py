"""
ML Feature Loader Module

This module provides the MLFeatureLoader class and convenience functions for loading
and preparing features for ML models from consolidated yearly parquet files.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from datetime import date
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

from src.feature_engineering.technical_indicators.feature_storage import FeatureStorage
from src.feature_engineering.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__, utility='feature_engineering')


class MLFeatureLoader:
    """
    Utility class for loading and preparing features for ML models
    """
    
    def __init__(self, storage: Optional[FeatureStorage] = None, use_consolidated: bool = True):
        """
        Initialize ML feature loader
        
        Args:
            storage: FeatureStorage instance (creates new if None)
            use_consolidated: Whether to use consolidated yearly files (recommended)
        """
        self.storage = storage or FeatureStorage()
        self.use_consolidated = use_consolidated
        self.scalers = {}
        
        # Initialize consolidated storage if enabled
        if self.use_consolidated:
            from src.feature_engineering.technical_indicators.consolidated_storage import ConsolidatedFeatureStorage
            self.consolidated_storage = ConsolidatedFeatureStorage()
            logger.info("MLFeatureLoader initialized with consolidated yearly files")
        else:
            logger.info("MLFeatureLoader initialized with individual ticker files")
        
    def load_ml_dataset(self, ticker: Optional[str] = None, 
                        start_date: Optional[date] = "2024-01-01",
                        end_date: Optional[date] = "2025-07-01",
                        categories: Optional[List[str]] = None,
                        target_column: str = None,
                        prediction_horizon: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load features with pre-calculated future price targets
        
        Args:
            ticker: Single ticker symbol (None for ALL tickers)
            start_date: Start date filter
            end_date: End date filter
            categories: Feature categories to include
            target_column: Deprecated - uses Future_High_XD columns instead
            prediction_horizon: Days ahead (determines which Future_High_XD column to use)
            
        Returns:
            Tuple of (features DataFrame, target Series from Future_High_XD)
        """
        # Apply config defaults
        prediction_horizon = prediction_horizon or config.ml.DEFAULT_PREDICTION_HORIZON
        
        if ticker is None:
            logger.info(f"Loading ML dataset for ALL tickers using {'consolidated' if self.use_consolidated else 'individual'} storage")
        else:
            logger.info(f"Loading ML dataset for ticker '{ticker}' using {'consolidated' if self.use_consolidated else 'individual'} storage")
        
        if self.use_consolidated:
            return self._load_from_consolidated(ticker, start_date, end_date, categories, prediction_horizon)
        else:
            return self._load_from_individual_files(ticker, start_date, end_date, categories, prediction_horizon)
    
    def _load_from_consolidated(self, ticker: Optional[str] = None, 
                                start_date: Optional[date] = "2024-01-01",
                                end_date: Optional[date] = "2025-06-28",
                                categories: Optional[List[str]] = None,
                                prediction_horizon: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
        """Load data from consolidated yearly files with pre-calculated targets"""
        try:
            # Load consolidated features with filters
            features = self.consolidated_storage.load_consolidated_features(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                categories=categories
            )
            
            if features.empty:
                raise ValueError("No features loaded from consolidated storage")
            
            logger.info(f"Loaded {len(features)} records with {len(features.columns)} features from consolidated storage")
            
            # Extract pre-calculated future price targets
            target_column = f"Future_High_{prediction_horizon}D"
            
            if target_column in features.columns:
                # Use pre-calculated future high prices as targets
                targets = features[target_column].copy()
                targets.name = f'future_high_{prediction_horizon}d'
                
                # Remove target column from features to avoid data leakage
                features_clean = features.drop(columns=[col for col in features.columns 
                                                        if col.startswith('Future_')])
                
                # Remove rows with NaN targets
                valid_mask = targets.notna()
                features_final = features_clean[valid_mask].copy()
                targets_final = targets[valid_mask].copy()
                
                logger.info(f"Using pre-calculated target '{target_column}': {len(targets_final)} valid samples")
                logger.info(f"Target range: [{targets_final.min():.2f}, {targets_final.max():.2f}]")
                logger.info(f"Final dataset: {len(features_final)} samples, {len(features_final.columns)} features")
                
                return features_final, targets_final
            else:
                # Fallback to any available Future_High_XD column
                future_cols = [col for col in features.columns if col.startswith('Future_High_')]
                if future_cols:
                    fallback_col = future_cols[0]
                    logger.warning(f"Target column '{target_column}' not found. Using '{fallback_col}' instead.")
                    
                    targets = features[fallback_col].copy()
                    targets.name = fallback_col.lower()
                    
                    # Remove target columns from features
                    features_clean = features.drop(columns=[col for col in features.columns 
                                                            if col.startswith('Future_')])
                    
                    # Remove rows with NaN targets
                    valid_mask = targets.notna()
                    features_final = features_clean[valid_mask].copy()
                    targets_final = targets[valid_mask].copy()
                    
                    logger.info(f"Using fallback target '{fallback_col}': {len(targets_final)} valid samples")
                    return features_final, targets_final
                else:
                    raise ValueError(f"No future price target columns found. Expected '{target_column}' or similar Future_High_XD columns.")
                
        except Exception as e:
            logger.error(f"Error loading from consolidated storage: {str(e)}")
            raise
    
    def _load_from_individual_files(self, ticker: Optional[str], 
                                    start_date: Optional[date],
                                    end_date: Optional[date],
                                    categories: Optional[List[str]],
                                    prediction_horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Load data from individual ticker files with pre-calculated targets - single ticker only"""
        if ticker is None:
            raise ValueError("Individual file loading requires a specific ticker. Use consolidated storage for loading all tickers.")
        
        try:
            # Load features for the single ticker
            features, metadata = self.storage.load_features(
                ticker, start_date, end_date, categories
            )
            
            if features.empty:
                raise ValueError(f"No features found for {ticker}")
            
            # Add ticker column
            features['ticker'] = ticker
            
            # Extract pre-calculated future price targets
            target_column = f"Future_High_{prediction_horizon}D"
            
            if target_column in features.columns:
                # Use pre-calculated future high prices as targets
                targets = features[target_column].copy()
                targets.name = f'future_high_{prediction_horizon}d'
                
                # Remove target columns from features to avoid data leakage
                features_clean = features.drop(columns=[col for col in features.columns 
                                                       if col.startswith('Future_')])
                
                # Remove rows with NaN targets
                valid_mask = targets.notna()
                features_final = features_clean[valid_mask].copy()
                targets_final = targets[valid_mask].copy()
                
                logger.info(f"Created ML dataset for {ticker}: {len(features_final)} samples, {len(features_final.columns)} features")
                logger.info(f"Using pre-calculated target '{target_column}': {len(targets_final)} valid samples")
                
                return features_final, targets_final
            else:
                # Fallback to any available Future_High_XD column
                future_cols = [col for col in features.columns if col.startswith('Future_High_')]
                if future_cols:
                    fallback_col = future_cols[0]
                    logger.warning(f"Target column '{target_column}' not found for {ticker}. Using '{fallback_col}' instead.")
                    
                    targets = features[fallback_col].copy()
                    targets.name = fallback_col.lower()
                    
                    # Remove target columns from features
                    features_clean = features.drop(columns=[col for col in features.columns 
                                                           if col.startswith('Future_')])
                    
                    # Remove rows with NaN targets
                    valid_mask = targets.notna()
                    features_final = features_clean[valid_mask].copy()
                    targets_final = targets[valid_mask].copy()
                    
                    logger.info(f"Created ML dataset for {ticker}: {len(features_final)} samples, {len(features_final.columns)} features")
                    return features_final, targets_final
                else:
                    raise ValueError(f"No future price target columns found for {ticker}. Expected '{target_column}' or similar Future_High_XD columns.")

        except Exception as e:
            logger.error(f"Error loading features for {ticker}: {str(e)}")
            raise
    
    def _handle_missing_values(self, features: pd.DataFrame, target: pd.Series, method: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle missing values in features and target"""
        if method == 'drop':
            # Drop rows with any missing values
            valid_idx = features.dropna().index.intersection(target.dropna().index)
            return features.loc[valid_idx], target.loc[valid_idx]
        
        elif method == 'fill_mean':
            features_filled = features.fillna(features.mean())
            return features_filled, target.fillna(target.mean())
        
        elif method == 'fill_median':
            features_filled = features.fillna(features.median())
            return features_filled, target.fillna(target.median())
        
        elif method == 'fill_zero':
            return features.fillna(0), target.fillna(0)
        
        else:
            raise ValueError(f"Unknown missing value method: {method}")
    
    def _select_features(self, features: pd.DataFrame, target: pd.Series, config: Dict) -> pd.DataFrame:
        """Select features based on configuration"""
        # This is a placeholder for feature selection logic
        # Could implement correlation filtering, variance filtering, etc.
        return features
    
    def _create_splits(self, features: pd.DataFrame, target: pd.Series, 
                        test_size: float, val_size: float) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """Create train/validation/test splits"""
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, target, test_size=test_size, random_state=42, stratify=None
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=None
        )
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
    
    def _scale_features(self, splits: Dict, method: str) -> Dict:
        """Scale features using specified method"""
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit scaler on training data only
        numeric_columns = splits['X_train'].select_dtypes(include=[np.number]).columns
        
        # Fit and transform training data
        splits['X_train'][numeric_columns] = scaler.fit_transform(splits['X_train'][numeric_columns])
        
        # Transform validation and test data
        splits['X_val'][numeric_columns] = scaler.transform(splits['X_val'][numeric_columns])
        splits['X_test'][numeric_columns] = scaler.transform(splits['X_test'][numeric_columns])
        
        # Store scaler for future use
        self.scalers[method] = scaler
        
        return splits
    
    def _get_feature_category(self, feature_name: str) -> str:
        """Determine feature category from feature name"""
        feature_name_lower = feature_name.lower()
        
        if any(x in feature_name_lower for x in ['sma', 'ema', 'macd', 'ichimoku']):
            return 'trend'
        elif any(x in feature_name_lower for x in ['rsi', 'stoch', 'roc', 'williams']):
            return 'momentum'
        elif any(x in feature_name_lower for x in ['bb', 'bollinger', 'atr', 'volatility']):
            return 'volatility'
        elif any(x in feature_name_lower for x in ['obv', 'vpt', 'ad_line', 'volume', 'mfi']):
            return 'volume'
        else:
            return 'basic'


# Convenience functions
def load_ml_ready_data(ticker: Optional[str] = None, 
                        categories: Optional[List[str]] = None,
                        start_date: Optional[date] = None,
                        end_date: Optional[date] = None,
                        prediction_horizon: int = 10,
                        use_consolidated: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function to load ML-ready dataset with pre-calculated future targets
    
    Args:
        ticker: Single ticker symbol (None for ALL tickers)
        categories: Feature categories to include
        start_date: Start date filter
        end_date: End date filter
        prediction_horizon: Days ahead (determines which Future_High_XD column to use)
        use_consolidated: Whether to use consolidated yearly files (recommended)
        
    Returns:
        Tuple of (features DataFrame, target Series from Future_High_XD)
    """
    loader = MLFeatureLoader(use_consolidated=use_consolidated)
    return loader.load_ml_dataset(ticker, start_date, end_date, categories, 
                                    prediction_horizon=prediction_horizon)


def load_yearly_data(year: int, 
                        ticker: Optional[str] = None,
                        categories: Optional[List[str]] = None,
                        prediction_horizon: int = 10) -> pd.DataFrame:
    """
    Convenience function to load data for a specific year
    
    Args:
        year: Year to load (e.g., 2024, 2025)
        ticker: Single ticker symbol (None for ALL available)
        categories: Feature categories to include
        prediction_horizon: Days ahead (determines which Future_High_XD column to use)
        
    Returns:
        DataFrame with features for the specified year
    """
    from datetime import date
    
    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    loader = MLFeatureLoader(use_consolidated=True)
    features = loader.consolidated_storage.load_consolidated_features(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                categories=categories
            )
    return features


def load_date_range_data(start_date: date, 
                        end_date: date,
                        ticker: Optional[str] = None,
                        categories: Optional[List[str]] = None,
                        prediction_horizon: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function to load data for a specific date range with pre-calculated future targets
    
    Args:
        start_date: Start date
        end_date: End date
        ticker: Single ticker symbol (None for ALL available)
        categories: Feature categories to include
        prediction_horizon: Days ahead (determines which Future_High_XD column to use)
        
    Returns:
        Tuple of (features DataFrame, target Series from Future_High_XD)
    """
    loader = MLFeatureLoader(use_consolidated=True)
    return loader.load_ml_dataset(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        categories=categories,
        prediction_horizon=prediction_horizon
    )


def load_all_data(ticker: Optional[str] = None) -> pd.DataFrame:
    """
    Load all data from yearly parquet files with comprehensive feature analysis
    
    Args:
        ticker: Single ticker symbol (None for ALL available tickers)
        
    Returns:
        Combined DataFrame with all loaded data
    """
    years_to_load = [2024, 2025]
    logger.info("=" * 80)
    logger.info("LOADING ALL DATA FROM YEARLY PARQUET FILES")
    logger.info("=" * 80)
    
    try:
        from src.feature_engineering.data_loader import StockDataLoader
        data_loader = StockDataLoader()
        
        # Load data for multiple years
        all_features_data = []
        
        for year in years_to_load:
            logger.info(f"Loading data for year {year}...")
            
            try:
                features = load_yearly_data(
                    year=year,
                    ticker=ticker,
                    categories=None
                )
                
                if not features.empty:
                    logger.info(f"✓ {year} data loaded: {features.shape[0]:,} records, {features.shape[1]} columns")
                    
                    # Add year column for tracking
                    features['data_year'] = year
                    all_features_data.append(features)
                    
                    # Log unique tickers for this year
                    if 'ticker' in features.columns:
                        year_tickers = features['ticker'].unique()
                        logger.info(f"📊 Tickers in {year}: {len(year_tickers)} ({year_tickers[:5]}...)")
                    
                    # Log date range for this year
                    if 'date' in features.columns:
                        features['date'] = pd.to_datetime(features['date'])
                        date_range = f"{features['date'].min().date()} to {features['date'].max().date()}"
                        logger.info(f"📅 Date range: {date_range}")
                else:
                    logger.warning(f"No data found for {year}")
                    
            except Exception as e:
                logger.error(f"Error loading {year} data: {str(e)}")
        
        # Combine all yearly data
        if all_features_data:
            logger.info("Combining and analyzing ALL yearly data...")
            combined_features = pd.concat(all_features_data, ignore_index=True)
            
            logger.info(f"✓ Combined dataset: {combined_features.shape[0]:,} total records")
            logger.info(f"✓ Total features: {combined_features.shape[1]} columns")
            
            # Add ticker_id column from database metadata
            if 'ticker' in combined_features.columns:
                logger.info("Adding ticker_id column from database metadata...")
                
                # Get unique tickers from the combined data
                unique_tickers = combined_features['ticker'].unique()
                logger.info("📊 Fetching metadata for all tickers in one query...")
                
                try:
                    # Get all ticker metadata in one efficient query
                    all_metadata_df = data_loader.get_ticker_metadata(ticker=None)
                    
                    if not all_metadata_df.empty:
                        # Create ticker_id mapping from the DataFrame
                        ticker_id_mapping = dict(zip(all_metadata_df['ticker'], all_metadata_df['id']))
                        
                        # Map ticker_id to the combined features
                        combined_features['ticker_id'] = combined_features['ticker'].map(ticker_id_mapping)
                        
                        # Calculate success statistics
                        successful_mappings = combined_features['ticker_id'].notna().sum()
                        total_records = len(combined_features)
                        unique_mapped = len([t for t in unique_tickers if t in ticker_id_mapping])
                        
                        logger.info(f"✅ Successfully mapped ticker_id for {unique_mapped}/{len(unique_tickers)} unique tickers")
                        logger.info(f"✅ {successful_mappings:,}/{total_records:,} total records have ticker_id")
                        
                        # Log statistics about ticker_id mapping
                        null_ticker_ids = combined_features['ticker_id'].isnull().sum()
                        if null_ticker_ids > 0:
                            logger.warning(f"⚠ {null_ticker_ids:,} records have null ticker_id")
                            missing_tickers = [t for t in unique_tickers if t not in ticker_id_mapping]
                            if missing_tickers:
                                logger.warning(f"⚠ Tickers not found in database: {missing_tickers[:10]}{'...' if len(missing_tickers) > 10 else ''}")
                    else:
                        logger.error("❌ No ticker metadata retrieved from database")
                        combined_features['ticker_id'] = None
                        
                except Exception as e:
                    logger.error(f"❌ Error fetching all ticker metadata: {str(e)}")
                    combined_features['ticker_id'] = None
            
            # Close data loader connection
            data_loader.close()
            
            return combined_features
        else:
            logger.warning("No yearly data could be loaded")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"CRITICAL ERROR in load_all_data: {str(e)}")
        raise 
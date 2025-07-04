"""
ML Utilities for Feature Engineering

This module provides utilities for loading and preparing features for ML models.
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
    
    def prepare_features(self, features: pd.DataFrame, 
                        target: pd.Series,
                        test_size: float = None,
                        validation_size: float = None,
                        scaling_method: str = None,
                        handle_missing: str = None,
                        feature_selection: Optional[Dict] = None) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        Prepare features for ML training
        
        Args:
            features: Features DataFrame
            target: Target Series
            test_size: Proportion for test set
            validation_size: Proportion for validation set
            scaling_method: 'standard', 'minmax', 'robust', or 'none'
            handle_missing: 'drop', 'fill_mean', 'fill_median', 'fill_zero'
            feature_selection: Feature selection configuration
            
        Returns:
            Dictionary with train/val/test splits
        """
        # Apply config defaults
        test_size = test_size or config.ml.DEFAULT_TEST_SIZE
        validation_size = validation_size or config.ml.DEFAULT_VALIDATION_SIZE
        scaling_method = scaling_method or config.ml.DEFAULT_SCALING_METHOD
        handle_missing = handle_missing or config.ml.DEFAULT_MISSING_STRATEGY
        
        logger.info("Preparing features for ML training")
        
        # Handle missing values
        features_clean, target_clean = self._handle_missing_values(
            features, target, handle_missing
        )
        
        # Feature selection
        if feature_selection:
            features_clean = self._select_features(features_clean, target_clean, feature_selection)
        
        # Create train/val/test splits
        splits = self._create_splits(features_clean, target_clean, test_size, validation_size)
        
        # Scale features
        if scaling_method != 'none':
            splits = self._scale_features(splits, scaling_method)
        
        logger.info(f"Prepared dataset: Train={len(splits['X_train'])}, "
                    f"Val={len(splits['X_val'])}, Test={len(splits['X_test'])}")
        
        return splits
    
    def get_feature_importance_data(self, ticker: Optional[str] = None,
                                    categories: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get feature statistics for importance analysis
        
        Args:
            ticker: Single ticker symbol (None for ALL tickers)
            categories: Feature categories to analyze
            
        Returns:
            DataFrame with feature statistics
        """
        if ticker is None:
            logger.info("Analyzing feature importance for ALL tickers")
            # For ALL tickers, use consolidated storage
            if self.use_consolidated:
                features = self.consolidated_storage.load_consolidated_features(
                    ticker=None, categories=categories
                )
                return self._analyze_consolidated_features(features)
            else:
                raise ValueError("Loading all tickers requires consolidated storage. Set use_consolidated=True.")
        else:
            logger.info(f"Analyzing feature importance for ticker '{ticker}'")
        
        feature_stats = []
        
        try:
        # Load features for single ticker
            features, metadata = self.storage.load_features(ticker, categories=categories)
            
            if features.empty:
                logger.warning(f"No features found for {ticker}")
                return pd.DataFrame()
            
            # Calculate statistics for each feature
            for col in features.columns:
                if features[col].dtype in ['float64', 'int64']:
                    stats = {
                        'ticker': ticker,
                        'feature': col,
                        'category': self._get_feature_category(col),
                        'mean': features[col].mean(),
                        'std': features[col].std(),
                        'min': features[col].min(),
                        'max': features[col].max(),
                        'missing_pct': features[col].isnull().mean() * 100,
                        'unique_values': features[col].nunique(),
                        'skewness': features[col].skew(),
                        'kurtosis': features[col].kurtosis()
                    }
                    feature_stats.append(stats)
                    
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {str(e)}")
            return pd.DataFrame()
        
        if not feature_stats:
            logger.warning(f"No features found for {ticker}")
            return pd.DataFrame()
        
        stats_df = pd.DataFrame(feature_stats)
        logger.info(f"Analyzed {len(stats_df)} features for ticker '{ticker}'")
        
        return stats_df
    
    def _analyze_consolidated_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Analyze features from consolidated storage"""
        if features.empty:
            return pd.DataFrame()
        
        feature_stats = []
        
        # Get all numeric columns except metadata
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        metadata_cols = ['ticker', 'date', 'data_year']
        feature_cols = [col for col in numeric_cols if col not in metadata_cols]
        
        for col in feature_cols:
            stats = {
                'feature': col,
                'category': self._get_feature_category(col),
                'mean': features[col].mean(),
                'std': features[col].std(),
                'min': features[col].min(),
                'max': features[col].max(),
                'missing_pct': features[col].isnull().mean() * 100,
                'unique_values': features[col].nunique(),
                'skewness': features[col].skew(),
                'kurtosis': features[col].kurtosis()
            }
            feature_stats.append(stats)
        
        if not feature_stats:
            return pd.DataFrame()
        
        stats_df = pd.DataFrame(feature_stats)
        logger.info(f"Analyzed {len(stats_df)} features across all tickers")
        
        return stats_df
    
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

def prepare_ml_features(features: pd.DataFrame, target: pd.Series,
                        test_size: float = None, scaling_method: str = None) -> Dict:
    """
    Convenience function to prepare features for ML
    
    Args:
        features: Features DataFrame
        target: Target Series
        test_size: Test set proportion
        scaling_method: Scaling method to use
        
    Returns:
        Dictionary with prepared train/val/test splits
    """
    # Apply config defaults
    test_size = test_size or config.ml.DEFAULT_TEST_SIZE
    scaling_method = scaling_method or config.ml.DEFAULT_SCALING_METHOD
    
    loader = MLFeatureLoader(use_consolidated=True)  # Use consolidated by default
    return loader.prepare_features(features, target, test_size=test_size, scaling_method=scaling_method)

def load_yearly_data(year: int, 
                        ticker: Optional[str] = None,
                        categories: Optional[List[str]] = None,
                        prediction_horizon: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function to load data for a specific year with pre-calculated future targets
    
    Args:
        year: Year to load (e.g., 2024, 2025)
        ticker: Single ticker symbol (None for ALL available)
        categories: Feature categories to include
        prediction_horizon: Days ahead (determines which Future_High_XD column to use)
        
    Returns:
        Tuple of (features DataFrame, target Series from Future_High_XD)
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

def load_all_data(ticker: Optional[str] = None) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Load all data from yearly parquet files with comprehensive feature analysis
    
    Args:
        ticker: Single ticker symbol (None for ALL available tickers)
        
    Returns:
        Dictionary containing:
        - 'combined_data': Combined DataFrame with all loaded data
        - 'individual_years': Dict of DataFrames by year
        - 'analysis': Comprehensive analysis results (if return_analysis=True)
        - 'success': Boolean indicating if data was successfully loaded
    """
    years_to_load = [2024, 2025]
    logger.info("=" * 80)
    logger.info("LOADING ALL DATA FROM YEARLY PARQUET FILES")
    logger.info("=" * 80)
    
    result = {
        'combined_data': pd.DataFrame(),
        'individual_years': {},
        'analysis': {},
        'success': False
    }
    
    try:
        # Initialize ML Feature Loader with consolidated storage
        loader = MLFeatureLoader(use_consolidated=True)
        
        # Initialize data loader for ticker metadata
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
                    result['individual_years'][year] = features.copy()
                    
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
                logger.info(f"📊 Fetching metadata for all tickers in one query...")
                
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

def prepare_ml_data_for_training(prediction_horizon: int = 10, 
                                split_date: str = '2025-02-01',
                                ticker: Optional[str] = None) -> Dict[str, Union[pd.DataFrame, pd.Series, str]]:
    """
    Comprehensive data preparation function for ML training
    
    This function loads all available data, prepares features and targets,
    creates temporal train/test splits, and performs data cleaning.
    
    Args:
        prediction_horizon: Days ahead for target prediction (default: 10)
        split_date: Date to split train/test data (default: '2025-02-01')
        ticker: Single ticker symbol (None for ALL available tickers)
        
    Returns:
        Dictionary containing:
        - 'X_train': Training features
        - 'X_test': Test features  
        - 'y_train': Training targets
        - 'y_test': Test targets
        - 'target_column': Name of target column used
        - 'train_date_range': Date range of training data
        - 'test_date_range': Date range of test data
        - 'feature_count': Number of features
        - 'train_samples': Number of training samples
        - 'test_samples': Number of test samples
    """
    logger.info("=" * 80)
    logger.info("🎯 COMPREHENSIVE ML DATA PREPARATION")
    logger.info("=" * 80)
    
    try:
        # 1. Load data using load_all_data
        logger.info("1. Loading dataset using load_all_data()...")
        combined_data = load_all_data(ticker=ticker)
        
        if combined_data.empty:
            raise ValueError("No data loaded. Check data availability.")
        
        logger.info(f"✅ Data loaded: {combined_data.shape[0]:,} records, {combined_data.shape[1]} features")
        
        # 2. Prepare features and targets
        logger.info("2. Preparing features and targets...")
        
        # Define and validate target column
        target_column = f"Future_High_{prediction_horizon}D"
        
        if target_column not in combined_data.columns:
            # Look for any Future_High_XD column as fallback
            future_cols = [col for col in combined_data.columns if col.startswith('Future_High_')]
            if not future_cols:
                raise ValueError(f"No future price target columns found. Expected '{target_column}' or similar.")
            target_column = future_cols[0]
            logger.warning(f"⚠ Using fallback target column: {target_column}")
        
        # Extract targets
        y = combined_data[target_column].copy()
        
        # Prepare features (exclude metadata and target columns)
        exclude_cols = [
            'ticker', 'date', 'data_year',  # Metadata
            'feature_version', 'calculation_date', 'start_date', 'end_date', 
            'feature_categories', 'file_path', 'warnings',  # Feature engineering metadata
            'quality_score', 'record_count', 'total_features', 'file_size_mb'  # Data quality metrics
        ]
        # Also exclude all Future_* columns to avoid data leakage
        future_cols = [col for col in combined_data.columns if col.startswith('Future_')]
        exclude_cols.extend(future_cols)
        
        feature_cols = [col for col in combined_data.columns if col not in exclude_cols]
        X = combined_data[feature_cols].copy()
        
        # 3. Add temporal features
        logger.info("3. Adding temporal features...")
        
        # Get the date column (before we excluded it)
        date_col = combined_data['date'].copy()
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(date_col):
            date_col = pd.to_datetime(date_col)
        
        # Create date-based features
        # Date as integer (days since reference date)
        X['date_int'] = (date_col - pd.Timestamp('2020-01-01')).dt.days
        X['year'] = date_col.dt.year
        X['month'] = date_col.dt.month
        X['day_of_year'] = date_col.dt.dayofyear
        X['quarter'] = date_col.dt.quarter
        
        logger.info("✅ Added temporal features: date_int, year, month, day_of_year, quarter")
        logger.info(f"📋 Total features: {len(X.columns)}")
        
        # 4. Clean data
        logger.info("4. Cleaning and preprocessing data...")
        
        # Remove rows with NaN targets
        valid_mask = y.notna() & np.isfinite(y)
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()
        
        logger.info(f"✅ After target cleaning: {len(X_clean)} valid samples")
        
        # Handle missing values in features
        # Replace infinite values with NaN first
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with median (more robust than mean)
        X_clean = X_clean.fillna(X_clean.median())
        
        # Final safety check - replace any remaining problematic values
        X_clean = X_clean.replace([np.nan, np.inf, -np.inf], 0)
        
        logger.info(f"✅ Final dataset: {len(X_clean)} samples, {len(X_clean.columns)} features")
        logger.info(f"✅ Target range: [{y_clean.min():.2f}, {y_clean.max():.2f}]")
        
        # 5. Date-based train/test split
        logger.info("5. Creating date-based train/test split...")
        
        # Check if date column exists
        if 'date' not in combined_data.columns:
            raise ValueError("'date' column not found in data. Cannot perform date-based split.")
        
        # Apply the same valid_mask to get corresponding dates
        dates_clean = combined_data['date'][valid_mask].copy()
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(dates_clean):
            dates_clean = pd.to_datetime(dates_clean)
        
        # Define split date
        split_date_dt = pd.to_datetime(split_date)
        
        # Create train/test masks based on date
        train_mask = dates_clean < split_date_dt
        test_mask = dates_clean >= split_date_dt
        
        # Split the data
        X_train = X_clean[train_mask].copy()
        X_test = X_clean[test_mask].copy()
        y_train = y_clean[train_mask].copy()
        y_test = y_clean[test_mask].copy()
        
        # Get date ranges for logging
        train_date_range = f"{dates_clean[train_mask].min().strftime('%Y-%m-%d')} to {dates_clean[train_mask].max().strftime('%Y-%m-%d')}" if train_mask.any() else "No training data"
        test_date_range = f"{dates_clean[test_mask].min().strftime('%Y-%m-%d')} to {dates_clean[test_mask].max().strftime('%Y-%m-%d')}" if test_mask.any() else "No test data"
        
        logger.info(f"✅ Train set: {len(X_train)} samples ({train_date_range})")
        logger.info(f"✅ Test set: {len(X_test)} samples ({test_date_range})")
        
        # Validation checks
        if len(X_test) == 0:
            raise ValueError(f"No test data found after {split_date}. Check your data date range.")
        
        if len(X_train) == 0:
            raise ValueError(f"No training data found before {split_date}. Check your data date range.")
        
        # 6. Prepare return dictionary
        result = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'target_column': target_column,
            'train_date_range': train_date_range,
            'test_date_range': test_date_range,
            'feature_count': len(X_train.columns),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'prediction_horizon': prediction_horizon,
            'split_date': split_date
        }
        
        # 7. Summary logging
        logger.info("=" * 80)
        logger.info("✅ ML DATA PREPARATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"🎯 Target: {target_column} ({prediction_horizon}-day horizon)")
        logger.info(f"📊 Features: {result['feature_count']} total")
        logger.info(f"📅 Train period: {train_date_range}")
        logger.info(f"📅 Test period: {test_date_range}")
        logger.info(f"🔢 Train samples: {result['train_samples']:,}")
        logger.info(f"🔢 Test samples: {result['test_samples']:,}")
        logger.info(f"📏 Split date: {split_date}")
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR in prepare_ml_data_for_training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def prepare_ml_data_for_prediction(prediction_horizon: int = 10) -> Dict[str, Union[pd.DataFrame, pd.Series, str]]:
    """
    Comprehensive data preparation function for ML training
    
    This function loads all available data, prepares features and targets,
    creates temporal train/test splits, and performs data cleaning.
    
    Args:
        prediction_horizon: Days ahead for target prediction (default: 10)
        split_date: Date to split train/test data (default: '2025-02-01')
        ticker: Single ticker symbol (None for ALL available tickers)
        
    Returns:
        Dictionary containing:
        - 'X_test': Test features  
        - 'y_test': Test targets
        - 'target_column': Name of target column used
        - 'feature_count': Number of features
    """
    logger.info("=" * 80)
    logger.info("🎯 COMPREHENSIVE ML DATA PREPARATION")
    logger.info("=" * 80)
    
    try:
        # 1. Load data using load_all_data
        logger.info("1. Loading dataset using load_all_data()...")
        combined_data = load_all_data(ticker=None)
        
        if combined_data.empty:
            raise ValueError("No data loaded. Check data availability.")
        
        logger.info(f"✅ Data loaded: {combined_data.shape[0]:,} records, {combined_data.shape[1]} features")
        
        # 2. Prepare features and targets
        logger.info("2. Preparing features and targets...")
        
        # Define and validate target column
        target_column = f"Future_High_{prediction_horizon}D"
        
        if target_column not in combined_data.columns:
            # Look for any Future_High_XD column as fallback
            future_cols = [col for col in combined_data.columns if col.startswith('Future_High_')]
            if not future_cols:
                raise ValueError(f"No future price target columns found. Expected '{target_column}' or similar.")
            target_column = future_cols[0]
            logger.warning(f"⚠ Using fallback target column: {target_column}")
        
        # Extract targets
        y = combined_data[target_column].copy()
        
        # Prepare features (exclude metadata and target columns)
        exclude_cols = [
            'ticker', 'date', 'data_year',  # Metadata
            'feature_version', 'calculation_date', 'start_date', 'end_date', 
            'feature_categories', 'file_path', 'warnings',  # Feature engineering metadata
            'quality_score', 'record_count', 'total_features', 'file_size_mb'  # Data quality metrics
        ]
        # Also exclude all Future_* columns to avoid data leakage
        future_cols = [col for col in combined_data.columns if col.startswith('Future_')]
        exclude_cols.extend(future_cols)
        
        feature_cols = [col for col in combined_data.columns if col not in exclude_cols]
        X = combined_data[feature_cols].copy()
        
        # 3. Add temporal features
        logger.info("3. Adding temporal features...")
        
        # Get the date column (before we excluded it)
        date_col = combined_data['date'].copy()
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(date_col):
            date_col = pd.to_datetime(date_col)
        
        # Create date-based features
        X['date_int'] = (date_col - pd.Timestamp('2020-01-01')).dt.days
        X['year'] = date_col.dt.year
        X['month'] = date_col.dt.month
        X['day_of_year'] = date_col.dt.dayofyear
        X['quarter'] = date_col.dt.quarter
        
        logger.info("✅ Added temporal features: date_int, year, month, day_of_year, quarter")
        logger.info(f"📋 Total features: {len(X.columns)}")
        
        # 4. Clean data
        logger.info("4. Cleaning and preprocessing data...")
        
        # Handle missing values in features
        # Replace infinite values with NaN first
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with median (more robust than mean)
        X = X.fillna(X.median())
        
        # Final safety check - replace any remaining problematic values
        X = X.replace([np.nan, np.inf, -np.inf], 0)
        
        logger.info(f"✅ Final dataset: {len(X)} samples, {len(X.columns)} features")
        logger.info(f"✅ Target range: [{y.min():.2f}, {y.max():.2f}]")
        
        # 5. Date-based train/test split
        logger.info("5. Creating date-based train/test split...")

        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(date_col):
            date_col = pd.to_datetime(date_col)
        
        # Define split date
        split_date_dt = pd.to_datetime('2025-02-01')
        
        # Create train/test masks based on date
        train_mask = date_col < split_date_dt
        test_mask = date_col >= split_date_dt
        
        # Split the data
        X_test = X[test_mask].copy()
        y_test = y[test_mask].copy()
        
        # Get date ranges for logging
        test_date_range = f"{date_col[test_mask].min().strftime('%Y-%m-%d')} to {date_col[test_mask].max().strftime('%Y-%m-%d')}" if test_mask.any() else "No test data"
        
        logger.info(f"✅ Test set: {len(X_test)} samples ({test_date_range})")
        
        # 6. Prepare return dictionary
        result = {
            'X_test': X_test,
            'y_test': y_test,
            'target_column': target_column,
            'test_date_range': test_date_range,
            'feature_count': len(X_test.columns),
            'prediction_horizon': prediction_horizon,
        }
        
        # 7. Summary logging
        logger.info("=" * 80)
        logger.info("✅ ML DATA PREPARATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"🎯 Target: {target_column} ({prediction_horizon}-day horizon)")
        logger.info(f"📊 Features: {result['feature_count']} total")
        logger.info(f"📅 Test period: {test_date_range}")
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR in prepare_ml_data_for_prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """
    Main function to load all data from yearly parquet files and log all available features
    """
    import sys
    
    print("=" * 80)
    print("ML UTILITIES - YEARLY PARQUET DATA LOADING & FEATURE ANALYSIS")
    print("=" * 80)
    
    try:
        # Initialize ML Feature Loader with consolidated storage
        print("1. Initializing MLFeatureLoader with consolidated yearly parquet files...")
        loader = MLFeatureLoader(use_consolidated=True)
        print("   ✓ MLFeatureLoader initialized successfully")
        print()
        
        # Load data for multiple years to get comprehensive feature set
        all_features_data = []
        years_to_load = [2024, 2025]  # Load multiple years for comprehensive analysis
        
        for year in years_to_load:
            print(f"4.{year - 2022}. Loading data for year {year}...")
            
            try:
                features, targets = load_yearly_data(
                    year=year,
                    ticker=None,  # Load ALL available tickers (no filter)
                    categories=None  # Load all categories
                )
                
                if not features.empty:
                    print(f"     ✓ {year} data loaded: {features.shape[0]:,} records, {features.shape[1]} columns")
                    
                    # Add year column for tracking
                    features['data_year'] = year
                    all_features_data.append(features)
                    
                    # Log unique tickers for this year
                    if 'ticker' in features.columns:
                        year_tickers = features['ticker'].unique()
                        print(f"     📊 Tickers in {year}: {len(year_tickers)} ({year_tickers[:5]}...)")
                    
                    # Log date range for this year
                    if 'date' in features.columns:
                        features['date'] = pd.to_datetime(features['date'])
                        date_range = f"{features['date'].min().date()} to {features['date'].max().date()}"
                        print(f"     📅 Date range: {date_range}")
                else:
                    print(f"     ⚠ No data found for {year}")
                    
            except Exception as e:
                print(f"     ❌ Error loading {year} data: {str(e)}")
            
            print()
        
        # Combine all yearly data
        if all_features_data:
            print("5. Combining and analyzing ALL yearly data...")
            combined_features = pd.concat(all_features_data, ignore_index=True)
            print(f"   ✓ Combined dataset: {combined_features.shape[0]:,} total records")
            print(f"   ✓ Total features: {combined_features.shape[1]} columns")
            print()
            
            # Comprehensive feature analysis
            print("6. COMPREHENSIVE FEATURE ANALYSIS")
            print("=" * 50)
            
            # Exclude metadata columns for feature analysis
            metadata_cols = ['ticker', 'date', 'data_year']
            feature_cols = [col for col in combined_features.columns if col not in metadata_cols]
            
            print(f"   Total feature columns (excluding metadata): {len(feature_cols)}")
            
            # Categorize features by type
            feature_categories = {
                'trend': [],
                'momentum': [],
                'volatility': [],
                'volume': [],
                'basic': [],
                'fundamental': [],
                'technical': [],
                'other': []
            }
            
            for col in feature_cols:
                col_lower = col.lower()
                if any(x in col_lower for x in ['sma', 'ema', 'macd', 'ichimoku', 'trend']):
                    feature_categories['trend'].append(col)
                elif any(x in col_lower for x in ['rsi', 'stoch', 'roc', 'williams', 'momentum']):
                    feature_categories['momentum'].append(col)
                elif any(x in col_lower for x in ['bb', 'bollinger', 'atr', 'volatility', 'vix']):
                    feature_categories['volatility'].append(col)
                elif any(x in col_lower for x in ['obv', 'vpt', 'ad_line', 'volume', 'mfi']):
                    feature_categories['volume'].append(col)
                elif any(x in col_lower for x in ['open', 'high', 'low', 'close', 'price']):
                    feature_categories['basic'].append(col)
                elif any(x in col_lower for x in ['pe', 'pb', 'roe', 'debt', 'revenue', 'earnings', 'fundamental']):
                    feature_categories['fundamental'].append(col)
                elif any(x in col_lower for x in ['technical', 'indicator', 'signal']):
                    feature_categories['technical'].append(col)
                else:
                    feature_categories['other'].append(col)
            
            # Feature statistics analysis
            print("   📈 FEATURE STATISTICS:")
            numeric_features = combined_features.select_dtypes(include=[np.number])
            
            if not numeric_features.empty:
                print(f"   Numeric features: {len(numeric_features.columns)}")
                print(f"   Missing values: {numeric_features.isnull().sum().sum():,}")
                print(f"   Missing percentage: {(numeric_features.isnull().sum().sum() / (len(numeric_features) * len(numeric_features.columns))) * 100:.2f}%")
            print()
            
            # Features with most missing values
            missing_counts = numeric_features.isnull().sum().sort_values(ascending=False)
            features_with_missing = missing_counts[missing_counts > 0]
            
            if not features_with_missing.empty:
                print("   ⚠ FEATURES WITH MISSING VALUES:")
                for feature, missing_count in features_with_missing.head(10).items():
                    missing_pct = (missing_count / len(numeric_features)) * 100
                    print(f"     {feature}: {missing_count:,} missing ({missing_pct:.1f}%)")
                print()
            
            # Data distribution by year and ticker
            print("   📅 DATA DISTRIBUTION:")
            if 'data_year' in combined_features.columns:
                year_dist = combined_features['data_year'].value_counts().sort_index()
                print("   By Year:")
                for year, count in year_dist.items():
                    print(f"     {year}: {count:,} records")
                print()
            
            if 'ticker' in combined_features.columns:
                ticker_dist = combined_features['ticker'].value_counts()
                print(f"   By Ticker (top 10 of {len(ticker_dist)} total):")
                for ticker, count in ticker_dist.head(10).items():
                    print(f"     {ticker}: {count:,} records")
                print()
            
            # Feature completeness analysis
            print("   ✅ FEATURE COMPLETENESS ANALYSIS:")
            completeness_stats = []
            
            for col in feature_cols:
                if col in combined_features.columns:
                    total_records = len(combined_features)
                    non_null_records = combined_features[col].count()
                    completeness_pct = (non_null_records / total_records) * 100
                    
                    completeness_stats.append({
                        'feature': col,
                        'completeness': completeness_pct,
                        'non_null_count': non_null_records,
                        'data_type': str(combined_features[col].dtype)
                    })
            
            if completeness_stats:
                completeness_df = pd.DataFrame(completeness_stats)
                
                # Features with 100% completeness
                complete_features = completeness_df[completeness_df['completeness'] == 100.0]
                print(f"   Perfect features (100% complete): {len(complete_features)}")
                
                # Features with >90% completeness
                mostly_complete = completeness_df[completeness_df['completeness'] > 90.0]
                print(f"   High-quality features (>90% complete): {len(mostly_complete)}")
                
                # Features with <50% completeness (potentially problematic)
                sparse_features = completeness_df[completeness_df['completeness'] < 50.0]
                print(f"   Sparse features (<50% complete): {len(sparse_features)}")
                
                if not sparse_features.empty:
                    print("   ⚠ Most sparse features:")
                    for _, row in sparse_features.head(5).iterrows():
                        print(f"     {row['feature']}: {row['completeness']:.1f}% complete")
            print()
        
            # Sample data preview
            print("   👀 SAMPLE DATA PREVIEW:")
            print("   First 5 records (selected columns):")
            
            # Select interesting columns for preview
            preview_cols = ['ticker', 'date']
            if feature_cols:
                # Add some representative features from each category
                for category, features in feature_categories.items():
                    if features:
                        preview_cols.extend(features[:2])  # First 2 from each category
            
            preview_cols = [col for col in preview_cols if col in combined_features.columns][:15]  # Limit to 15 columns
            
            sample_data = combined_features[preview_cols].head()
            print(sample_data.to_string(index=False))
            print()
            
        else:
            print("   ❌ No yearly data could be loaded")
            print()
        
        # Final summary
        print("7. FINAL SUMMARY")
        print("=" * 30)
        
        if all_features_data:
            total_records = sum(len(df) for df in all_features_data)
            total_features = len(feature_cols) if 'feature_cols' in locals() else 0
            total_years = len(all_features_data)
            
            print(f"   ✅ Successfully loaded {total_records:,} records")
            print(f"   ✅ Discovered {total_features} unique features")
            print(f"   ✅ Analyzed {total_years} years of data")
        else:
            print("   ❌ No data could be loaded from yearly parquet files")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 
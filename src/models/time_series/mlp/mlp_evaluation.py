"""
MLP Evaluation Module

This module contains feature selection methods, evaluation utilities, and data validation functions.
Includes threshold optimization and data cleaning utilities.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

from src.models.time_series.mlp.mlp_predictor import MLPPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLPEvaluationMixin:
    """
    Mixin class providing evaluation and feature selection functionality for MLPPredictor.
    """
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                                n_features_to_select: int = 50) -> List[str]:
        """
        Simple and reliable feature selection using correlation.
        
        Args:
            X: Feature matrix
            y: Target values  
            n_features_to_select: Number of features to select
            
        Returns:
            List of selected feature names
        """
        logger.info(f"🔍 Selecting {n_features_to_select} features from {len(X.columns)} total features...")
        
        # Clean data
        X_clean, y_clean = self._clean_data_simple(X, y)
        
        # Calculate correlations with target
        correlations = X_clean.corrwith(y_clean).abs()
        
        # Select top features
        selected_features = correlations.nlargest(n_features_to_select).index.tolist()
        
        logger.info(f"✅ Selected {len(selected_features)} features using correlation")
        logger.info(f"   Top 5 features: {selected_features[:5]}")
        
        return selected_features

    def _clean_data_simple(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Simple data cleaning for feature selection.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Tuple of cleaned (X, y)
        """
        # Handle NaN/Inf in features
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        # Handle NaN/Inf in target
        valid_mask = y.notna() & ~np.isinf(y)
        X_clean = X_clean[valid_mask]
        y_clean = y[valid_mask]
        
        logger.info(f"   Cleaned data: {len(y_clean)} samples, {len(X_clean.columns)} features")
        
        return X_clean, y_clean


def validate_and_clean_data(X: pd.DataFrame, y: pd.Series, logger) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Validate and clean input data to prevent NaN/Inf issues
    
    Args:
        X: Feature matrix
        y: Target values
        logger: Logger instance
        
    Returns:
        Tuple of cleaned (X, y)
    """
    logger.info("🔍 Validating and cleaning input data...")
    
    original_shape = X.shape
    
    # Check for NaN/Inf values in features
    nan_features = X.columns[X.isna().any()].tolist()
    inf_features = X.columns[np.isinf(X.select_dtypes(include=[np.number])).any()].tolist()
    
    if nan_features:
        logger.warning(f"   Found NaN values in {len(nan_features)} features")
        X = X.fillna(X.median())
    
    if inf_features:
        logger.warning(f"   Found Inf values in {len(inf_features)} features")
        # Replace inf with large finite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
    
    # Check for NaN/Inf in target
    if y.isna().any() or np.isinf(y).any():
        logger.warning("   Found NaN/Inf values in target, removing these samples...")
        valid_mask = ~(y.isna() | np.isinf(y))
        X = X[valid_mask]
        y = y[valid_mask]
        logger.info(f"   Kept {len(y)} valid samples after cleaning (removed {original_shape[0] - len(y)} samples)")
    
    # Check for constant features (can cause issues)
    constant_features = X.columns[X.nunique() <= 1].tolist()
    if constant_features:
        logger.warning(f"   Found {len(constant_features)} constant features, removing them...")
        X = X.drop(columns=constant_features)
    
    # Check for features with very high variance (potential outliers)
    feature_vars = X.var()
    high_var_features = feature_vars[feature_vars > feature_vars.quantile(0.99)].index.tolist()
    if high_var_features:
        logger.warning(f"   Found {len(high_var_features)} features with very high variance")
        # Clip extreme values instead of removing features
        for col in X.columns:
            q1, q3 = X[col].quantile([0.01, 0.99])
            X[col] = X[col].clip(q1, q3)
    
    logger.info(f"✅ Data validation completed. Final shape: {X.shape}")
    return X, y


# Extend MLPPredictor with evaluation mixin
class MLPPredictorWithEvaluation(MLPPredictor, MLPEvaluationMixin):
    """
    MLPPredictor with evaluation and feature selection capabilities.
    """
    pass 
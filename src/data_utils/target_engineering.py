"""
Target Engineering Module

This module provides functions for transforming prediction targets,
including the critical Phase 1 fix that converts absolute price targets
to percentage return targets for realistic XGBoost predictions.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


def convert_absolute_to_percentage_returns(combined_data: pd.DataFrame, 
                                            prediction_horizon: int = 10) -> Tuple[pd.DataFrame, str]:
    """
    Convert absolute future price targets to percentage returns
    
    This is the CRITICAL fix for XGBoost unrealistic predictions.
    Instead of predicting absolute prices ($254.75), we predict percentage returns (+5.2%).
    
    Args:
        combined_data: DataFrame with Future_High_XD columns and close prices
        prediction_horizon: Days ahead for prediction
        
    Returns:
        Tuple of (updated_data, new_target_column_name)
    """
    target_column = f"Future_High_{prediction_horizon}D"
    new_target_column = f"Future_Return_{prediction_horizon}D"
    
    logger.info(f"🎯 Converting absolute targets to percentage returns (horizon: {prediction_horizon}d)")
    
    if target_column not in combined_data.columns:
        # Look for any Future_High_XD column as fallback
        future_cols = [col for col in combined_data.columns if col.startswith('Future_High_')]
        if not future_cols:
            raise ValueError(f"No future price target columns found. Expected '{target_column}' or similar.")
        target_column = future_cols[0]
        logger.warning(f"⚠ Using fallback target column: {target_column}")
        new_target_column = target_column.replace('Future_High_', 'Future_Return_')
    
    if 'close' not in combined_data.columns:
        raise ValueError("'close' column not found. Required for percentage return calculation.")
    
    # Create percentage return target
    # Formula: (Future_High - Current_Close) / Current_Close * 100
    future_prices = combined_data[target_column]
    current_prices = combined_data['close']
    
    # Calculate percentage returns (as decimals: 0.05 = 5%, -0.07 = -7%)
    percentage_returns = (future_prices - current_prices) / current_prices
    
    # Add new target column
    combined_data[new_target_column] = percentage_returns
    
    # Log transformation statistics
    valid_returns = percentage_returns.dropna()
    if not valid_returns.empty:
        logger.info("✅ Target transformation completed:")
        logger.info(f"   Original target range: ${current_prices.min():.2f} - ${future_prices.max():.2f}")
        logger.info(f"   New percentage returns: {valid_returns.min():.4f} to {valid_returns.max():.4f} (decimal format)")
        logger.info(f"   Mean return: {valid_returns.mean():.4f} ({valid_returns.mean()*100:.2f}%)")
        logger.info(f"   Std return: {valid_returns.std():.4f} ({valid_returns.std()*100:.2f}%)")
        
        # Sanity check - warn if returns are too extreme (in decimal format)
        extreme_positive = (valid_returns > 0.5).sum()  # >50% return (0.5 in decimal)
        extreme_negative = (valid_returns < -0.5).sum()  # <-50% return (-0.5 in decimal)
        if extreme_positive > 0 or extreme_negative > 0:
            logger.warning(f"⚠ Found extreme returns: {extreme_positive} > +50%, {extreme_negative} < -50%")
    
    return combined_data, new_target_column


def convert_percentage_predictions_to_prices(predictions: np.ndarray, 
                                            current_prices: np.ndarray,
                                            apply_bounds: bool = True,
                                            max_daily_move: float = 10.0) -> np.ndarray:
    """
    Convert percentage return predictions back to absolute prices with bounds checking
    
    This function converts the model's percentage predictions back to price predictions
    while applying realistic bounds to prevent extreme predictions.
    
    Args:
        predictions: Array of percentage return predictions (e.g., [5.2, -2.1, 8.7])
        current_prices: Array of current prices (e.g., [100.0, 50.0, 200.0])
        apply_bounds: Whether to apply realistic bounds
        max_daily_move: Maximum daily move percentage (default: 10%)
        
    Returns:
        Array of predicted prices with bounds applied
    """
    # Convert percentage returns to absolute prices
    # Formula: Future_Price = Current_Price * (1 + Return_Decimal)
    # predictions are already in decimal format (0.05 = 5%)
    predicted_prices = current_prices * (1 + predictions)
    
    if apply_bounds:
        # For 10-day horizon, reasonable bounds might be ±30% (3% per day on average)
        max_10d_move = max_daily_move * np.sqrt(10)  # Scale for 10-day horizon
        
        # Calculate bounds (convert percentage to decimal)
        upper_bound = current_prices * (1 + max_10d_move / 100)
        lower_bound = current_prices * (1 - max_10d_move / 100)
        
        # Apply bounds
        bounded_predictions = np.clip(predicted_prices, lower_bound, upper_bound)
        
        # Log bound applications
        capped_high = (predicted_prices > upper_bound).sum()
        capped_low = (predicted_prices < lower_bound).sum()
        
        if capped_high > 0 or capped_low > 0:
            # logger.info(f"🛡️ Applied prediction bounds: {capped_high} capped high, {capped_low} capped low")
            logger.info(f"   Bounds: ±{max_10d_move:.1f}% for 10-day horizon")
        
        return bounded_predictions
    else:
        return predicted_prices


def create_target_features(data: pd.DataFrame, 
                            target_column: str,
                            lookback_periods: list = [5, 10, 20]) -> pd.DataFrame:
    """
    Create additional features based on target patterns
    
    This can help the model understand target behavior patterns
    and improve prediction accuracy.
    
    Args:
        data: DataFrame with target column
        target_column: Name of the target column
        lookback_periods: List of periods to look back for patterns
        
    Returns:
        DataFrame with additional target-based features
    """
    logger.info(f"🎯 Creating target-based features from '{target_column}'...")
    
    if target_column not in data.columns:
        logger.warning(f"Target column '{target_column}' not found. Skipping target feature creation.")
        return data
    
    enhanced_data = data.copy()
    target_series = enhanced_data[target_column]
    
    # 1. Target volatility features
    for period in lookback_periods:
        if len(target_series) >= period:
            # Rolling standard deviation of target
            vol_col = f"Target_Vol_{period}D"
            enhanced_data[vol_col] = target_series.rolling(window=period, min_periods=1).std()
            
            # Rolling mean of target
            mean_col = f"Target_Mean_{period}D"
            enhanced_data[mean_col] = target_series.rolling(window=period, min_periods=1).mean()
            
            # Target momentum (current vs rolling mean)
            momentum_col = f"Target_Momentum_{period}D"
            enhanced_data[momentum_col] = target_series - enhanced_data[mean_col]
    
    # 2. Target regime features
    if len(target_series) >= 20:
        # High/low target regime
        rolling_median = target_series.rolling(window=20, min_periods=1).median()
        enhanced_data['Target_Above_Median'] = (target_series > rolling_median).astype(int)
        
        # Target percentile rank
        enhanced_data['Target_Percentile'] = target_series.rolling(window=50, min_periods=1).rank(pct=True)
    
    # 3. Target trend features
    if len(target_series) >= 10:
        # Simple trend direction
        enhanced_data['Target_Trend_5D'] = (target_series > target_series.shift(5)).astype(int)
        enhanced_data['Target_Trend_10D'] = (target_series > target_series.shift(10)).astype(int)
    
    new_features_count = len(enhanced_data.columns) - len(data.columns)
    logger.info(f"✅ Created {new_features_count} target-based features")
    
    return enhanced_data 
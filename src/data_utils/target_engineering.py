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


def validate_target_quality(targets: pd.Series, 
                            target_name: str = "target",
                            min_samples: int = 100,
                            max_extreme_pct: float = 5.0) -> dict:
    """
    Validate the quality of target data for ML training
    
    Args:
        targets: Target Series to validate
        target_name: Name of the target for logging
        min_samples: Minimum number of valid samples required
        max_extreme_pct: Maximum percentage of extreme values allowed
        
    Returns:
        Dictionary with validation results
    """
    logger.info(f"🔍 Validating target quality for '{target_name}'...")
    
    validation_results = {
        'is_valid': True,
        'issues': [],
        'stats': {},
        'recommendations': []
    }
    
    # Basic statistics
    valid_targets = targets.dropna()
    total_samples = len(targets)
    valid_samples = len(valid_targets)
    missing_pct = ((total_samples - valid_samples) / total_samples) * 100
    
    validation_results['stats'] = {
        'total_samples': total_samples,
        'valid_samples': valid_samples,
        'missing_samples': total_samples - valid_samples,
        'missing_percentage': missing_pct,
        'mean': valid_targets.mean() if not valid_targets.empty else np.nan,
        'std': valid_targets.std() if not valid_targets.empty else np.nan,
        'min': valid_targets.min() if not valid_targets.empty else np.nan,
        'max': valid_targets.max() if not valid_targets.empty else np.nan,
        'median': valid_targets.median() if not valid_targets.empty else np.nan
    }
    
    # Check 1: Sufficient samples
    if valid_samples < min_samples:
        validation_results['is_valid'] = False
        validation_results['issues'].append(f"Insufficient valid samples: {valid_samples} < {min_samples}")
        validation_results['recommendations'].append("Expand date range or check data availability")
    
    # Check 2: Missing data percentage
    if missing_pct > 20:  # More than 20% missing
        validation_results['issues'].append(f"High missing data: {missing_pct:.1f}%")
        if missing_pct > 50:
            validation_results['is_valid'] = False
        validation_results['recommendations'].append("Investigate data collection issues")
    
    # Check 3: Extreme values (for percentage returns)
    if not valid_targets.empty:
        if 'return' in target_name.lower() or 'pct' in target_name.lower():
            # For percentage returns (in decimal format), check for unrealistic values
            extreme_positive = (valid_targets > 1.0).sum()   # >100% return (1.0 in decimal)
            extreme_negative = (valid_targets < -0.5).sum()  # >50% loss (-0.5 in decimal)
            extreme_total = extreme_positive + extreme_negative
            extreme_pct = (extreme_total / valid_samples) * 100
            
            if extreme_pct > max_extreme_pct:
                validation_results['issues'].append(f"High extreme values: {extreme_pct:.1f}% ({extreme_total} samples)")
                validation_results['recommendations'].append("Consider outlier removal or data cleaning")
                if extreme_pct > 10:  # More than 10% extreme values
                    validation_results['is_valid'] = False
        
        # Check 4: Variance (targets should have some variance)
        if valid_targets.std() < 1e-6:
            validation_results['is_valid'] = False
            validation_results['issues'].append("Target has zero or near-zero variance")
            validation_results['recommendations'].append("Check if target calculation is correct")
    
    # Check 5: Data type validation
    if not pd.api.types.is_numeric_dtype(targets):
        validation_results['is_valid'] = False
        validation_results['issues'].append("Target is not numeric")
        validation_results['recommendations'].append("Convert target to numeric type")
    
    # Log validation results
    if validation_results['is_valid']:
        logger.info(f"✅ Target validation passed for '{target_name}'")
        logger.info(f"   Valid samples: {valid_samples:,}")
        logger.info(f"   Range: [{validation_results['stats']['min']:.2f}, {validation_results['stats']['max']:.2f}]")
        logger.info(f"   Mean±Std: {validation_results['stats']['mean']:.2f}±{validation_results['stats']['std']:.2f}")
    else:
        logger.warning(f"⚠️ Target validation failed for '{target_name}'")
        for issue in validation_results['issues']:
            logger.warning(f"   Issue: {issue}")
        for rec in validation_results['recommendations']:
            logger.info(f"   Recommendation: {rec}")
    
    return validation_results


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
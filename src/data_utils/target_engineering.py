"""
Target Engineering Module

This module provides functions for transforming prediction targets,
including the critical Phase 1 fix that converts absolute price targets
to percentage return targets for realistic predictions.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from src.utils.core.logger import get_logger

logger = get_logger(__name__)


def _downcast_numeric(df: pd.DataFrame) -> None:
    """Downcast numeric columns to reduce memory on large DataFrames."""
    try:
        n_rows = df.shape[0]
        if n_rows <= 250_000:
            return
        num_cols = df.select_dtypes(include=["float64", "int64"]).columns
        for c in num_cols:
            try:
                if pd.api.types.is_float_dtype(df[c].dtype):
                    df[c] = df[c].astype("float32")
                else:
                    df[c] = pd.to_numeric(df[c], downcast="integer")
            except Exception:
                continue
    except Exception:
        return


def convert_absolute_to_percentage_returns(
    combined_data: pd.DataFrame, prediction_horizon: int = 10, filter_outliers: bool = True
) -> Tuple[pd.DataFrame, str]:
    """
    Convert absolute future price targets to percentage returns

    This is the CRITICAL fix for unrealistic predictions.
    Instead of predicting absolute prices ($254.75), we predict percentage returns (+5.2%).

    Args:
        combined_data: DataFrame with Future_Close_XD columns and close prices
        prediction_horizon: Days ahead for prediction

    Returns:
        Tuple of (updated_data, new_target_column_name)
    """
    target_column = f"Future_Close_{prediction_horizon}D"
    new_target_column = f"Future_Return_{prediction_horizon}D"

    logger.info(
        f"Converting absolute targets to percentage returns (horizon: {prediction_horizon}D)"
    )
    # Look for any Future_High_XD column as fallback
    if target_column not in combined_data.columns:
        future_cols = [col for col in combined_data.columns if col.startswith("Future_High_")]
        if not future_cols:
            raise ValueError(
                f"No future price target columns found. Expected '{target_column}' or similar."
            )
        target_column = future_cols[0]
        logger.warning(f"Using fallback target column: {target_column}")
        new_target_column = target_column.replace("Future_High_", "Future_Return_")

    if "close" not in combined_data.columns:
        raise ValueError("'close' column not found. Required for percentage return calculation.")

    # Perform memory-optimized numeric downcast
    _downcast_numeric(combined_data)

    # Vectorized return calculation (faster and cleaner than chunked)
    # returns = (future - close) / close

    try:
        close_prices = combined_data["close"].astype("float32")
        future_prices = combined_data[target_column].astype("float32")

        # Calculate returns with safe division
        returns = (future_prices - close_prices) / close_prices

        # Handle infinite values
        returns = returns.replace([np.inf, -np.inf], np.nan)

        # Assign to DataFrame
        combined_data[new_target_column] = returns

        # Filter outliers and NaNs
        # Outliers: > 100% gain or < -70% loss (sanity bounds)
        if filter_outliers:
            valid_mask = (returns <= 1.0) & (returns >= -0.7) & returns.notna()

            n_total = len(combined_data)
            n_valid = valid_mask.sum()
            n_dropped = n_total - n_valid

            if n_dropped > 0:
                logger.info(
                    f"Filtering {n_dropped} rows ({n_dropped / n_total:.1%}) with invalid/outlier targets"
                )
                combined_data = combined_data[valid_mask].copy()

    except Exception as e:
        logger.error(f"Error calculating returns: {e}")
        raise

    # Log transformation statistics using the newly created column
    valid_returns = combined_data[new_target_column]
    if not valid_returns.empty:
        logger.info("Target transformation completed:")
        logger.info(
            f"New percentage returns: {valid_returns.min():.4f} to {valid_returns.max():.4f} (decimal format)"
        )
        logger.info(f"Mean return: {valid_returns.mean():.4f} ({valid_returns.mean() * 100:.2f}%)")
        logger.info(f"Std return: {valid_returns.std():.4f} ({valid_returns.std() * 100:.2f}%)")

    return combined_data, new_target_column


def convert_percentage_predictions_to_prices(
    predictions: np.ndarray,
    current_prices: np.ndarray,
    apply_bounds: bool = False,
    max_daily_move: float = 10.0,
) -> np.ndarray:
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
    predicted_prices = current_prices * (1 + predictions)

    if apply_bounds:
        # For 10-day horizon, reasonable bounds might be ±30% (3% per day on average)
        max_10d_move = max_daily_move * np.sqrt(10)

        # Calculate bounds (convert percentage to decimal)
        upper_bound = current_prices * (1 + max_10d_move / 100)
        lower_bound = current_prices * (1 - max_10d_move / 100)

        # Apply bounds
        bounded_predictions = np.clip(predicted_prices, lower_bound, upper_bound)

        # Log bound applications
        capped_high = (predicted_prices > upper_bound).sum()
        capped_low = (predicted_prices < lower_bound).sum()

        if capped_high > 0 or capped_low > 0:
            logger.info(f"Bounds: ±{max_10d_move:.1f}% for 10-day horizon")
            logger.info(f"Capped high: {capped_high} times and capped low: {capped_low} times")

        return bounded_predictions
    else:
        return predicted_prices

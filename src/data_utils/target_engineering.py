"""
Target Engineering Module

This module provides functions for transforming prediction targets,
including the critical Phase 1 fix that converts absolute price targets
to percentage return targets for realistic predictions.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _maybe_downcast_numeric(df: pd.DataFrame) -> None:
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


def _compute_returns_in_chunks(
    df: pd.DataFrame, src_col: str, dst_col: str, chunk_size: int = 100_000
) -> pd.DataFrame:
    """Compute percentage returns in row-wise chunks and filter outliers."""
    n_rows = df.shape[0]
    df[dst_col] = pd.Series(np.nan, index=df.index, dtype="float32")
    outlier_mask = np.zeros(n_rows, dtype=bool)
    for start in range(0, n_rows, chunk_size):
        end = min(n_rows, start + chunk_size)
        fut = df[src_col].iloc[start:end].to_numpy(dtype="float32")
        cur = df["close"].iloc[start:end].to_numpy(dtype="float32")
        with np.errstate(divide="ignore", invalid="ignore"):
            # Avoid exact float equality by using a small tolerance for zero
            tol = 1e-12
            returns = np.where(np.abs(cur) > tol, (fut - cur) / cur, np.nan).astype("float32")
        df.iloc[start:end, df.columns.get_loc(dst_col)] = returns
        # Use tolerances for boundary comparisons to avoid strict float equality
        outlier_mask[start:end] = (returns > (1.0 + 1e-9)) | (returns < (-0.7 - 1e-9))
    if outlier_mask.any():
        keep_idx = np.nonzero(~outlier_mask)[0]
        return df.iloc[keep_idx]
    return df


def convert_absolute_to_percentage_returns(
    combined_data: pd.DataFrame, prediction_horizon: int = 10
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
        f"🎯 Converting absolute targets to percentage returns (horizon: {prediction_horizon}D)"
    )
    # Look for any Future_High_XD column as fallback
    if target_column not in combined_data.columns:
        future_cols = [
            col for col in combined_data.columns if col.startswith("Future_High_")
        ]
        if not future_cols:
            raise ValueError(
                f"No future price target columns found. Expected '{target_column}' or similar."
            )
        target_column = future_cols[0]
        logger.warning(f"⚠ Using fallback target column: {target_column}")
        new_target_column = target_column.replace("Future_High_", "Future_Return_")

    if "close" not in combined_data.columns:
        raise ValueError(
            "'close' column not found. Required for percentage return calculation."
        )

    # Perform memory-optimized numeric downcast and chunked return computation
    _maybe_downcast_numeric(combined_data)
    combined_data = _compute_returns_in_chunks(
        combined_data, target_column, new_target_column, chunk_size=100_000
    )

    # Log transformation statistics using the newly created column
    valid_returns = combined_data[new_target_column].dropna()
    if not valid_returns.empty:
        logger.info("✅ Target transformation completed:")
        # Use safe aggregations on the DataFrame to avoid creating large temporaries
        orig_min = combined_data["close"].min()
        fut_max = combined_data[target_column].max()
        logger.info(f"   Original target range: ${orig_min:.2f} - ${fut_max:.2f}")
        logger.info(
            f"   New percentage returns: {valid_returns.min():.4f} to {valid_returns.max():.4f} (decimal format)"
        )
        logger.info(
            f"   Mean return: {valid_returns.mean():.4f} ({valid_returns.mean() * 100:.2f}%)"
        )
        logger.info(
            f"   Std return: {valid_returns.std():.4f} ({valid_returns.std() * 100:.2f}%)"
        )

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
    # predictions are already in decimal format (0.05 = 5%)
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
            logger.info(f"   Bounds: ±{max_10d_move:.1f}% for 10-day horizon")
            logger.info(
                f"   Capped high: {capped_high} times and capped low: {capped_low} times"
            )

        return bounded_predictions
    else:
        return predicted_prices

"""
Stationarity Transformer Module

This module provides functions to transform time-series data to be stationary
and to reverse the transformation.
"""

import pandas as pd
from statsmodels.tsa.stattools import kpss
from typing import Tuple, Optional, Dict, List
from joblib import Parallel, delayed
import os

from src.utils.logger import get_logger

logger = get_logger(__name__)

def _kpss_test(series: pd.Series, downsample_window: Optional[int] = None) -> Tuple[float, float]:
    """Helper function to run KPSS test and handle errors, with optional downsampling.
    Returns: (p_value, kpss_statistic). Null hypothesis: series is stationary."""
    series_clean = series.dropna()
    if series_clean.empty or len(series_clean) < 10:
        logger.warning(f"Series '{series.name}' is empty or too short after dropping NaNs (len={len(series_clean)}). Skipping KPSS.")
        return 0.0, float('inf')
    try:
        # Remove downsampling logic: always use full series
        # if downsample_window is not None and len(series_clean) > downsample_window:
        #     series_clean = series_clean.iloc[-downsample_window:]
        kpss_result = kpss(series_clean, nlags='auto')
        kpss_statistic, p_value = kpss_result[0], kpss_result[1]
        return p_value, kpss_statistic
    except Exception as e:
        logger.error(f"KPSS test failed for series '{series.name}': {e}")
        return 0.0, float('inf')


def transform_dataframe_to_stationary(df: pd.DataFrame, n_jobs: int = None, verbose: bool = True, nan_handling: str = 'none', downsample_window: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Transforms multiple columns in a DataFrame to be stationary using KPSS test and controlled parallel processing.
    Args:
        df: DataFrame with time-series data.
        n_jobs: Number of parallel jobs to run (default: auto-detect up to 4).
        verbose: If True, set logger to INFO, else WARNING.
        nan_handling: How to handle NaNs after transformation ('none', 'drop', 'bfill', 'ffill').
        downsample_window: If set, only the most recent N rows are used for KPSS tests (default: 50000).
    Returns:
        df_transformed: The transformed DataFrame.
        transformation_manifest: Dict of column: transformation.
        failed_columns: List of columns that could not be made stationary.
    """
    if n_jobs is None:
        n_jobs = min(4, os.cpu_count() or 2)
    if verbose:
        logger.setLevel('INFO')
    else:
        logger.setLevel('WARNING')
    logger.info(f"🚀 [START] Parallel stationarity transformation for {len(df.columns)} columns using {n_jobs} cores (KPSS)...")
    logger.info("Step 1: Running initial KPSS tests in parallel...")
    initial_results = Parallel(n_jobs=n_jobs)(
        delayed(_kpss_test)(df[col], downsample_window) for col in df.columns
    )
    stationary_cols = set()
    non_stationary_cols = []
    transformation_manifest = {}
    for col, (p_value, _) in zip(df.columns, initial_results):
        # KPSS: Null hypothesis is stationary, so p >= 0.05 means stationary
        if p_value >= 0.05:
            stationary_cols.add(col)
            transformation_manifest[col] = "none"
            logger.debug(f"   Column '{col}' is already stationary (KPSS p={p_value:.4f})")
        else:
            non_stationary_cols.append(col)
    logger.info(f"✅ {len(stationary_cols)} columns are already stationary (KPSS).")
    logger.info(f"🔄 {len(non_stationary_cols)} columns require transformation.")
    if not non_stationary_cols:
        logger.info("[END] All columns stationary. No transformation needed.")
        return df, transformation_manifest, []
    logger.info("Step 2: Applying pct_change and diff transformations to non-stationary columns...")
    pct_change_transformed = df[non_stationary_cols].pct_change()
    diff_transformed = df[non_stationary_cols].diff()
    logger.info("Step 3: Running KPSS tests on pct_change columns in parallel...")
    pct_change_results = Parallel(n_jobs=n_jobs)(
        delayed(_kpss_test)(pct_change_transformed[col], downsample_window) for col in non_stationary_cols
    )
    logger.info("Step 4: Running KPSS tests on diff columns in parallel...")
    diff_results = Parallel(n_jobs=n_jobs)(
        delayed(_kpss_test)(diff_transformed[col], downsample_window) for col in non_stationary_cols
    )
    logger.info("Step 5: Selecting best transformation for each non-stationary column...")
    failed_columns = []
    for i, col in enumerate(non_stationary_cols):
        p_pct, kpss_pct = pct_change_results[i]
        p_diff, kpss_diff = diff_results[i]
        best_transform = None
        min_kpss = float('inf')
        # KPSS: p >= 0.05 means stationary
        if p_pct >= 0.05 and kpss_pct < min_kpss:
            min_kpss = kpss_pct
            best_transform = 'percentage_change'
        if p_diff >= 0.05 and kpss_diff < min_kpss:
            min_kpss = kpss_diff
            best_transform = 'first_difference'
        if best_transform:
            transformation_manifest[col] = best_transform
            if best_transform == 'percentage_change':
                df[col] = pct_change_transformed[col]
            else:
                df[col] = diff_transformed[col]
            logger.info(f"   Column '{col}': applied '{best_transform}' (KPSS p_pct={p_pct:.4f}, p_diff={p_diff:.4f})")
        else:
            logger.warning(f"   Could not make column '{col}' stationary (KPSS p_pct={p_pct:.4f}, p_diff={p_diff:.4f}). Leaving as is.")
            transformation_manifest[col] = 'failed'
            failed_columns.append(col)
    # NaN handling
    if nan_handling == 'drop':
        df.dropna(inplace=True)
        logger.info("NaN handling: dropped all rows with NaNs after transformation.")
    elif nan_handling == 'bfill':
        df.fillna(method='bfill', inplace=True)
        logger.info("NaN handling: backfilled NaNs after transformation.")
    elif nan_handling == 'ffill':
        df.fillna(method='ffill', inplace=True)
        logger.info("NaN handling: forward filled NaNs after transformation.")
    else:
        logger.info("NaN handling: left NaNs as-is after transformation.")
    logger.info("✅ [END] Parallel stationarity transformation complete (KPSS).")
    return df, transformation_manifest, failed_columns


def transform_to_stationary(series: pd.Series, verbose: bool = True) -> Tuple[pd.Series, Optional[str]]:
    """
    Transforms a time series to be stationary using KPSS test.
    It tries different transformations (percentage change, first difference)
    and uses the KPSS test to check for stationarity.
    Args:
        series: The time series to transform.
        verbose: If True, set logger to INFO, else WARNING.
    Returns:
        A tuple containing:
        - The transformed, stationary series.
        - The name of the transformation applied, or 'none' if the series is already stationary.
        If no transformation makes it stationary, it returns the original series and None.
    """
    if verbose:
        logger.setLevel('INFO')
    else:
        logger.setLevel('WARNING')
    series_clean = series.dropna()
    if series_clean.empty or len(series_clean) < 10:
        logger.warning(f"Series '{series.name}' is empty or too short after dropping NaNs (len={len(series_clean)}). Skipping KPSS.")
        return series, None
    try:
        kpss_stat, p_value = kpss(series_clean, nlags='auto')[:2]
    except Exception as e:
        logger.error(f"KPSS test failed for series '{series.name}': {e}")
        return series, None
    # KPSS: Null hypothesis is stationary
    if p_value >= 0.05:
        logger.debug(f"Series '{series.name}' is already stationary (KPSS).")
        return series, "none"
    transformations = {
        'percentage_change': series.pct_change(),
        'first_difference': series.diff(),
    }
    best_transformation = None
    min_kpss_statistic = float('inf')
    for name, transformed_series in transformations.items():
        transformed_series_clean = transformed_series.dropna()
        if transformed_series_clean.empty or len(transformed_series_clean) < 10:
            logger.warning(f"Transformed series '{series.name}' ({name}) is empty or too short after dropping NaNs (len={len(transformed_series_clean)}). Skipping KPSS.")
            continue
        try:
            kpss_stat, p_value = kpss(transformed_series_clean, nlags='auto')[:2]
        except Exception as e:
            logger.error(f"KPSS test failed for transformation '{name}' on series '{series.name}': {e}")
            continue
        if p_value >= 0.05 and kpss_stat < min_kpss_statistic:
            min_kpss_statistic = kpss_stat
            best_transformation = (name, transformed_series)
    if best_transformation:
        name, series_to_return = best_transformation
        logger.info(f"Best stationarity transformation for '{series.name}': {name} (KPSS)")
        return series_to_return, name
    logger.warning(f"Could not find a transformation to make series '{series.name}' stationary (KPSS).")
    return series, None


def inverse_transform_dataframe(
    transformed_df: pd.DataFrame,
    original_df: pd.DataFrame,
    transformation_manifest: Dict[str, str]
) -> pd.DataFrame:
    """
    Reverts multiple transformed columns in a DataFrame back to their original scale.
    """
    df_inversed = transformed_df.copy()
    for col, transform_name in transformation_manifest.items():
        if transform_name not in ["none", "failed", None]:
            df_inversed[col] = inverse_transform_series(
                transformed_df[col],
                original_df[col],
                transform_name
            )
    return df_inversed


def inverse_transform_series(
    transformed_series: pd.Series,
    original_series: pd.Series,
    transform_name: Optional[str]
) -> pd.Series:
    """
    Reverts a stationary series back to its original scale.

    Args:
        transformed_series: The series that was made stationary.
        original_series: The original series, used as a reference for reversing the transformation.
        transform_name: The name of the transformation that was applied.

    Returns:
        The series in its original scale.
    """
    if transform_name is None or transform_name == "none":
        return transformed_series

    first_valid_index = transformed_series.first_valid_index()
    if first_valid_index is None:
        return transformed_series # Return empty if all are NaN

    # Get the location of the first valid value in the transformed series
    # and find the value in the original series at the PREVIOUS location.
    prev_loc = original_series.index.get_loc(first_valid_index) - 1
    if prev_loc < 0:
        logger.error(f"Cannot inverse transform '{transform_name}' for series '{original_series.name}'. No previous value available.")
        # Return the series as-is if we can't inverse it
        return transformed_series
        
    last_known_value = original_series.iloc[prev_loc]
    
    # We only need to inverse transform the non-NaN part of the series
    valid_transformed_series = transformed_series.loc[first_valid_index:]
    
    if transform_name == 'first_difference':
        inversed_values = last_known_value + valid_transformed_series.cumsum()
    elif transform_name == 'percentage_change':
        inversed_values = last_known_value * (1 + valid_transformed_series).cumprod()
    else:
        raise ValueError(f"Unknown transformation: '{transform_name}'")

    return inversed_values 
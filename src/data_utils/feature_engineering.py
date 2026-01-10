"""
Feature Engineering Module

This module provides functions for transforming and enhancing features,
including the Phase 2 fixes that add price-normalized features and
prediction bounds features for better performance.
"""

import pandas as pd
import numpy as np
import time
from sklearn.feature_selection import mutual_info_regression
from src.utils.core.logger import get_logger
from src.data_utils.safe_math import SafeDivideSpec, compute_bb_position

logger = get_logger(__name__)


def fast_correlation_based_filter(
    x: pd.DataFrame, y: pd.Series = None, threshold: float = 0.9
) -> list:
    """
    Fast Correlation-Based Filter algorithm for feature selection.

    This implements an optimized version of correlation-based feature selection that:
    1. First filters features based on relevance to target (if y provided)
    2. Then removes redundant features using correlation analysis
    3. Much faster than computing full correlation matrix for large feature sets

    Args:
        x: Feature matrix
        y: Target values (optional, improves filtering if provided)
        threshold: Correlation threshold for redundancy removal

    Returns:
        List of feature names to keep
    """
    # Early returns for edge cases
    if x.empty or len(x.columns) <= 1:
        return x.columns.tolist()

    # Filter based on relevance to target if available
    x_filtered = _filter_relevant_features_optimized(x, y)

    # Remove redundant features based on correlation
    return _remove_correlated_features_optimized(x, x_filtered, threshold)


def _filter_relevant_features_optimized(x: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
    """Optimized version of relevance filtering using sampling for performance."""
    if y is None or y.empty:
        return x

    try:
        # PERFORMANCE OPTIMIZATION: Use sample for mutual information calculation
        # For large datasets (>10k samples), sample 10k rows to speed up MI calculation
        sample_size = min(len(x), 10000)  # Cap at 10k samples
        if len(x) > sample_size:
            logger.debug(
                f"Sampling {sample_size} rows from {len(x)} for mutual information calculation"
            )
            # Use random sampling but maintain alignment with y
            sample_indices = np.random.choice(len(x), size=sample_size, replace=False)
            x_sample = x.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]
        else:
            x_sample = x
            y_sample = y

        # PERFORMANCE OPTIMIZATION: Limit features processed to prevent excessive computation
        max_features_for_mi = 100  # Cap at 100 features for MI calculation
        if len(x_sample.columns) > max_features_for_mi:
            logger.debug(
                f"Limiting mutual information calculation to top {max_features_for_mi} features by variance"
            )
            # Pre-select top features by variance to reduce MI computation time
            variances = x_sample.var()
            top_features = variances.nlargest(max_features_for_mi).index
            x_sample = x_sample[top_features]

        # Use mutual information to identify relevant features
        with np.errstate(invalid="ignore"):  # Handle NaN/inf values
            mi_scores = mutual_info_regression(x_sample, y_sample)

        # Keep only features with some predictive power (top 80% by MI score)
        if len(mi_scores) > 5:  # Only filter if we have enough features
            relevance_threshold = np.percentile(mi_scores, 20)  # Bottom 20% cutoff
            relevant_mask = mi_scores >= relevance_threshold

            # Map back to original feature names (important fix!)
            relevant_features = x_sample.columns[relevant_mask]
            x_filtered = x[relevant_features]
            logger.debug(
                f"Filtered to {x_filtered.shape[1]} relevant features based on mutual information"
            )
            return x_filtered

    except Exception as e:
        logger.warning(
            f"Mutual information calculation failed, proceeding without relevance filtering: {e}"
        )

    return x


def _remove_correlated_features_optimized(
    original_x: pd.DataFrame, x_filtered: pd.DataFrame, threshold: float
) -> list:
    """Optimized version of correlation filtering with chunking and limits."""
    if x_filtered.shape[1] <= 1:
        return x_filtered.columns.tolist()

    try:
        # PERFORMANCE OPTIMIZATION: Limit features to prevent excessive correlation computation
        max_features_for_corr = 150  # Cap at 150 features for correlation analysis
        if x_filtered.shape[1] > max_features_for_corr:
            logger.debug(
                f"Limiting correlation analysis to top {max_features_for_corr} features by variance"
            )
            variances = x_filtered.var()
            top_features = variances.nlargest(max_features_for_corr).index
            x_filtered = x_filtered[top_features]

        # PERFORMANCE OPTIMIZATION: Use chunked correlation computation for large feature sets
        num_features = x_filtered.shape[1]
        if num_features > 50:
            logger.debug(f"Using chunked correlation computation for {num_features} features")
            # Process in chunks to avoid memory issues
            corr_matrix = _compute_correlation_matrix_chunked(x_filtered, chunk_size=50)
        else:
            # Compute correlation matrix only for potentially relevant features
            corr_matrix = x_filtered.corr().abs()

        # Use triangular matrix to avoid redundant comparisons
        upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))

        # PERFORMANCE OPTIMIZATION: Vectorized approach to find correlated features
        to_drop = set()

        # Find columns that have high correlations
        for column in upper.columns:
            # Get correlations above threshold
            correlated_mask = upper[column] > threshold
            if correlated_mask.any():
                correlated_features = upper.index[correlated_mask].tolist()

                if correlated_features:
                    # Among correlated features, keep the one with highest variance
                    correlated_group = correlated_features + [column]
                    variances = x_filtered[correlated_group].var()

                    # Keep the feature with maximum variance, drop others
                    keep_feature = variances.idxmax()
                    to_drop.update([f for f in correlated_group if f != keep_feature])

        # Return features to keep
        features_to_keep = [col for col in original_x.columns if col not in to_drop]
        logger.debug(f"Removed {len(to_drop)} redundant features via correlation filtering")

        return features_to_keep

    except Exception as e:
        logger.warning(f"Correlation filtering failed, returning all features: {e}")
        return original_x.columns.tolist()


def _compute_correlation_matrix_chunked(x: pd.DataFrame, chunk_size: int = 50) -> pd.DataFrame:
    """Compute correlation matrix in chunks to handle large feature sets efficiently."""
    features = x.columns.tolist()
    num_features = len(features)

    # Initialize empty correlation matrix
    corr_matrix = pd.DataFrame(index=features, columns=features, dtype=np.float64)

    # Fill diagonal with 1.0
    np.fill_diagonal(corr_matrix.values, 1.0)

    # Compute correlations in chunks
    for i in range(num_features):
        for j in range(i + 1, num_features):
            if j - i <= chunk_size:  # Only compute within chunk size
                # Compute correlation between feature i and j
                corr_val = x[features[i]].corr(x[features[j]])
                corr_matrix.iloc[i, j] = abs(corr_val)
                corr_matrix.iloc[j, i] = abs(corr_val)

    return corr_matrix


def add_price_normalized_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price-normalized features to convert absolute price features to ratios

    This addresses the core issue where absolute price features (SMA_20 = $150)
    need to be converted to relative features (close/SMA_20 = 0.95).

    Args:
        features_df: DataFrame with price-based features

    Returns:
        DataFrame with additional normalized features
    """

    if "close" not in features_df.columns:
        logger.warning("'close' column not found. Skipping price normalization.")
        return features_df

    features_enhanced = features_df.copy()
    current_price = features_enhanced["close"]

    # 1. SMA Ratios - Convert absolute SMA values to relative positions
    sma_cols = [col for col in features_enhanced.columns if col.startswith("SMA_")]
    for sma_col in sma_cols:
        if sma_col in features_enhanced.columns:
            ratio_col = f"{sma_col}_Ratio"
            features_enhanced[ratio_col] = current_price / features_enhanced[sma_col]

    # 2. Bollinger Bands Ratios
    if "BB_Lower" in features_enhanced.columns:
        # Price position within BB bands
        if "BB_Upper" in features_enhanced.columns or any(
            "BB" in col for col in features_enhanced.columns
        ):
            # Calculate BB_Upper from BB_Lower and BB_Width if available
            if (
                "BB_Width" in features_enhanced.columns
                and "BB_Upper" not in features_enhanced.columns
            ):
                bb_middle = features_enhanced["BB_Lower"] + (features_enhanced["BB_Width"] / 2)
                features_enhanced["BB_Upper"] = bb_middle + (features_enhanced["BB_Width"] / 2)

            if "BB_Upper" in features_enhanced.columns:
                # Price position within bands (0 = at lower band, 1 = at upper band) using safe division
                spec = SafeDivideSpec(denom_epsilon=0.0, fill_value=None, sanitize_infinite=True)
                features_enhanced["BB_Position"] = compute_bb_position(
                    current_price,
                    features_enhanced["BB_Lower"],
                    features_enhanced["BB_Upper"],
                    spec,
                )
    # 3. ATR-normalized features
    if "ATR" in features_enhanced.columns:
        # Price volatility relative to ATR
        features_enhanced["Price_ATR_Ratio"] = current_price / features_enhanced["ATR"]

    # 4. Volume-Price Efficiency
    if "volume" in features_enhanced.columns:
        # Price change per unit volume
        if "Return_1D" in features_enhanced.columns:
            features_enhanced["Return_Volume_Efficiency"] = features_enhanced["Return_1D"].abs() / (
                features_enhanced["volume"] + 1e-8
            )

    # 5. Ichimoku Ratios
    ichimoku_cols = [
        col
        for col in features_enhanced.columns
        if col.startswith("Ichimoku_")
        and not col.endswith(
            ("_Above_Kijun", "_Above_Cloud", "_Below_Cloud", "_Green", "_Thickness")
        )
    ]
    for ich_col in ichimoku_cols:
        if ich_col in features_enhanced.columns:
            ratio_col = f"{ich_col}_Ratio"
            features_enhanced[ratio_col] = current_price / features_enhanced[ich_col]

    # 6. Price momentum ratios
    if "open" in features_enhanced.columns:
        features_enhanced["Close_Open_Ratio"] = current_price / features_enhanced["open"]

    return features_enhanced


def add_prediction_bounds_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that help the model understand realistic prediction bounds

    This helps prevent the model from making extreme predictions by providing
    context about typical price movement ranges.

    Args:
        features_df: DataFrame with features

    Returns:
        DataFrame with additional bound-related features
    """

    features_enhanced = features_df.copy()

    # 1. Historical volatility context
    if "ATR_Percent" in features_enhanced.columns:
        features_enhanced["Expected_Daily_Move"] = features_enhanced["ATR_Percent"]
        features_enhanced["Expected_10D_Move"] = features_enhanced["ATR_Percent"] * np.sqrt(10)

    # 3. Current volatility regime helps set expectation bounds
    if (
        "Vol_Regime_High" in features_enhanced.columns
        and "Vol_Regime_Low" in features_enhanced.columns
    ):
        features_enhanced["Vol_Regime_Context"] = (
            features_enhanced["Vol_Regime_High"] * 2 + features_enhanced["Vol_Regime_Low"] * 0.5
        )

    # 4. RSI mean reversion context
    if "RSI_14" in features_enhanced.columns:
        # RSI distance from 50 (neutral) indicates mean reversion pressure
        features_enhanced["RSI_Mean_Reversion_Pressure"] = (
            abs(features_enhanced["RSI_14"] - 50) / 50
        )

    return features_enhanced


def clean_data_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data specifically for training/prediction

    This function implements the critical data cleaning transformations
    that were proven to fix the broken model.

    Args:
        df: Input DataFrame with mixed data types

    Returns:
        Cleaned DataFrame ready for training
    """
    logger.info(f"Starting data cleaning on {len(df)} samples...")

    # 1. Handle infinite and extremely large values
    df_clean = df.copy()

    # Replace infinite values with NaN
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Handle extremely large values (beyond float32 range)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Cap values at reasonable limits
        max_val = np.finfo(np.float32).max / 10  # Conservative limit
        min_val = np.finfo(np.float32).min / 10

        # Count extreme values before capping
        extreme_count = ((df_clean[col] > max_val) | (df_clean[col] < min_val)).sum()
        if extreme_count > 0:
            logger.info(f"Capping {extreme_count} extreme values in {col}")
            df_clean[col] = df_clean[col].clip(lower=min_val, upper=max_val)

    # 3. Keep numeric columns as float32 to reduce memory usage (float64 not required)

    df_clean[numeric_cols] = (
        df_clean[numeric_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    )

    logger.info(f"Training data cleaning completed: {len(df_clean)} samples ready")

    return df_clean


def analyze_feature_diversity(df: pd.DataFrame, min_variance_threshold: float = 1e-8) -> dict:
    """
    Analyze feature diversity to identify potential model training issues

    Args:
        df: DataFrame to analyze
        min_variance_threshold: Minimum variance threshold for useful features

    Returns:
        Dictionary with diversity analysis results
    """
    logger.info("Analyzing feature diversity...")

    # Calculate variances only for numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    variances = numeric_df.var() if not numeric_df.empty else pd.Series(dtype=float)

    # Categorize features by variance
    zero_variance = variances[variances == 0].index.tolist()
    low_variance = variances[(variances > 0) & (variances < min_variance_threshold)].index.tolist()
    high_variance = variances[variances >= min_variance_threshold].sort_values(ascending=False)

    # Check for constant features (only numeric columns for variance analysis)
    constant_features = []
    for col in numeric_df.columns:
        if numeric_df[col].nunique() <= 1:
            constant_features.append(col)

    analysis = {
        "total_features": len(df.columns),
        "numeric_features": len(numeric_df.columns),
        "zero_variance_count": len(zero_variance),
        "low_variance_count": len(low_variance),
        "constant_feature_count": len(constant_features),
        "useful_feature_count": len(high_variance),
        "zero_variance_features": zero_variance,
        "low_variance_features": low_variance,
        "constant_features": constant_features,
        "high_variance_features": high_variance.head(20).to_dict(),
        "feature_variances": variances.describe().to_dict() if not variances.empty else {},
    }

    logger.info("Feature Diversity Analysis:")
    logger.info(f"Total features: {analysis['total_features']}")
    logger.info(f"Numeric features: {analysis['numeric_features']}")
    logger.info(f"Zero variance: {analysis['zero_variance_count']}")
    logger.info(f"Low variance: {analysis['low_variance_count']}")
    logger.info(f"Constant features: {analysis['constant_feature_count']}")
    logger.info(f"Useful features: {analysis['useful_feature_count']}")

    return analysis


def clean_features_for_training(
    X: pd.DataFrame,
    y: pd.Series,
    remove_constants: bool = True,
    remove_zero_variance: bool = True,
    remove_high_correlation: bool = True,
    correlation_threshold: float = 0.99,
) -> tuple:
    """
    Clean features specifically for model training

    Args:
        X: Feature matrix
        y: Target values
        remove_constants: Remove constant features
        remove_zero_variance: Remove zero variance features
        remove_high_correlation: Remove highly correlated features
        correlation_threshold: Correlation threshold for removal

    Returns:
        Tuple of (cleaned_X, cleaned_y, removed_features_info)
    """
    logger.info(f"Cleaning features for training: {X.shape[0]} samples, {X.shape[1]} features")

    x_clean = X.copy()
    y_clean = y.copy()
    removed_features = {
        "constant": [],
        "zero_variance": [],
        "high_correlation": [],
        "non_numeric": [],
    }

    # 1. Remove non-numeric columns but preserve essential price columns
    essential_columns = ["close", "ticker_id", "date_int"]

    non_numeric_cols = x_clean.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numeric_to_remove = [col for col in non_numeric_cols if col not in essential_columns]

    if non_numeric_to_remove:
        logger.info(
            f"Removing {len(non_numeric_to_remove)} non-numeric columns: {non_numeric_to_remove}"
        )
        removed_features["non_numeric"] = non_numeric_to_remove
        x_clean = x_clean.drop(columns=non_numeric_to_remove)

    # Ensure essential numeric columns are properly typed
    logger.info("Converting essential columns to numeric...")
    cols = [c for c in essential_columns if c in x_clean.columns and x_clean[c].dtype == "object"]
    if cols:
        # convert the block at once and assign in a single operation
        x_clean.loc[:, cols] = x_clean[cols].apply(pd.to_numeric, errors="coerce")
        logger.info(f"Converted {len(cols)} essential columns to numeric: {cols}")

    # 2-4. Remove constant / zero-variance / highly correlated features (optimized)
    if remove_constants or remove_zero_variance or remove_high_correlation:
        # Cache numeric selection and prepare fast membership checks
        numeric_x = x_clean.select_dtypes(include=[np.number])
        essential_set = set(essential_columns)

    # 2. Remove constant features (vectorized nunique)
    if remove_constants and not numeric_x.empty:
        logger.info(f"Removing constant features: {remove_constants}")
        nunique_series = numeric_x.nunique()
        constant_features = [
            col for col in nunique_series[nunique_series <= 1].index if col not in essential_set
        ]
        if constant_features:
            logger.info(
                f"Removing {len(constant_features)} constant features (preserving essential columns)"
            )
            removed_features["constant"] = constant_features
            x_clean = x_clean.drop(columns=constant_features)
            # refresh numeric view after drop
            numeric_x = x_clean.select_dtypes(include=[np.number])

    # 3. Remove zero variance features (vectorized variance check)
    if remove_zero_variance and not numeric_x.empty:
        logger.info(f"Removing zero variance features: {remove_zero_variance}")
        # Use population variance (ddof=0) and isclose to handle floating precision
        variances = numeric_x.var(axis=0, ddof=0)
        zero_var_idx = variances[np.isclose(variances, 0.0)].index
        zero_var_features = [col for col in zero_var_idx if col not in essential_set]
        if zero_var_features:
            logger.info(
                f"Removing {len(zero_var_features)} zero variance features (preserving essential columns)"
            )
            removed_features["zero_variance"] = zero_var_features
            x_clean = x_clean.drop(columns=zero_var_features)
            numeric_x = x_clean.select_dtypes(include=[np.number])

    # 4. Remove highly correlated features (> correlation_threshold)
    if remove_high_correlation and not numeric_x.empty and len(numeric_x.columns) > 1:
        logger.info(f"Removing highly correlated features: {remove_high_correlation}")
        # Use fast correlation-based filter algorithm with timing
        start_time = time.time()
        logger.info(
            f"Starting correlation filtering on {numeric_x.shape[1]} features, {numeric_x.shape[0]} samples"
        )
        features_to_keep = fast_correlation_based_filter(numeric_x, y_clean, correlation_threshold)
        elapsed_time = time.time() - start_time
        logger.info(f"Correlation filtering completed in {elapsed_time:.2f} seconds")

        # Determine which features were removed
        current_features = set(numeric_x.columns)
        kept_features = set(features_to_keep)
        removed_corr_features = current_features - kept_features

        # Filter out essential features from removal list
        high_corr_features = [col for col in removed_corr_features if col not in essential_set]

        if high_corr_features:
            logger.info(
                f"Removing {len(high_corr_features)} highly correlated features (>{correlation_threshold}, preserving essential columns)"
            )
            removed_features["high_correlation"] = high_corr_features
            x_clean = x_clean.drop(columns=high_corr_features)

    # 5. Remove rows with remaining NaN values in target
    nan_mask = y_clean.notna()
    if not nan_mask.all():
        nan_count = (~nan_mask).sum()
        logger.info(f"Removing {nan_count} samples with NaN targets")
        x_clean = x_clean[nan_mask]
        y_clean = y_clean[nan_mask]

    # 6. Ensure X and y have same length
    min_length = min(len(x_clean), len(y_clean))
    x_clean = x_clean.iloc[:min_length]
    y_clean = y_clean.iloc[:min_length]

    logger.info(
        f"Feature cleaning completed: {x_clean.shape[0]} samples, {x_clean.shape[1]} features"
    )
    logger.info(f"Removed: {sum(len(v) for v in removed_features.values())} features total")

    return x_clean, y_clean, removed_features


def add_date_features(features_df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """
    Add temporal features based on date information

    Args:
        features_df: DataFrame with date column
        date_column: Name of the date column

    Returns:
        DataFrame with additional temporal features
    """

    if date_column not in features_df.columns:
        logger.warning(f"Date column '{date_column}' not found. Skipping temporal features.")
        return features_df

    features_enhanced = features_df.copy()
    date_col = features_enhanced[date_column].copy()

    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(date_col):
        date_col = pd.to_datetime(date_col)

    # Create date-based features
    features_enhanced["date_int"] = (date_col - pd.Timestamp("2020-01-01")).dt.days
    features_enhanced["year"] = date_col.dt.year
    features_enhanced["month"] = date_col.dt.month
    features_enhanced["day_of_year"] = date_col.dt.dayofyear
    features_enhanced["quarter"] = date_col.dt.quarter
    features_enhanced["day_of_week"] = date_col.dt.dayofweek
    features_enhanced["is_month_end"] = date_col.dt.is_month_end.astype(int)
    features_enhanced["is_quarter_end"] = date_col.dt.is_quarter_end.astype(int)

    return features_enhanced

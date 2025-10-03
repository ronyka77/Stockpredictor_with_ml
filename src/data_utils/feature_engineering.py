"""
Feature Engineering Module

This module provides functions for transforming and enhancing features,
including the Phase 2 fixes that add price-normalized features and
prediction bounds features for better performance.
"""

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
        logger.warning("⚠ 'close' column not found. Skipping price normalization.")
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
                bb_middle = features_enhanced["BB_Lower"] + (
                    features_enhanced["BB_Width"] / 2
                )
                features_enhanced["BB_Upper"] = bb_middle + (
                    features_enhanced["BB_Width"] / 2
                )

            if "BB_Upper" in features_enhanced.columns:
                # Price position within bands (0 = at lower band, 1 = at upper band)
                features_enhanced["BB_Position"] = (
                    current_price - features_enhanced["BB_Lower"]
                ) / (features_enhanced["BB_Upper"] - features_enhanced["BB_Lower"])

    # 3. ATR-normalized features
    if "ATR" in features_enhanced.columns:
        # Price volatility relative to ATR
        features_enhanced["Price_ATR_Ratio"] = current_price / features_enhanced["ATR"]

    # 4. Volume-Price Efficiency
    if "volume" in features_enhanced.columns:
        # Price change per unit volume
        if "Return_1D" in features_enhanced.columns:
            features_enhanced["Return_Volume_Efficiency"] = features_enhanced[
                "Return_1D"
            ].abs() / (features_enhanced["volume"] + 1e-8)

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
        features_enhanced["Close_Open_Ratio"] = (
            current_price / features_enhanced["open"]
        )

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
        features_enhanced["Expected_10D_Move"] = features_enhanced[
            "ATR_Percent"
        ] * np.sqrt(10)

    # 3. Current volatility regime helps set expectation bounds
    if (
        "Vol_Regime_High" in features_enhanced.columns
        and "Vol_Regime_Low" in features_enhanced.columns
    ):
        features_enhanced["Vol_Regime_Context"] = (
            features_enhanced["Vol_Regime_High"] * 2
            + features_enhanced["Vol_Regime_Low"] * 0.5
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
    logger.info(f"🧹 Starting data cleaning on {len(df)} samples...")

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
            logger.info(f"   Capping {extreme_count} extreme values in {col}")
            df_clean[col] = df_clean[col].clip(lower=min_val, upper=max_val)

    # 2. Fill NaN values with median (more robust than mean)
    nan_counts = df_clean[numeric_cols].isnull().sum()
    if nan_counts.sum() > 0:
        logger.info(f"   Filling {nan_counts.sum()} NaN values with column medians")
        for col in numeric_cols:
            if nan_counts[col] > 0:
                median_val = df_clean[col].median()
                if pd.isna(median_val):  # All values are NaN
                    df_clean[col].fillna(0.0, inplace=True)
                else:
                    df_clean[col].fillna(median_val, inplace=True)

    # 3. Ensure all numeric columns are float64 for consistency
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").astype(np.float64)

    logger.info(f"✅ Training data cleaning completed: {len(df_clean)} samples ready")

    return df_clean


def analyze_feature_diversity(
    df: pd.DataFrame, min_variance_threshold: float = 1e-8
) -> dict:
    """
    Analyze feature diversity to identify potential model training issues

    Args:
        df: DataFrame to analyze
        min_variance_threshold: Minimum variance threshold for useful features

    Returns:
        Dictionary with diversity analysis results
    """
    logger.info("📊 Analyzing feature diversity...")

    # Calculate variances only for numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    variances = numeric_df.var() if not numeric_df.empty else pd.Series(dtype=float)

    # Categorize features by variance
    zero_variance = variances[variances == 0].index.tolist()
    low_variance = variances[
        (variances > 0) & (variances < min_variance_threshold)
    ].index.tolist()
    high_variance = variances[variances >= min_variance_threshold].sort_values(
        ascending=False
    )

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
        "feature_variances": variances.describe().to_dict()
        if not variances.empty
        else {},
    }

    logger.info("📊 Feature Diversity Analysis:")
    logger.info(f"   Total features: {analysis['total_features']}")
    logger.info(f"   Numeric features: {analysis['numeric_features']}")
    logger.info(f"   Zero variance: {analysis['zero_variance_count']}")
    logger.info(f"   Low variance: {analysis['low_variance_count']}")
    logger.info(f"   Constant features: {analysis['constant_feature_count']}")
    logger.info(f"   Useful features: {analysis['useful_feature_count']}")

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
    logger.info(
        f"🔧 Cleaning features for training: {X.shape[0]} samples, {X.shape[1]} features"
    )

    X_clean = X.copy()
    y_clean = y.copy()
    removed_features = {
        "constant": [],
        "zero_variance": [],
        "high_correlation": [],
        "non_numeric": [],
    }

    # 1. Remove non-numeric columns but preserve essential price columns
    essential_columns = ["close", "ticker_id", "date_int"]

    non_numeric_cols = X_clean.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numeric_to_remove = [
        col for col in non_numeric_cols if col not in essential_columns
    ]

    if non_numeric_to_remove:
        logger.info(
            f"   Removing {len(non_numeric_to_remove)} non-numeric columns (preserving essential price columns)"
        )
        removed_features["non_numeric"] = non_numeric_to_remove
        X_clean = X_clean.drop(columns=non_numeric_to_remove)

    # Ensure essential numeric columns are properly typed
    for col in essential_columns:
        if col in X_clean.columns and X_clean[col].dtype == "object":
            try:
                X_clean[col] = pd.to_numeric(X_clean[col], errors="coerce")
                logger.info(f"   Converted essential column '{col}' to numeric")
            except Exception as e:
                logger.error(
                    f"   Error converting essential column '{col}' to numeric: {e}"
                )

    # 2. Remove constant features
    if remove_constants:
        constant_features = []
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X_clean[col].nunique() <= 1 and col not in essential_columns:
                constant_features.append(col)

        if constant_features:
            logger.info(
                f"   Removing {len(constant_features)} constant features (preserving essential columns)"
            )
            removed_features["constant"] = constant_features
            X_clean = X_clean.drop(columns=constant_features)

    # 3. Remove zero variance features
    if remove_zero_variance:
        numeric_X = X_clean.select_dtypes(include=[np.number])
        if not numeric_X.empty:
            variances = numeric_X.var()
            zero_var_features = [
                col
                for col in variances[variances == 0].index.tolist()
                if col not in essential_columns
            ]
        else:
            zero_var_features = []

        if zero_var_features:
            logger.info(
                f"   Removing {len(zero_var_features)} zero variance features (preserving essential columns)"
            )
            removed_features["zero_variance"] = zero_var_features
            X_clean = X_clean.drop(columns=zero_var_features)

    # 4. Remove highly correlated features (> correlation_threshold)
    if remove_high_correlation:
        numeric_X_remaining = X_clean.select_dtypes(include=[np.number])
        if not numeric_X_remaining.empty and len(numeric_X_remaining.columns) > 1:
            corr_matrix = numeric_X_remaining.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            high_corr_features = [
                column
                for column in upper_triangle.columns
                if any(upper_triangle[column] > correlation_threshold)
                and column not in essential_columns
            ]
        else:
            high_corr_features = []

        if high_corr_features:
            logger.info(
                f"   Removing {len(high_corr_features)} highly correlated features (>{correlation_threshold}, preserving essential columns)"
            )
            removed_features["high_correlation"] = high_corr_features
            X_clean = X_clean.drop(columns=high_corr_features)

    # 5. Remove rows with remaining NaN values in target
    nan_mask = y_clean.notna()
    if not nan_mask.all():
        nan_count = (~nan_mask).sum()
        logger.info(f"   Removing {nan_count} samples with NaN targets")
        X_clean = X_clean[nan_mask]
        y_clean = y_clean[nan_mask]

    # 6. Ensure X and y have same length
    min_length = min(len(X_clean), len(y_clean))
    X_clean = X_clean.iloc[:min_length]
    y_clean = y_clean.iloc[:min_length]

    logger.info(
        f"✅ Feature cleaning completed: {X_clean.shape[0]} samples, {X_clean.shape[1]} features"
    )
    logger.info(
        f"   Removed: {sum(len(v) for v in removed_features.values())} features total"
    )

    return X_clean, y_clean, removed_features


def add_date_features(
    features_df: pd.DataFrame, date_column: str = "date"
) -> pd.DataFrame:
    """
    Add temporal features based on date information

    Args:
        features_df: DataFrame with date column
        date_column: Name of the date column

    Returns:
        DataFrame with additional temporal features
    """

    if date_column not in features_df.columns:
        logger.warning(
            f"Date column '{date_column}' not found. Skipping temporal features."
        )
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

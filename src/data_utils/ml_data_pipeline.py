"""
ML Data Pipeline Module

This module provides the main pipeline functions for preparing data for ML training
and prediction, including the comprehensive data loading, feature engineering,
target transformation, and data cleaning operations.
"""


import pandas as pd
import numpy as np
from pathlib import Path
import time
from typing import Dict, Union, Optional, Tuple
from src.utils.logger import get_logger
from src.data_utils.ml_feature_loader import load_all_data
from src.data_utils.target_engineering import convert_absolute_to_percentage_returns, _maybe_downcast_numeric
from src.data_utils.feature_engineering import (
    add_price_normalized_features,
    add_prediction_bounds_features,
    clean_data_for_training,
    analyze_feature_diversity,
    clean_features_for_training,
    add_date_features,
)
from src.utils.cleaned_data_cache import CleanedDataCache, collect_garbage


logger = get_logger(__name__)

# Global cache instance
_cleaned_data_cache = CleanedDataCache()


def filter_dates_to_weekdays(
    dates: pd.Series, weekdays: Tuple[int, ...]
) -> Tuple[pd.Series, pd.Series]:
    """Filter a date Series to the provided weekdays.

    Args:
        dates: a pandas Series of datetimes (can be full or already subsetted)
        weekdays: tuple of pandas weekday integers (0=Mon .. 6=Sun)

    Returns:
        filtered_mask: boolean Series aligned to `dates` indicating which rows to keep
        filtered_dates: the `dates` Series after filtering (subset)
    """

    if dates.empty:
        return pd.Series(dtype=bool), dates

    # boolean Series aligned to `dates`
    keep = dates.dt.dayofweek.isin(weekdays)

    filtered_dates = dates[keep]

    return keep, filtered_dates






def prepare_ml_data_for_training(
    prediction_horizon: int = 10,
    split_date: str = "2025-06-15",
    ticker: Optional[str] = None,
) -> Dict[str, Union[pd.DataFrame, pd.Series, str]]:
    """
    Comprehensive data preparation function for ML training
    This function loads all available data, prepares features and targets,
    creates temporal train/test splits, and performs data cleaning.
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

        # Early validation: ensure required date column exists before further processing
        if "date" not in combined_data.columns:
            raise ValueError(
                "'date' column not found in data. Cannot perform date-based split."
            )

        logger.info(
            f"✅ Data loaded: {combined_data.shape[0]:,} records, {combined_data.shape[1]} features"
        )

        # 2. Prepare features and targets
        logger.info("2. Preparing features and targets...")

        # Early numeric downcast to reduce peak memory usage on large datasets
        try:
            _maybe_downcast_numeric(combined_data)
            logger.info("✅ Applied early numeric downcast to reduce memory usage")
        except Exception as e:
            logger.warning(f"Early downcast skipped: {e}")

        # PHASE 1 FIX: Convert absolute targets to percentage returns
        combined_data, target_column = convert_absolute_to_percentage_returns(
            combined_data, prediction_horizon
        )

        # Extract percentage return targets (instead of absolute prices)
        y = combined_data[target_column].copy()

        # Prepare features (exclude metadata and target columns)
        exclude_cols = [
            "ticker",
            "date",
            "data_year",
            "feature_version",
            "calculation_date",
            "start_date",
            "end_date",
            "feature_categories",
            "file_path",
            "warnings",
            "quality_score",
            "record_count",
            "total_features",
            "file_size_mb",
        ]
        # Also exclude all Future_* columns to avoid data leakage
        future_cols = [
            col for col in combined_data.columns if col.startswith("Future_")
        ]
        exclude_cols.extend(future_cols)

        feature_cols = [col for col in combined_data.columns if col not in exclude_cols]
        X = combined_data[feature_cols].copy()

        # Persist original X to disk (parquet) instead of keeping full in-memory copy
        transformation_manifest = {}
        try:
            data_folder = Path("checkpoints")
            data_folder.mkdir(parents=True, exist_ok=True)
            x_original_path = data_folder / f"x_original_h{prediction_horizon}_{int(time.time())}.parquet"
            X.to_parquet(x_original_path, index=False)
            transformation_manifest["x_original_path"] = str(x_original_path)
            logger.info(f"Persisted x_original to {x_original_path} to reduce memory")
        except Exception as e:
            logger.warning(f"Failed to persist x_original to disk: {e}")
            transformation_manifest["x_original_path"] = None

        # PHASE 2 FIX: Add price-normalized features
        X = add_price_normalized_features(X)

        # PHASE 2 FIX: Add prediction bounds features
        X = add_prediction_bounds_features(X)

        # 4. Add temporal features

        # Add the date column to features temporarily for temporal feature creation
        X["date"] = combined_data["date"].copy()
        X = add_date_features(X, "date")

        # Remove the date column after temporal features are created
        logger.info(f"📋 Total features: {len(X.columns)}")

        # 5. Clean data
        logger.info("5. Cleaning and preprocessing data...")

        # Remove rows with NaN targets
        valid_mask = y.notna()
        x_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()

        logger.info(f"✅ After target cleaning: {len(x_clean)} valid samples")

        # Replace infinite values with NaN first
        x_clean = x_clean.replace([np.inf, -np.inf], np.nan)

        # Fill NaN with median (more robust than mean) but do it per-column in-place to avoid large temporaries
        numeric_cols = x_clean.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            try:
                med = x_clean[col].median()
                x_clean[col].fillna(med, inplace=True)
            except Exception:
                # fallback to column-wise fillna without raising
                x_clean[col] = x_clean[col].fillna(0)

        # Final safety check - replace any remaining problematic values
        x_clean = x_clean.replace([np.nan, np.inf, -np.inf], 0)

        # 6. Date-based train/test split
        logger.info("6. Creating date-based train/test split...")

        # Check if date column exists
        if "date" not in X.columns:
            raise ValueError(
                "'date' column not found in data. Cannot perform date-based split."
            )

        dates_all = X["date"][valid_mask].copy()
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(dates_all):
            dates_all = pd.to_datetime(dates_all)

        # Define split date
        split_date_dt = pd.to_datetime(split_date)

        # Create train/test masks based on date
        train_mask = dates_all < split_date_dt
        test_mask = dates_all >= split_date_dt

        # Split the data
        x_train = x_clean[train_mask].copy()
        x_test = x_clean[test_mask].copy()
        y_train = y_clean[train_mask].copy()
        y_test = y_clean[test_mask].copy()

        # Apply the same valid_mask to get corresponding dates
        test_dates = dates_all[test_mask].copy()

        filtered_mask, dates_clean = filter_dates_to_weekdays(test_dates, (0, 4))

        # Get test dates for modification and logging
        test_dates = dates_clean
        train_dates = dates_all[train_mask]
        x_test = x_test[filtered_mask]
        y_test = y_test[filtered_mask]

        # Get date ranges for logging
        train_date_range = (
            f"{train_dates.min().strftime('%Y-%m-%d')} to {train_dates.max().strftime('%Y-%m-%d')}"
            if train_mask.any()
            else "No training data"
        )
        test_date_range = (
            f"{test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')}"
            if not test_dates.empty
            else "No test data"
        )

        logger.info(f"✅ Train set: {len(x_train)} samples ({train_date_range})")
        logger.info(f"✅ Test set: {len(x_test)} samples ({test_date_range})")

        # Validation checks
        if len(x_test) == 0:
            raise ValueError(
                f"No test data found after {split_date}. Check your data date range."
            )

        if len(x_train) == 0:
            raise ValueError(
                f"No training data found before {split_date}. Check your data date range."
            )

        # 7. Prepare return dictionary
        result = {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
            # Persisted path to original features instead of full in-memory copy
            "X_original_path": transformation_manifest.get("x_original_path"),
            "target_column": target_column,
            "transformation_manifest": transformation_manifest,
            "train_date_range": train_date_range,
            "test_date_range": test_date_range,
            "feature_count": len(x_train.columns),
            "train_samples": len(x_train),
            "test_samples": len(x_test),
            "prediction_horizon": prediction_horizon,
            "split_date": split_date,
        }

        # 8. Summary logging
        logger.info("=" * 80)
        logger.info("✅ ML DATA PREPARATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"🎯 Target: {target_column} ({prediction_horizon}-day horizon)")
        logger.info(f"📊 Features: {result['feature_count']} total")
        logger.info(f"📏 Split date: {split_date}")
        logger.info("=" * 80)
        collect_garbage()
        return result

    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR in prepare_ml_data_for_training: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


def prepare_ml_data_for_prediction(
    prediction_horizon: int = 10,
) -> Dict[str, Union[pd.DataFrame, pd.Series, str]]:
    """
    Comprehensive data preparation function for ML prediction
    This function loads all available data, prepares features and targets,
    creates temporal train/test splits, and performs data cleaning.
    Args:
        prediction_horizon: Days ahead for target prediction (default: 10)

    Returns:
        Dictionary containing:
        - 'x_test': Test features
        - 'y_test': Test targets
        - 'target_column': Name of target column used
        - 'feature_count': Number of features
    """
    logger.info("=" * 80)
    logger.info("🎯 COMPREHENSIVE ML DATA PREPARATION FOR PREDICTION")
    logger.info("=" * 80)

    try:
        combined_data = load_all_data(ticker=None)

        if combined_data.empty:
            raise ValueError("No data loaded. Check data availability.")

        combined_data, target_column = convert_absolute_to_percentage_returns(
            combined_data, prediction_horizon
        )
        date_col = combined_data["date"].copy()
        # keep Mondays and Fridays only (0, 4)
        filtered_mask, filtered_dates = filter_dates_to_weekdays(date_col, (0, 4))
        combined_data = combined_data[filtered_mask]

        # Extract percentage return targets (instead of absolute prices)
        y = combined_data[target_column].copy()

        # Prepare features (exclude metadata and target columns)
        exclude_cols = [
            "ticker",
            "data_year",  # Metadata
            "feature_version",
            "calculation_date",
            "start_date",
            "end_date",
            "feature_categories",
            "file_path",
            "warnings",  # Feature engineering metadata
            "quality_score",
            "record_count",
            "total_features",
            "file_size_mb",  # Data quality metrics
        ]

        # Also exclude all Future_* columns to avoid data leakage
        future_cols = [
            col for col in combined_data.columns if col.startswith("Future_")
        ]
        exclude_cols.extend(future_cols)

        feature_cols = [col for col in combined_data.columns if col not in exclude_cols]
        X = combined_data[feature_cols].copy()

        X = add_price_normalized_features(X)
        X = add_prediction_bounds_features(X)
        X["date"] = combined_data["date"].copy()
        X = add_date_features(X, "date")

        # Remove the date column after temporal features are created
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        X = X.replace([np.nan, np.inf, -np.inf], 0)

        if not filtered_dates.empty:
            logger.info("📅 Filtering prediction set to include only Fridays/Mondays.")
            X = X[filtered_mask]
            split_date_dt = pd.to_datetime("2025-06-15")
            test_mask = date_col >= split_date_dt
            x_test = X[test_mask].copy()
            y_test = y[test_mask].copy()

        test_date_range = (
            f"{date_col[test_mask].min().strftime('%Y-%m-%d')} to {date_col[test_mask].max().strftime('%Y-%m-%d')}"
            if test_mask.any()
            else "No test data"
        )

        logger.info(f"✅ Test set: {len(x_test)} samples ({test_date_range})")

        # 6. Prepare return dictionary
        result = {
            "x_test": x_test,
            "y_test": y_test,
            "target_column": target_column,
            "test_date_range": test_date_range,
            "feature_count": len(x_test.columns),
            "prediction_horizon": prediction_horizon,
        }

        # 7. Summary logging
        logger.info("=" * 80)
        logger.info("✅ ML DATA PREPARATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"🎯 Target: {target_column} ({prediction_horizon}-day horizon)")
        logger.info(f"📊 Features: {result['feature_count']} total")
        logger.info("=" * 80)
        collect_garbage()
        return result

    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR in prepare_ml_data_for_prediction: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


def prepare_ml_data_for_training_with_cleaning(
    prediction_horizon: int = 10,
    split_date: str = "2025-06-15",
    ticker: str = None,
    clean_features: bool = True,
    _filter_train_set: bool = True,
) -> dict:
    """
    Enhanced version of prepare_ml_data_for_training with integrated data cleaning and caching
    """
    logger.info(
        f"📊 [START] Preparing ML data with cleaning (horizon: {prediction_horizon}d, split: {split_date})"
    )
    # Generate cache key based on parameters
    cache_params = {
        "prediction_horizon": prediction_horizon,
        "split_date": split_date,
        "ticker": ticker,
        "clean_features": clean_features,
        "function": "prepare_ml_data_for_training_with_cleaning",
    }
    cache_key = _cleaned_data_cache._generate_cache_key(**cache_params)
    # Check if cached data exists and is newer than 24 hours
    if _cleaned_data_cache.cache_exists(cache_key, "training"):
        cache_age_hours = _cleaned_data_cache.get_cache_age_hours(cache_key, "training")
        if cache_age_hours is not None and cache_age_hours > 24:
            logger.info(
                f"🗑️ Cache too old ({cache_age_hours:.1f}h), deleting stale cache..."
            )
            _cleaned_data_cache.clear_cache(cache_key, "training")
        else:
            logger.info(
                f"💾 Loading cached cleaned training data (key: {cache_key[:8]}...)"
            )
            try:
                return _cleaned_data_cache.load_cleaned_data(cache_key, "training")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load cached data: {str(e)}")
                logger.info("🔄 Falling back to fresh data preparation...")

    # 1. Use original data preparation function
    logger.info("Step 1: Running original data preparation pipeline...")
    data_result = prepare_ml_data_for_training(
        prediction_horizon=prediction_horizon,
        split_date=split_date,
        ticker=ticker,
    )
    logger.info(
        f"   Loaded: {len(data_result['x_train'])} train, {len(data_result['x_test'])} test samples, {data_result['feature_count']} features"
    )
    # 2. Apply data cleaning (always performed)
    logger.info("Step 2: Applying data cleaning to combined train/test set...")
    combined_x = pd.concat(
        [data_result["x_train"], data_result["x_test"]], ignore_index=True
    )
    combined_y = pd.concat(
        [data_result["y_train"], data_result["y_test"]], ignore_index=True
    )
    combined_x_clean = clean_data_for_training(combined_x)
    logger.info(
        f"   After cleaning: {len(combined_x_clean)} samples, {combined_x_clean.shape[1]} features"
    )
    # Split back into train/test
    train_size = len(data_result["x_train"])
    data_result["x_train"] = combined_x_clean.iloc[:train_size]
    data_result["x_test"] = combined_x_clean.iloc[train_size:]
    data_result["y_train"] = combined_y.iloc[:train_size]
    data_result["y_test"] = combined_y.iloc[train_size:]
    # 3. Apply feature cleaning if requested
    if clean_features:
        logger.info("Step 3: Applying feature cleaning to training set...")
        x_train_clean, y_train_clean, removed_features = clean_features_for_training(
            data_result["x_train"], data_result["y_train"]
        )
        features_to_keep = x_train_clean.columns
        x_test_clean = data_result["x_test"][features_to_keep]
        y_test_clean = data_result["y_test"]
        logger.info(
            f"   After feature cleaning: {len(x_train_clean)} train, {len(x_test_clean)} test samples, {len(features_to_keep)} features"
        )
        data_result["x_train"] = x_train_clean
        data_result["y_train"] = y_train_clean
        data_result["x_test"] = x_test_clean
        data_result["y_test"] = y_test_clean
        data_result["removed_features"] = removed_features
        data_result["feature_count"] = len(features_to_keep)
    # 4. Analyze final feature diversity (always performed after cleaning)
    logger.info("Step 4: Analyzing feature diversity in training set...")
    diversity_analysis = analyze_feature_diversity(data_result["x_train"])
    data_result["diversity_analysis"] = diversity_analysis
    logger.info(
        f"   Diversity: {diversity_analysis['useful_feature_count']} useful, {diversity_analysis['constant_feature_count']} constant, {diversity_analysis['zero_variance_count']} zero-variance features"
    )
    if diversity_analysis["constant_feature_count"] > 10:
        logger.warning(
            f"⚠️ Still {diversity_analysis['constant_feature_count']} constant features after cleaning"
        )
        logger.warning(
            "💡 Consider expanding date range or checking feature engineering"
        )
    # 5. Cache the cleaned data
    try:
        logger.info(
            f"💾 [CACHE] Caching cleaned training data (key: {cache_key[:8]}...)"
        )
        _cleaned_data_cache.save_cleaned_data(data_result, cache_key, "training")
        logger.info("✅ [CACHE] Data cached successfully.")
    except Exception as e:
        logger.warning(f"⚠️ [CACHE] Failed to cache cleaned data: {str(e)}")
    logger.info(
        f"✅ [END] Enhanced data preparation completed: {len(data_result['x_train'])} train, {len(data_result['x_test'])} test samples, {data_result['feature_count']} features"
    )
    collect_garbage()
    return data_result


def prepare_ml_data_for_prediction_with_cleaning(
    prediction_horizon: int = 10, days_back: int = 30
) -> dict:
    """
    Enhanced version of prepare_ml_data_for_prediction with integrated data cleaning and caching
    Args:
        prediction_horizon: Prediction horizon in days
        days_back: Number of days back to load data from

    Returns:
        Dictionary with cleaned prediction data
    """
    logger.info(
        f"📊 Preparing prediction data with cleaning (horizon: {prediction_horizon}d, days_back: {days_back})"
    )

    # Generate cache key based on parameters
    cache_params = {
        "prediction_horizon": prediction_horizon,
        "days_back": days_back,
        "function": "prepare_ml_data_for_prediction_with_cleaning",
    }
    cache_key = _cleaned_data_cache._generate_cache_key(**cache_params)

    # Check if cached data exists and is newer than 24 hours
    if _cleaned_data_cache.cache_exists(cache_key, "prediction"):
        cache_age_hours = _cleaned_data_cache.get_cache_age_hours(
            cache_key, "prediction"
        )
        if cache_age_hours is not None and cache_age_hours > 24:
            logger.info(
                f"🗑️ Cache too old ({cache_age_hours:.1f}h), deleting stale cache..."
            )
            _cleaned_data_cache.clear_cache(cache_key, "prediction")
        else:
            logger.info(
                f"💾 Loading cached cleaned prediction data (key: {cache_key[:8]}...)"
            )
            try:
                return _cleaned_data_cache.load_cleaned_data(cache_key, "prediction")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load cached data: {str(e)}")
                logger.info("🔄 Falling back to fresh data preparation...")

    # 1. Use original data preparation function
    data_result = prepare_ml_data_for_prediction(prediction_horizon=prediction_horizon)

    # 2. Filter to recent data if days_back is specified
    if days_back and "date_int" in data_result["x_test"].columns:
        max_date_int = data_result["x_test"]["date_int"].max()
        cutoff_date_int = max_date_int - days_back

        recent_mask = data_result["x_test"]["date_int"] >= cutoff_date_int
        data_result["x_test"] = data_result["x_test"][recent_mask]
        data_result["y_test"] = data_result["y_test"][recent_mask]

    data_result["x_test"] = clean_data_for_training(data_result["x_test"])

    # Analyze feature diversity for prediction data
    diversity_analysis = analyze_feature_diversity(data_result["x_test"])
    data_result["diversity_analysis"] = diversity_analysis

    # Warning if too many constant features (common in prediction with narrow date range)
    if diversity_analysis["constant_feature_count"] > 10:
        logger.warning(
            f"⚠️ {diversity_analysis['constant_feature_count']} constant features in prediction data"
        )
        logger.warning(
            f"💡 Consider increasing days_back from {days_back} to add date diversity"
        )

    # 4. Cache the cleaned data
    try:
        _cleaned_data_cache.save_cleaned_data(data_result, cache_key, "prediction")
        logger.info("✅ [CACHE] Data cached successfully.")
    except Exception as e:
        logger.warning(f"⚠️ Failed to cache cleaned data: {str(e)}")

    logger.info(
        f"   Prediction data: {len(data_result['x_test'])} samples, {len(data_result['x_test'].columns)} features"
    )
    collect_garbage()
    return data_result

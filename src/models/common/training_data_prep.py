from typing import Any, Dict, Tuple

import pandas as pd

from src.utils.logger import get_logger
from src.data_utils.ml_data_pipeline import (
    prepare_ml_data_for_training_with_cleaning,
)
from src.models.time_series.mlp.mlp_architecture import MLPDataUtils


logger = get_logger(__name__)


def prepare_common_training_data(
    *,
    prediction_horizon: int,
    outlier_quantiles: Tuple[float, float] = (0.05, 0.95),
    recent_date_int_cut: int = 15) -> Dict[str, Any]:
    """
    Centralized data preparation for time-series model training (MLP, RealMLP).
    Steps:
    1) Load and clean data via prepare_ml_data_for_training_with_cleaning
    2) Remove target outliers using quantiles (default middle 95%)
    3) Validate & clean features (basic sanitization)
    4) Optionally remove most recent date_int rows (default last 15 unique values)

    Returns a dict with cleaned X_train, X_test, y_train, y_test and metadata
    from the source loader (target_column, train/test date ranges, etc.).
    """

    # 1) Load base dataset
    data_result = prepare_ml_data_for_training_with_cleaning(
        prediction_horizon=prediction_horizon,
        ticker=None,
        clean_features=True,
    )

    X_train: pd.DataFrame = data_result["X_train"]
    X_test: pd.DataFrame = data_result["X_test"]
    y_train: pd.Series = data_result["y_train"]
    y_test: pd.Series = data_result["y_test"]

    # 2) Remove target outliers
    try:
        logger.info("Removing outliers from target variables (quantile trim)...")
        q_low, q_high = outlier_quantiles
        y_lo = y_train.quantile(q_low)
        y_hi = y_train.quantile(q_high)
        keep_mask = (y_train >= y_lo) & (y_train <= y_hi)
        X_train = X_train[keep_mask]
        y_train = y_train[keep_mask]
        logger.info(
            "Target outlier removal: %d \u2192 %d samples (%d removed)",
            int(len(keep_mask)), int(keep_mask.sum()), int(len(keep_mask) - int(keep_mask.sum())),
        )
        logger.info(
            "   Training target range: %.6f to %.6f (avg: %.6f)",
            float(y_train.min()), float(y_train.max()), float(y_train.mean()),
        )
    except Exception as e:
        logger.warning(f"Outlier removal skipped due to error: {e}")

    # 3) Basic validation and cleaning
    X_train_clean = MLPDataUtils.validate_and_clean_data(X_train)
    X_test_clean = MLPDataUtils.validate_and_clean_data(X_test)

    # 4) Optional date_int trimming on test set
    try:
        if "date_int" in X_test_clean.columns and recent_date_int_cut and recent_date_int_cut > 0:
            uniq = X_test_clean["date_int"].drop_duplicates().sort_values()
            if len(uniq) > recent_date_int_cut:
                threshold = uniq.iloc[-recent_date_int_cut - 1]
                logger.info(f"date_int threshold: {threshold}")
                mask = X_test_clean["date_int"] < threshold
                X_test_clean = X_test_clean[mask]
                y_test = y_test[mask]
                logger.info(f"Removed rows with date_int >= {threshold} (kept {len(X_test_clean)} samples)")
        elif "date_int" not in X_test_clean.columns:
            logger.warning("'date_int' column not found - skipping date filtering")
    except Exception as e:
        logger.warning(f"Date filtering skipped due to error: {e}")

    # Bundle results
    result: Dict[str, Any] = {
        "X_train": X_train_clean,
        "X_test": X_test_clean,
        "y_train": y_train,
        "y_test": y_test,
        "target_column": data_result.get("target_column"),
        "train_date_range": data_result.get("train_date_range"),
        "test_date_range": data_result.get("test_date_range"),
        "original": data_result,
        "params": {
            "prediction_horizon": prediction_horizon,
            "outlier_quantiles": outlier_quantiles,
            "recent_date_int_cut": recent_date_int_cut,
        },
    }

    return result



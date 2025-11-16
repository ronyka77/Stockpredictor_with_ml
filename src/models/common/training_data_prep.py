from typing import Any, Dict, Optional, Tuple
import os
from datetime import datetime

import pandas as pd

from src.utils.core.logger import get_logger
from src.data_utils.ml_data_pipeline import prepare_ml_data_for_training_with_cleaning
from src.models.time_series.mlp.mlp_architecture import MLPDataUtils


logger = get_logger(__name__)


def export_dataset_sample_to_csv(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    target_column: str,
    sample_size: int = 1000,
    predictions_dir: str = "predictions",
    random_state: int = 42,
) -> None:
    """
    Extract random sample from test dataset and export to CSV for inspection/analysis.

    Args:
        x_test: Test feature DataFrame
        y_test: Test target Series
        target_column: Name of the target column
        sample_size: Maximum number of rows to sample (default: 1000)
        predictions_dir: Directory to save the CSV file (default: "predictions")
        random_state: Random state for reproducible sampling (default: 42)
    """
    try:
        logger.info(f"Extracting {sample_size} random rows from test dataset for CSV export...")

        # Create dataset with all columns
        full_dataset = x_test.copy()
        full_dataset[target_column] = y_test

        # Sample up to sample_size random rows
        actual_sample_size = min(sample_size, len(full_dataset))
        random_sample = full_dataset.sample(n=actual_sample_size, random_state=random_state)

        # Create predictions directory if it doesn't exist
        os.makedirs(predictions_dir, exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"dataset_sample_{actual_sample_size}rows_{timestamp}.csv"
        csv_path = os.path.join(predictions_dir, csv_filename)

        # Save to CSV
        random_sample.to_csv(csv_path, index=False)
        logger.info(f"Saved {actual_sample_size} random rows to {csv_path}")
        logger.info(f"Sample shape: {random_sample.shape}")
        logger.info(f"Columns: {list(random_sample.columns)}")

    except Exception as e:
        logger.warning(f"CSV export failed: {e}")


def prepare_common_training_data(
    *,
    prediction_horizon: int,
    outlier_quantiles: Tuple[float, float] = (0.05, 0.95),
    recent_date_int_cut: int = 10,
) -> Dict[str, Any]:
    """
    Centralized data preparation for time-series model training (MLP, RealMLP).
    Steps:
    1) Load and clean data via prepare_ml_data_for_training_with_cleaning
    2) Remove target outliers using quantiles (default middle 95%)
    3) Validate & clean features (basic sanitization)
    4) Optionally remove most recent date_int rows (default last 15 unique values)
    5) Extract 1000 random rows from test dataset to CSV for inspection/analysis

    Returns a dict with cleaned x_train, x_test, y_train, y_test and metadata
    from the source loader (target_column, train/test date ranges, etc.).
    """

    # 1) Load base dataset
    data_result = prepare_ml_data_for_training_with_cleaning(
        prediction_horizon=prediction_horizon, clean_features=True
    )

    x_train: pd.DataFrame = data_result["x_train"]
    x_test: pd.DataFrame = data_result["x_test"]
    y_train: pd.Series = data_result["y_train"]
    y_test: pd.Series = data_result["y_test"]

    # 2) Remove target outliers
    x_train, y_train = _remove_target_outliers(x_train, y_train, outlier_quantiles)

    # 3) Basic validation and cleaning
    x_train_clean = MLPDataUtils.validate_and_clean_data(x_train)
    x_test_clean = MLPDataUtils.validate_and_clean_data(x_test)

    # 4) Optional date_int trimming on test set
    x_test_clean, y_test = _filter_recent_dates(x_test_clean, y_test, recent_date_int_cut)

    # 5) Extract random sample to CSV for inspection/analysis
    # target_col = data_result.get("target_column", "target")
    # export_dataset_sample_to_csv(
    #     x_test=x_test_clean,
    #     y_test=y_test,
    #     target_column=target_col,
    #     sample_size=1000,
    #     predictions_dir="predictions",
    #     random_state=42,
    # )

    # Bundle results
    result: Dict[str, Any] = {
        "x_train": x_train_clean,
        "x_test": x_test_clean,
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


def _remove_target_outliers(
    x_train: pd.DataFrame, y_train: pd.Series, outlier_quantiles: Tuple[float, float]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Remove outliers from target variable using quantile trimming."""
    try:
        logger.info("Removing outliers from target variables (quantile trim)...")
        q_low, q_high = outlier_quantiles
        y_lo = y_train.quantile(q_low)
        y_hi = y_train.quantile(q_high)
        keep_mask = (y_train >= y_lo) & (y_train <= y_hi)
        x_train_filtered = x_train[keep_mask]
        y_train_filtered = y_train[keep_mask]
        logger.info(
            f"Target outlier removal: {int(len(keep_mask))} : {int(keep_mask.sum())} samples ({int(len(keep_mask) - int(keep_mask.sum()))} removed"
        )
        logger.info(
            f"Training target range: {float(y_train_filtered.min()):.6f} to {float(y_train_filtered.max()):.6f} (avg: {float(y_train_filtered.mean()):.6f})"
        )
        return x_train_filtered, y_train_filtered
    except Exception as e:
        logger.warning(f"Outlier removal skipped due to error: {e}")
        return x_train, y_train


def _filter_recent_dates(
    x_test: pd.DataFrame, y_test: pd.Series, recent_date_int_cut: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """Filter out most recent dates from test set."""
    try:
        if "date_int" in x_test.columns and recent_date_int_cut and recent_date_int_cut > 0:
            uniq = x_test["date_int"].drop_duplicates().sort_values()
            if len(uniq) > recent_date_int_cut:
                threshold = uniq.iloc[-recent_date_int_cut - 1]
                logger.info(f"date_int threshold: {threshold}")
                mask = x_test["date_int"] < threshold
                x_test_filtered = x_test[mask]
                y_test_filtered = y_test[mask]

                # Parse threshold as date for logging
                max_tested_date = _parse_date_threshold(threshold)
                logger.info(
                    f"Removed rows with (kept {len(x_test_filtered)} samples), max_tested_date: {max_tested_date}"
                )
                return x_test_filtered, y_test_filtered
        elif "date_int" not in x_test.columns:
            logger.warning("'date_int' column not found - skipping date filtering")
    except Exception as e:
        logger.warning(f"Date filtering skipped due to error: {e}")

    return x_test, y_test


def _parse_date_threshold(threshold: int) -> Optional[str]:
    """Parse date threshold for logging purposes."""
    try:
        # Try interpreting `threshold` as an offset in days from an origin
        max_tested_ts = pd.to_datetime(threshold, unit="D", origin="2020-01-01", errors="coerce")
        if pd.isna(max_tested_ts):
            # Fallback to generic parsing (handles YYYY, YYYYMM, YYYYMMDD, ISO strings)
            max_tested_ts = pd.to_datetime(
                str(threshold), errors="coerce", infer_datetime_format=True
            )

        if pd.isna(max_tested_ts):
            return None
        else:
            return max_tested_ts.strftime("%Y-%m-%d")
    except Exception:
        return None

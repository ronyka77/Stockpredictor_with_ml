"""Training pipeline for the skforecast ensemble.

This script uses `prepare_common_training_data` from the repository to obtain preprocessed
training and validation sets and then fits the ensemble and saves artifacts.

Invoke with `uv run src/pipelines/train_skforecast_ensemble.py` from project root.
"""

import logging
import os
import pandas as pd

from src.models.common.training_data_prep import prepare_common_training_data
from src.models.skforecast.ensemble_forecaster import EnsembleForecaster
from src.models.skforecast.utils import (
    build_last_window_from_series,
    cast_numeric_columns_to_float32,
)
from src.utils.mlops.mlflow_utils import MLFlowManager
from mlflow.models.signature import infer_signature
from datetime import datetime, timezone


logger = logging.getLogger(__name__)


def _prepare_training_data(prediction_horizon: int) -> dict:
    """Load and prepare training data with memory optimization."""
    logger.info("Loading preprocessed training data via prepare_common_training_data...")
    data = prepare_common_training_data(prediction_horizon=prediction_horizon)

    x_train = data["x_train"]
    x_test = data["x_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    # Downcast numeric exogenous columns to float32 to reduce memory usage during training
    if x_train is not None:
        x_train = cast_numeric_columns_to_float32(x_train)
    if x_test is not None:
        x_test = cast_numeric_columns_to_float32(x_test)

    return {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}


def _normalize_data_indexes(
    x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> tuple:
    """Normalize data indexes for skforecast compatibility."""
    try:
        # Handle training data indexes
        if x_train is not None and "date_int" in x_train.columns:
            x_train, y_train = _normalize_single_dataset_indexes(x_train, y_train)

        # Handle test data indexes
        if x_test is not None and "date_int" in x_test.columns:
            x_test, y_test = _normalize_single_dataset_indexes(x_test, y_test)
        else:
            # Ensure y_test has proper index even without date_int column
            if not isinstance(getattr(y_test, "index", None), (pd.DatetimeIndex, pd.RangeIndex)):
                y_test.index = pd.RangeIndex(len(y_test))
    except Exception:
        logger.exception("Failed to normalize train/test indexes; proceeding without changes")

    return x_train, x_test, y_train, y_test


def _normalize_single_dataset_indexes(x_data: pd.DataFrame, y_data: pd.Series) -> tuple:
    """Normalize indexes for a single dataset (train or test)."""
    idx = pd.to_datetime(x_data["date_int"], unit="D", origin="2020-01-01")

    # Try to infer a regular frequency for DatetimeIndex
    freq = pd.infer_freq(idx)
    if freq is not None:
        idx = pd.DatetimeIndex(idx.values, freq=freq)
        x_data = x_data.set_index(idx)
    else:
        # Use RangeIndex for non-regular dates
        x_data.index = pd.RangeIndex(len(x_data))
        idx = None

    # Align y_data to the index
    if hasattr(y_data, "index"):
        if idx is not None and len(idx) == len(y_data):
            y_data.index = idx
        elif idx is not None and len(idx) > len(y_data):
            y_data.index = idx[-len(y_data) :]
        else:
            y_data.index = pd.RangeIndex(len(y_data))
    else:
        y_data = pd.Series(y_data)
        y_data.index = pd.RangeIndex(len(y_data))

    return x_data, y_data


def _setup_mlflow_experiment() -> None:
    """Setup MLflow experiment for ensemble training."""
    try:
        MLFlowManager().setup_experiment("skforecast-ensemble")
    except Exception:
        logger.exception("Failed to setup MLflow experiment; continuing without explicit setup")


def _create_model_signature(
    x_train: pd.DataFrame, y_train: pd.Series, forecaster: EnsembleForecaster
) -> tuple:
    """Create MLflow model signature and input example."""
    # Prepare input example
    try:
        input_example = x_train.iloc[:1]
    except Exception:
        input_example = None

    # Prepare sample output
    try:
        last_window = build_last_window_from_series(y_train, forecaster.lags)
        sample_output = forecaster.predict(last_window=last_window)
    except Exception:
        # Fallback to a small slice of y_train
        sample_output = y_train.iloc[:1]

    # Create signature
    try:
        signature = infer_signature(input_example, sample_output)
    except Exception:
        signature = None

    return signature, input_example


def _save_model_to_mlflow(
    forecaster: EnsembleForecaster, signature, input_example: pd.DataFrame
) -> None:
    """Save trained model to MLflow."""
    run_name = f"ensemble-train-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    try:
        forecaster.save(
            use_mlflow=True,
            registered_model_name="skforecast-ensemble-default",
            signature=signature,
            input_example=input_example,
            run_name=run_name,
        )
    except Exception:
        logger.exception("Failed to save ensemble to MLflow")


def run_train(*, lags: int = 30, steps: int = 20, models_dir: str = "models/skforecast") -> None:
    """Run a full training pass using the repo's standard data prep.

    The function intentionally avoids argument parsing (project rule). Configuration can be
    wired in by editing this file or reading a YAML from `src/configs/skforecast_ensemble.yml`.
    """
    # Prepare training data
    data = _prepare_training_data(steps)
    x_train = data["x_train"]
    x_test = data["x_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    # Ensure model dir exists
    os.makedirs(models_dir, exist_ok=True)

    # Normalize data indexes for skforecast compatibility
    x_train, x_test, y_train, y_test = _normalize_data_indexes(x_train, x_test, y_train, y_test)

    # Setup and train forecaster
    forecaster = EnsembleForecaster(lags=lags, steps=steps, models_dir=models_dir)
    _setup_mlflow_experiment()

    logger.info("Fitting ensemble forecaster (this may take a while)...")
    forecaster.fit(y_train, exog_train=x_train, validation=(y_test, x_test), save_artifacts=True)
    logger.info(f"Training complete — artifacts saved to {models_dir}")

    # Create signature and save to MLflow
    signature, input_example = _create_model_signature(x_train, y_train, forecaster)
    _save_model_to_mlflow(forecaster, signature, input_example)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_train()

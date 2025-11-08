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
from src.models.skforecast.utils import build_last_window_from_series, cast_numeric_columns_to_float32
from src.utils.mlops.mlflow_utils import MLFlowManager
from mlflow.models.signature import infer_signature
from datetime import datetime, timezone


logger = logging.getLogger(__name__)


def run_train(*, lags: int = 30, steps: int = 20, models_dir: str = "models/skforecast") -> None:
    """Run a full training pass using the repo's standard data prep.

    The function intentionally avoids argument parsing (project rule). Configuration can be
    wired in by editing this file or reading a YAML from `src/configs/skforecast_ensemble.yml`.
    """
    logger.info("Loading preprocessed training data via prepare_common_training_data...")
    data = prepare_common_training_data(prediction_horizon=steps)

    x_train = data["x_train"]
    x_test = data["x_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    # Downcast numeric exogenous columns to float32 to reduce memory usage during training
    if x_train is not None:
        x_train = cast_numeric_columns_to_float32(x_train)
    if x_test is not None:
        x_test = cast_numeric_columns_to_float32(x_test)

    # Ensure model dir exists
    os.makedirs(models_dir, exist_ok=True)

    # If a `date_int` column exists in exog, convert it to a DatetimeIndex and
    # align both exog and target series. skforecast requires y to have a
    # DatetimeIndex or RangeIndex.
    try:
        if x_train is not None and "date_int" in x_train.columns:
            idx_train = pd.to_datetime(x_train["date_int"], unit="D", origin="2020-01-01")

            # Try to infer a regular frequency. If present, create a DatetimeIndex
            # with that frequency. For non-regular dates we fall back to RangeIndex
            # for both x and y to avoid pandas frequency validation errors.
            freq = pd.infer_freq(idx_train)
            if freq is not None:
                idx_train = pd.DatetimeIndex(idx_train.values, freq=freq)
                x_train = x_train.set_index(idx_train)
            else:
                # non-regular -> use RangeIndex for alignment
                x_train.index = pd.RangeIndex(len(x_train))
                idx_train = None

            # Align y_train to the datetime index (or to RangeIndex fallback)
            if hasattr(y_train, "index"):
                if idx_train is not None and len(idx_train) == len(y_train):
                    y_train.index = idx_train
                elif idx_train is not None and len(idx_train) > len(y_train):
                    y_train.index = idx_train[-len(y_train) :]
                else:
                    y_train.index = pd.RangeIndex(len(y_train))
            else:
                y_train = pd.Series(y_train)
                y_train.index = pd.RangeIndex(len(y_train))

        if x_test is not None and "date_int" in x_test.columns:
            idx_test = pd.to_datetime(x_test["date_int"], unit="D", origin="2020-01-01")
            freq_t = pd.infer_freq(idx_test)
            if freq_t is not None:
                idx_test = pd.DatetimeIndex(idx_test.values, freq=freq_t)
                x_test = x_test.set_index(idx_test)
            else:
                x_test.index = pd.RangeIndex(len(x_test))
                idx_test = None

            if hasattr(y_test, "index"):
                if idx_test is not None and len(idx_test) == len(y_test):
                    y_test.index = idx_test
                elif idx_test is not None and len(idx_test) > len(y_test):
                    y_test.index = idx_test[-len(y_test) :]
                else:
                    y_test.index = pd.RangeIndex(len(y_test))
            else:
                y_test = pd.Series(y_test)
                y_test.index = pd.RangeIndex(len(y_test))
        else:
            if not isinstance(getattr(y_test, "index", None), (pd.DatetimeIndex, pd.RangeIndex)):
                y_test.index = pd.RangeIndex(len(y_test))
    except Exception:
        logger.exception("Failed to normalize train/test indexes; proceeding without changes")

    forecaster = EnsembleForecaster(lags=lags, steps=steps, models_dir=models_dir)
    # Ensure experiment exists in MLflow
    try:
        MLFlowManager().setup_experiment("skforecast-ensemble")
    except Exception:
        logger.exception("Failed to setup MLflow experiment; continuing without explicit setup")

    logger.info("Fitting ensemble forecaster (this may take a while)...")
    forecaster.fit(y_train, exog_train=x_train, validation=(y_test, x_test), save_artifacts=True)
    logger.info("Training complete — artifacts saved to %s", models_dir)

    # Prepare a small input_example (one-row exog) and sample output for signature
    try:
        input_example = x_train.iloc[:1]
    except Exception:
        input_example = None

    try:
        last_window = build_last_window_from_series(y_train, forecaster.lags)
        sample_output = forecaster.predict(last_window=last_window)
    except Exception:
        # Fallback to a small slice of y_train
        sample_output = y_train.iloc[:1]

    try:
        signature = infer_signature(input_example, sample_output)
    except Exception:
        signature = None

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_train()

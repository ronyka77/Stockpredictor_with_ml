"""
AutoGluon training entrypoint

Uses prepare_ml_data_for_training_with_cleaning to obtain x_train/y_train and
x_test/y_test, constructs train_df/valid_df with `Future_Return_XD`, trains an
AutoGluonModel, and logs artifacts/metrics to MLflow.

Run:
uv run python -m src.models.automl.autogluon_trainer --preset high_quality
"""

import contextlib
import os
import json
import psutil
import gc
import glob
import joblib
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Any, Dict, Tuple, Optional

from src.utils.logger import get_logger
from src.models.common.training_data_prep import prepare_common_training_data
from src.models.automl.autogluon_model import AutoGluonModel
from src.models.automl.autogluen_load_model import run_model_evaluation
from src.models.feature_selection.io import load_encoder
from src.models.feature_selection.autoencoder import encode_df


logger = get_logger(__name__)


def filter_to_selected_features(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Filter training and test datasets to only include features selected by AutoGluon.

    Args:
        x_train: Training feature DataFrame
        x_test: Test feature DataFrame

    Returns:
        Tuple of (filtered_x_train, filtered_x_test, metadata_dict)
    """
    # Always locate the most recent selection file in predictions/
    selected_features_path = _find_latest_artifact("predictions/selected_features_*.json")
    essential_columns = ["close", "ticker_id", "date_int"]

    if selected_features_path and os.path.exists(selected_features_path):
        try:
            with open(selected_features_path, "r") as f:
                selected_features_data = json.load(f)
                selected_features = selected_features_data["selected_features"]

            # Filter to only include selected features that exist in the dataset
            available_features = [col for col in selected_features if col in x_train.columns]
            # INSERT_YOUR_CODE
            # Ensure essential columns are present in available_features list (if available in x_train)
            for essential_col in essential_columns:
                if essential_col not in available_features:
                    available_features.append(essential_col)

            if len(available_features) != len(selected_features):
                missing_features = [col for col in selected_features if col not in x_train.columns]
                logger.warning(f"Missing features in dataset: {missing_features}")

            x_train_filtered = x_train[available_features]
            x_test_filtered = x_test[available_features]

            logger.info(
                f"Filtered to {len(available_features)} selected features,x_train shape: {x_train_filtered.shape}, x_test shape: {x_test_filtered.shape}"
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error loading selected features file: {e}. Using all features.")
            x_train_filtered, x_test_filtered = x_train, x_test
            available_features = list(x_train.columns)
    else:
        logger.warning(
            f"Selected features file not found: {selected_features_path}. Using all features."
        )
        x_train_filtered, x_test_filtered = x_train, x_test
        available_features = list(x_train.columns)

    return x_train_filtered, x_test_filtered, selected_features_path


def top_dataframes(top_n: int = 10):
    dfs = []
    logger.info("TOP dataframes:")
    for obj in gc.get_objects():
        with contextlib.suppress(Exception):
            if isinstance(obj, pd.DataFrame):
                size = obj.memory_usage(deep=True).sum()
                dfs.append((size, obj))
    dfs.sort(reverse=True, key=lambda x: x[0])
    for size, df in dfs[:top_n]:
        logger.info(f"{size / 1024**2:8.2f} MB | shape={df.shape} | columns={len(df.columns)}")
    return dfs


def _find_latest_artifact(pattern: str) -> Optional[str]:
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def apply_saved_encoder_and_scaler(
    x_train: pd.DataFrame, x_test: pd.DataFrame, batch_size: int = 2048, device: str = "cpu"
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[torch.nn.Module], Optional[StandardScaler]]:
    # locate and load latest artifacts
    encoder = None
    scaler = None
    encoder_pattern = "predictions/encoder_*.pt"
    encoder_meta_pattern = "predictions/encoder_meta_*.json"
    scaler_pattern = "predictions/scaler_*.joblib"

    enc_path = _find_latest_artifact(encoder_pattern)
    enc_meta = _find_latest_artifact(encoder_meta_pattern)
    scal_path = _find_latest_artifact(scaler_pattern)

    if enc_path and enc_meta:
        try:
            encoder = load_encoder(enc_path, enc_meta, device=device)
            logger.info(f"Loaded encoder from {enc_path} and metadata {enc_meta}")
        except Exception as e:
            logger.warning(f"Failed to load encoder: {e}")

    if scal_path:
        try:
            scaler = joblib.load(scal_path)
            logger.info(f"Loaded scaler from {scal_path}")
        except Exception as e:
            logger.warning(f"Failed to load scaler: {e}")

    used_encoding = False
    if encoder is None or scaler is None:
        return x_train, x_test, used_encoding

    try:
        logger.info(
            "Applying saved scaler and encoder to training and test data before training AutoGluon"
        )

        # Align DataFrame columns to scaler's expected feature names when available
        expected = getattr(scaler, "feature_names_in_", None)
        if expected is not None:
            expected = list(expected)
            missing = [c for c in expected if c not in x_train.columns]
            extra = [c for c in x_train.columns if c not in expected]
            if missing:
                logger.warning(
                    f"Scaler expects {len(missing)} missing cols: {missing}; filling with 0"
                )
                for c in missing:
                    x_train[c] = 0.0
                    x_test[c] = 0.0
            if extra:
                logger.warning(f"Dropping {len(extra)} extra cols: {extra}")
                x_train = x_train.drop(columns=extra)
                x_test = x_test.drop(columns=extra)
            # Reorder to match scaler
            x_train = x_train[expected]
            x_test = x_test[expected]
        else:
            n_in = getattr(scaler, "n_features_in_", None)
            if n_in is not None and n_in != x_train.shape[1]:
                raise ValueError(f"Scaler expects {n_in} features but input has {x_train.shape[1]}")

        x_train_scaled = scaler.transform(x_train.values.astype(float))
        x_test_scaled = scaler.transform(x_test.values.astype(float))
        x_enc_train = encode_df(encoder, x_train_scaled, batch_size=batch_size, device=device)
        x_enc_test = encode_df(encoder, x_test_scaled, batch_size=batch_size, device=device)

        enc_cols = [f"enc_{i}" for i in range(x_enc_train.shape[1])]
        x_train_enc = pd.DataFrame(x_enc_train, columns=enc_cols, index=x_train.index)
        x_test_enc = pd.DataFrame(x_enc_test, columns=enc_cols, index=x_test.index)
        logger.info(f"Using encoded features for AutoGluon training: {len(enc_cols)} dims")
        return x_train_enc, x_test_enc, encoder, scaler
    except Exception as e:
        logger.warning(
            f"Failed to apply encoder/scaler pre-transforms; falling back to original features: {e}"
        )
        return x_train, x_test, None, None


def train_autogluon(
    *, prediction_horizon: int = 20, presets: str = "best_quality"
) -> Dict[str, Any]:
    # Use centralized common training data preparation
    data = prepare_common_training_data(
        prediction_horizon=prediction_horizon, recent_date_int_cut=10
    )
    logger.info(f"target_column: {data.get('target_column')}")
    x_train: pd.DataFrame = data["x_train"]
    logger.info(f"x_train: {x_train.columns.tolist()}")
    y_train: pd.Series = data["y_train"]
    x_test: pd.DataFrame = data["x_test"]
    y_test: pd.Series = data["y_test"]

    # Filter datasets to only include selected features (auto-locate selection file if present)
    x_train, x_test, selected_features_path = filter_to_selected_features(x_train, x_test)

    # Try to apply saved encoder and scaler to produce encoded features before training
    # x_train, x_test, encoder, scaler = apply_saved_encoder_and_scaler(x_train, x_test, batch_size=2048, device="cpu")

    collected = gc.collect()
    logger.info(f"Garbage collected: {collected}")
    proc = psutil.Process(os.getpid())
    print("RSS MB:", proc.memory_info().rss / 1024**2)
    top_dataframes(10)
    # Build model
    config = {"label": f"Future_Return_{prediction_horizon}D", "presets": presets, "groups": "year"}
    model = AutoGluonModel(model_name="autogluon", config=config)
    model.fit(x_train, y_train, X_val=x_test, y_val=y_test)

    model_dir = model.predictor.path
    logger.info(f"Model directory: {model_dir}")
    run_model_evaluation(model_dir, prediction_horizon)

    return {
        "model": model,
        "feature_names": model.feature_names,
        "selected_features_path": selected_features_path,
    }


def main():
    prediction_horizon = 20
    train_autogluon(prediction_horizon=prediction_horizon, presets="best_quality")


if __name__ == "__main__":
    main()
